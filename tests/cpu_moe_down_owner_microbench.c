#define DS4_NO_GPU 1
#include "../ds4.c"

typedef struct {
    const uint8_t *base[DS4_N_EXPERT];
    const block_q8_K *midq;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    float *partial;
    float *out;
    uint64_t out_elems;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t row_bytes[DS4_N_EXPERT];
    uint64_t midq_blocks;
    uint32_t n_active;
    uint32_t n_tok;
    uint32_t n_threads;
} bench_expert_down_ctx;

static uint64_t bench_rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * UINT64_C(2685821657736338717);
}

static float bench_rng_f32(uint64_t *state) {
    const uint32_t bits = (uint32_t)(bench_rng_next(state) >> 40);
    return ((float)bits / 8388608.0f) - 1.0f;
}

static uint32_t bench_parse_u32(int *i, int argc, char **argv, const char *name, uint32_t value) {
    if (strcmp(argv[*i], name) != 0) return value;
    if (*i + 1 >= argc) {
        fprintf(stderr, "%s requires a value\n", name);
        exit(2);
    }
    char *end = NULL;
    unsigned long v = strtoul(argv[++(*i)], &end, 10);
    if (end == argv[*i] || *end != '\0' || v == 0 || v > UINT32_MAX) {
        fprintf(stderr, "invalid value for %s: %s\n", name, argv[*i]);
        exit(2);
    }
    return (uint32_t)v;
}

static void bench_fill_q2_rows(block_q2_K *rows, uint64_t n_rows, uint32_t blocks, uint64_t *rng) {
    for (uint64_t r = 0; r < n_rows; r++) {
        for (uint32_t b = 0; b < blocks; b++) {
            block_q2_K *x = rows + r * blocks + b;
            x->d = f32_to_f16(0.02f + 0.08f * fabsf(bench_rng_f32(rng)));
            x->dmin = f32_to_f16(0.01f + 0.04f * fabsf(bench_rng_f32(rng)));
            for (uint32_t i = 0; i < sizeof(x->scales); i++) {
                x->scales[i] = (uint8_t)bench_rng_next(rng);
            }
            for (uint32_t i = 0; i < sizeof(x->qs); i++) {
                x->qs[i] = (uint8_t)bench_rng_next(rng);
            }
        }
    }
}

static void bench_fill_q8_rows(block_q8_K *rows, uint32_t n_rows, uint32_t dim, uint64_t *rng) {
    float *tmp = xmalloc((size_t)dim * sizeof(tmp[0]));
    const uint32_t blocks = dim / QK_K;
    for (uint32_t r = 0; r < n_rows; r++) {
        for (uint32_t i = 0; i < dim; i++) {
            tmp[i] = bench_rng_f32(rng);
        }
        ds4_quantize_row_q8_K(tmp, rows + (uint64_t)r * blocks, dim);
    }
    free(tmp);
}

static void bench_build_uniform_routes(
        uint32_t  n_tok,
        uint32_t  n_active,
        uint32_t *active_expert,
        uint32_t *expert_offset,
        uint32_t *pair_ids,
        uint64_t *rng) {
    const uint32_t total_pairs = n_tok * DS4_N_EXPERT_USED;
    uint32_t *cursor = xcalloc(DS4_N_EXPERT, sizeof(cursor[0]));
    uint32_t *pair_expert = xmalloc((size_t)total_pairs * sizeof(pair_expert[0]));

    memset(expert_offset, 0, (DS4_N_EXPERT + 1) * sizeof(expert_offset[0]));
    for (uint32_t ai = 0; ai < n_active; ai++) active_expert[ai] = ai;

    for (uint32_t pair_id = 0; pair_id < total_pairs; pair_id++) {
        const uint32_t expert = (uint32_t)(bench_rng_next(rng) % n_active);
        pair_expert[pair_id] = expert;
        expert_offset[expert + 1]++;
    }
    for (uint32_t e = 0; e < DS4_N_EXPERT; e++) {
        expert_offset[e + 1] += expert_offset[e];
        cursor[e] = expert_offset[e];
    }
    for (uint32_t pair_id = 0; pair_id < total_pairs; pair_id++) {
        const uint32_t expert = pair_expert[pair_id];
        pair_ids[cursor[expert]++] = pair_id;
    }

    free(pair_expert);
    free(cursor);
}

static double bench_row_owned(
        matvec_q2_k_batch_accum_rows_ctx *ctx,
        uint32_t row_chunk,
        uint32_t iters) {
    const double t0 = now_sec();
    for (uint32_t it = 0; it < iters; it++) {
        ds4_parallel_for_dynamic_min_rows(ctx->out_dim,
                                          matvec_q2_k_batch_accum_rows_worker,
                                          ctx,
                                          row_chunk,
                                          512);
    }
    return now_sec() - t0;
}

static void bench_expert_owned_worker(void *vctx, uint64_t lane0, uint64_t lane1) {
    bench_expert_down_ctx *ctx = vctx;

    for (uint64_t lane = lane0; lane < lane1; lane++) {
        if (lane >= ctx->n_threads) return;
        float *partial = ctx->partial + lane * ctx->out_elems;
        memset(partial, 0, (size_t)ctx->out_elems * sizeof(partial[0]));

        for (uint32_t ai = (uint32_t)lane; ai < ctx->n_active; ai += ctx->n_threads) {
            const uint32_t expert = ctx->active_expert[ai];
            const uint32_t begin = ctx->expert_offset[expert];
            const uint32_t end = ctx->expert_offset[expert + 1];

            for (uint64_t row = 0; row < ctx->out_dim; row++) {
                const block_q2_K *br =
                    (const block_q2_K *)(ctx->base[expert] + row * ctx->row_bytes[expert]);

                uint32_t i = begin;
                for (; i + 3 < end; i += 4) {
                    const uint32_t pair_id0 = ctx->pair_ids[i + 0];
                    const uint32_t pair_id1 = ctx->pair_ids[i + 1];
                    const uint32_t pair_id2 = ctx->pair_ids[i + 2];
                    const uint32_t pair_id3 = ctx->pair_ids[i + 3];
                    const block_q8_K *xq0 = ctx->midq + (uint64_t)pair_id0 * ctx->midq_blocks;
                    const block_q8_K *xq1 = ctx->midq + (uint64_t)pair_id1 * ctx->midq_blocks;
                    const block_q8_K *xq2 = ctx->midq + (uint64_t)pair_id2 * ctx->midq_blocks;
                    const block_q8_K *xq3 = ctx->midq + (uint64_t)pair_id3 * ctx->midq_blocks;
                    float v[4];

                    ds4_vec_dot_q2_K_panel4_q8_K((int)ctx->in_dim,
                                                 v,
                                                 br,
                                                 xq0,
                                                 xq1,
                                                 xq2,
                                                 xq3);

                    const uint32_t pair_ids[4] = { pair_id0, pair_id1, pair_id2, pair_id3 };
                    for (uint32_t k = 0; k < 4; k++) {
                        const uint32_t token = pair_ids[k] / DS4_N_EXPERT_USED;
                        partial[(uint64_t)token * ctx->out_dim + row] += v[k];
                    }
                }

                for (; i + 1 < end; i += 2) {
                    const uint32_t pair_id0 = ctx->pair_ids[i + 0];
                    const uint32_t pair_id1 = ctx->pair_ids[i + 1];
                    const block_q8_K *xq0 = ctx->midq + (uint64_t)pair_id0 * ctx->midq_blocks;
                    const block_q8_K *xq1 = ctx->midq + (uint64_t)pair_id1 * ctx->midq_blocks;
                    float v[2];

                    ds4_vec_dot_q2_K_panel2_q8_K((int)ctx->in_dim,
                                                 v,
                                                 br,
                                                 xq0,
                                                 xq1);

                    const uint32_t pair_ids[2] = { pair_id0, pair_id1 };
                    for (uint32_t k = 0; k < 2; k++) {
                        const uint32_t token = pair_ids[k] / DS4_N_EXPERT_USED;
                        partial[(uint64_t)token * ctx->out_dim + row] += v[k];
                    }
                }

                for (; i < end; i++) {
                    const uint32_t pair_id = ctx->pair_ids[i];
                    const uint32_t token = pair_id / DS4_N_EXPERT_USED;
                    const block_q8_K *xq = ctx->midq + (uint64_t)pair_id * ctx->midq_blocks;
                    float v = 0.0f;
                    ds4_vec_dot_q2_K_q8_K((int)ctx->in_dim, &v, br, xq);
                    partial[(uint64_t)token * ctx->out_dim + row] += v;
                }
            }
        }
    }
}

static void bench_reduce_partials_worker(void *vctx, uint64_t idx0, uint64_t idx1) {
    bench_expert_down_ctx *ctx = vctx;
    for (uint64_t idx = idx0; idx < idx1; idx++) {
        float sum = 0.0f;
        for (uint32_t lane = 0; lane < ctx->n_threads; lane++) {
            sum += ctx->partial[(uint64_t)lane * ctx->out_elems + idx];
        }
        ctx->out[idx] = sum;
    }
}

static double bench_expert_owned(
        bench_expert_down_ctx *ctx,
        uint32_t iters) {
    const double t0 = now_sec();
    for (uint32_t it = 0; it < iters; it++) {
        ds4_parallel_for_min_rows(ctx->n_threads, bench_expert_owned_worker, ctx, 1);
        ds4_parallel_for_dynamic_min_rows(ctx->out_elems,
                                          bench_reduce_partials_worker,
                                          ctx,
                                          4096,
                                          4096);
    }
    return now_sec() - t0;
}

static double bench_checksum(const float *a, uint64_t n) {
    double sum = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        sum += (double)a[i] * 0.0000003;
    }
    return sum;
}

int main(int argc, char **argv) {
    uint32_t n_tok = 512;
    uint32_t n_active = 192;
    uint32_t in_dim = DS4_N_FF_EXP;
    uint32_t out_dim = DS4_N_EMBD;
    uint32_t row_chunk = 8;
    uint32_t threads = 16;
    uint32_t iters = 3;

    for (int i = 1; i < argc; i++) {
        n_tok = bench_parse_u32(&i, argc, argv, "--tokens", n_tok);
        n_active = bench_parse_u32(&i, argc, argv, "--active", n_active);
        in_dim = bench_parse_u32(&i, argc, argv, "--in-dim", in_dim);
        out_dim = bench_parse_u32(&i, argc, argv, "--out-dim", out_dim);
        row_chunk = bench_parse_u32(&i, argc, argv, "--row-chunk", row_chunk);
        threads = bench_parse_u32(&i, argc, argv, "--threads", threads);
        iters = bench_parse_u32(&i, argc, argv, "--iters", iters);
        if (strcmp(argv[i], "--help") == 0) {
            fprintf(stderr,
                    "usage: %s [--tokens N] [--active N] [--in-dim N] [--out-dim N] [--row-chunk N] [--threads N] [--iters N]\n",
                    argv[0]);
            return 0;
        }
    }
    if (n_active > DS4_N_EXPERT) ds4_die("active expert count exceeds DS4_N_EXPERT");
    if (in_dim % QK_K != 0) ds4_die("input dim must be QK_K aligned");
    if (threads > DS4_MAX_THREADS) ds4_die("thread count exceeds DS4_MAX_THREADS");

    g_requested_threads = (int)threads;
    ds4_threads_init();
    threads = g_pool.n_threads;

    const uint32_t total_pairs = n_tok * DS4_N_EXPERT_USED;
    const uint32_t blocks = in_dim / QK_K;
    const uint64_t out_elems = (uint64_t)n_tok * out_dim;
    const uint64_t weight_rows = (uint64_t)n_active * out_dim;
    uint64_t rng = UINT64_C(0x56f00834d91a73d5);

    uint32_t *active_expert = xmalloc((size_t)n_active * sizeof(active_expert[0]));
    uint32_t *expert_offset = xmalloc((DS4_N_EXPERT + 1) * sizeof(expert_offset[0]));
    uint32_t *pair_ids = xmalloc((size_t)total_pairs * sizeof(pair_ids[0]));
    block_q2_K *weights = xmalloc((size_t)(weight_rows * blocks) * sizeof(weights[0]));
    block_q8_K *midq = xmalloc((size_t)((uint64_t)total_pairs * blocks) * sizeof(midq[0]));
    float *row_out = xmalloc((size_t)out_elems * sizeof(row_out[0]));
    float *expert_out = xmalloc((size_t)out_elems * sizeof(expert_out[0]));
    float *partial = xmalloc((size_t)((uint64_t)threads * out_elems) * sizeof(partial[0]));

    bench_build_uniform_routes(n_tok, n_active, active_expert, expert_offset, pair_ids, &rng);
    bench_fill_q2_rows(weights, weight_rows, blocks, &rng);
    bench_fill_q8_rows(midq, total_pairs, in_dim, &rng);

    matvec_q2_k_batch_accum_rows_ctx row_ctx = {
        .moe = row_out,
        .midq = midq,
        .pair_ids = pair_ids,
        .expert_offset = expert_offset,
        .active_expert = active_expert,
        .n_active = n_active,
        .n_pairs = total_pairs,
        .n_tok = n_tok,
        .in_dim = in_dim,
        .out_dim = out_dim,
        .midq_blocks = blocks,
    };
    bench_expert_down_ctx expert_ctx = {
        .midq = midq,
        .pair_ids = pair_ids,
        .expert_offset = expert_offset,
        .active_expert = active_expert,
        .partial = partial,
        .out = expert_out,
        .out_elems = out_elems,
        .in_dim = in_dim,
        .out_dim = out_dim,
        .midq_blocks = blocks,
        .n_active = n_active,
        .n_tok = n_tok,
        .n_threads = threads,
    };

    for (uint32_t ai = 0; ai < n_active; ai++) {
        const uint32_t expert = active_expert[ai];
        const uint8_t *base = (const uint8_t *)(weights + (uint64_t)ai * out_dim * blocks);
        row_ctx.base[expert] = base;
        row_ctx.row_bytes[expert] = (uint64_t)blocks * sizeof(block_q2_K);
        expert_ctx.base[expert] = base;
        expert_ctx.row_bytes[expert] = (uint64_t)blocks * sizeof(block_q2_K);
    }

    (void)bench_row_owned(&row_ctx, row_chunk, 1);
    (void)bench_expert_owned(&expert_ctx, 1);

    double max_abs = 0.0;
    for (uint64_t i = 0; i < out_elems; i++) {
        const double d = fabs((double)row_out[i] - (double)expert_out[i]);
        if (d > max_abs) max_abs = d;
    }

    const double row_seconds = bench_row_owned(&row_ctx, row_chunk, iters);
    const double expert_seconds = bench_expert_owned(&expert_ctx, iters);
    const double dots = (double)total_pairs * (double)out_dim * (double)iters;
    const double partial_mb = (double)((uint64_t)threads * out_elems * sizeof(float)) / (1024.0 * 1024.0);
    const double output_mb = (double)(out_elems * sizeof(float)) / (1024.0 * 1024.0);
    const double checksum = bench_checksum(row_out, out_elems) + bench_checksum(expert_out, out_elems);

    printf("tokens=%u active=%u pairs=%u in_dim=%u out_dim=%u threads=%u row_chunk=%u iters=%u avx2=%s\n",
           n_tok, n_active, total_pairs, in_dim, out_dim, threads, row_chunk, iters,
#if defined(__AVX2__)
           "yes"
#else
           "no"
#endif
    );
    printf("memory partial_mb=%.1f output_mb=%.1f\n", partial_mb, output_mb);
    printf("correctness max_abs_diff=%.9g checksum=%.9f\n", max_abs, checksum);
    printf("row_owned    seconds=%.6f dots_per_s=%.3f\n", row_seconds, dots / row_seconds);
    printf("expert_owned seconds=%.6f dots_per_s=%.3f speedup=%.3fx\n",
           expert_seconds, dots / expert_seconds, row_seconds / expert_seconds);

    free(partial);
    free(expert_out);
    free(row_out);
    free(midq);
    free(weights);
    free(pair_ids);
    free(expert_offset);
    free(active_expert);
    ds4_threads_shutdown();
    return max_abs < 1.0e-2 ? 0 : 1;
}
