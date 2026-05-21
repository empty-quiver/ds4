#define DS4_NO_GPU 1
#include "../ds4.c"

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

static void bench_fill_q2_rows(block_q2_K *rows, uint32_t n_rows, uint32_t blocks, uint64_t *rng) {
    for (uint32_t r = 0; r < n_rows; r++) {
        for (uint32_t b = 0; b < blocks; b++) {
            block_q2_K *x = rows + (uint64_t)r * blocks + b;
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

static double bench_baseline(
        const block_q2_K *rows,
        const block_q8_K *xq,
        float            *out,
        uint32_t          rows_n,
        uint32_t          pairs,
        uint32_t          blocks,
        uint32_t          iters) {
    const double t0 = now_sec();
    for (uint32_t it = 0; it < iters; it++) {
        for (uint32_t r = 0; r < rows_n; r++) {
            const block_q2_K *row = rows + (uint64_t)r * blocks;
            for (uint32_t p = 0; p < pairs; p++) {
                ds4_vec_dot_q2_K_q8_K((int)(blocks * QK_K),
                                      out + (uint64_t)r * pairs + p,
                                      row,
                                      xq + (uint64_t)p * blocks);
            }
        }
    }
    return now_sec() - t0;
}

static double bench_panel4(
        const block_q2_K *rows,
        const block_q8_K *xq,
        float            *out,
        uint32_t          rows_n,
        uint32_t          pairs,
        uint32_t          blocks,
        uint32_t          iters) {
    const double t0 = now_sec();
    for (uint32_t it = 0; it < iters; it++) {
        for (uint32_t r = 0; r < rows_n; r++) {
            const block_q2_K *row = rows + (uint64_t)r * blocks;
            uint32_t p = 0;
            for (; p + 3 < pairs; p += 4) {
                float v[4];
                ds4_vec_dot_q2_K_panel4_q8_K((int)(blocks * QK_K),
                                             v,
                                             row,
                                             xq + (uint64_t)(p + 0) * blocks,
                                             xq + (uint64_t)(p + 1) * blocks,
                                             xq + (uint64_t)(p + 2) * blocks,
                                             xq + (uint64_t)(p + 3) * blocks);
                for (uint32_t k = 0; k < 4; k++) {
                    out[(uint64_t)r * pairs + p + k] = v[k];
                }
            }
            for (; p < pairs; p++) {
                ds4_vec_dot_q2_K_q8_K((int)(blocks * QK_K),
                                      out + (uint64_t)r * pairs + p,
                                      row,
                                      xq + (uint64_t)p * blocks);
            }
        }
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
    uint32_t rows = 1024;
    uint32_t pairs = 16;
    uint32_t dim = DS4_N_FF_EXP;
    uint32_t iters = 8;

    for (int i = 1; i < argc; i++) {
        rows = bench_parse_u32(&i, argc, argv, "--rows", rows);
        pairs = bench_parse_u32(&i, argc, argv, "--pairs", pairs);
        dim = bench_parse_u32(&i, argc, argv, "--dim", dim);
        iters = bench_parse_u32(&i, argc, argv, "--iters", iters);
        if (strcmp(argv[i], "--help") == 0) {
            fprintf(stderr, "usage: %s [--rows N] [--pairs N] [--dim N] [--iters N]\n", argv[0]);
            return 0;
        }
    }
    if (dim % QK_K != 0) ds4_die("microbench dim must be QK_K aligned");

    const uint32_t blocks = dim / QK_K;
    const uint64_t out_elems = (uint64_t)rows * pairs;
    uint64_t rng = UINT64_C(0xd1b54a32d192ed03);

    block_q2_K *down = xmalloc((size_t)((uint64_t)rows * blocks) * sizeof(down[0]));
    block_q8_K *xq = xmalloc((size_t)((uint64_t)pairs * blocks) * sizeof(xq[0]));
    float *base = xmalloc((size_t)out_elems * sizeof(base[0]));
    float *panel = xmalloc((size_t)out_elems * sizeof(panel[0]));

    bench_fill_q2_rows(down, rows, blocks, &rng);
    bench_fill_q8_rows(xq, pairs, dim, &rng);

    (void)bench_baseline(down, xq, base, rows, pairs, blocks, 1);
    (void)bench_panel4(down, xq, panel, rows, pairs, blocks, 1);

    double max_abs = 0.0;
    for (uint64_t i = 0; i < out_elems; i++) {
        const double d = fabs((double)base[i] - (double)panel[i]);
        if (d > max_abs) max_abs = d;
    }

    const double tb = bench_baseline(down, xq, base, rows, pairs, blocks, iters);
    const double tp = bench_panel4(down, xq, panel, rows, pairs, blocks, iters);
    const double dots = (double)rows * (double)pairs * (double)iters;
    const double checksum = bench_checksum(base, out_elems) + bench_checksum(panel, out_elems);

    printf("rows=%u pairs=%u dim=%u iters=%u avx2=%s\n",
           rows, pairs, dim, iters,
#if defined(__AVX2__)
           "yes"
#else
           "no"
#endif
    );
    printf("correctness max_abs_diff=%.9g checksum=%.9f\n", max_abs, checksum);
    printf("baseline seconds=%.6f dots_per_s=%.3f\n", tb, dots / tb);
    printf("panel4   seconds=%.6f dots_per_s=%.3f speedup=%.3fx\n", tp, dots / tp, tb / tp);

    free(panel);
    free(base);
    free(xq);
    free(down);
    return max_abs < 1.0e-3 ? 0 : 1;
}
