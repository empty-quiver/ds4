#define DS4_NO_GPU 1
#include "../ds4.c"

typedef struct {
    double total_seconds;
    double setup_seconds;
    double xq_seconds;
    double gate_up_seconds;
    double midq_seconds;
    double down_seconds;
} decode_stage_stats;

typedef struct {
    ds4_model model;
    ds4_layer_weights layer;
    ds4_tensor gate;
    ds4_tensor up;
    ds4_tensor down;
    uint8_t *map;
} synthetic_decode_model;

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
    if (end == argv[*i] || *end != '\0' || v > UINT32_MAX) {
        fprintf(stderr, "invalid value for %s: %s\n", name, argv[*i]);
        exit(2);
    }
    return (uint32_t)v;
}

static const char *bench_parse_str(int *i, int argc, char **argv, const char *name, const char *value) {
    if (strcmp(argv[*i], name) != 0) return value;
    if (*i + 1 >= argc) {
        fprintf(stderr, "%s requires a value\n", name);
        exit(2);
    }
    return argv[++(*i)];
}

static void bench_parse_selected(const char *s, int32_t selected[DS4_N_EXPERT_USED], uint32_t *n_selected) {
    if (!s || !s[0]) return;
    uint32_t n = 0;
    const char *p = s;
    while (*p) {
        if (n >= DS4_N_EXPERT_USED) {
            fprintf(stderr, "--selected accepts at most %u experts\n", DS4_N_EXPERT_USED);
            exit(2);
        }
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p || v < 0 || v >= DS4_N_EXPERT) {
            fprintf(stderr, "invalid expert id in --selected: %s\n", p);
            exit(2);
        }
        selected[n++] = (int32_t)v;
        if (*end == '\0') break;
        if (*end != ',') {
            fprintf(stderr, "expected comma in --selected: %s\n", end);
            exit(2);
        }
        p = end + 1;
    }
    if (n == 0) {
        fprintf(stderr, "--selected must name at least one expert\n");
        exit(2);
    }
    *n_selected = n;
}

static uint64_t bench_tensor_bytes(uint32_t type, uint64_t in_dim, uint64_t out_dim, uint64_t n_experts) {
    const gguf_type_info *info = tensor_type(type);
    if (!info || info->block_elems == 0) ds4_die("unsupported microbench tensor type");
    const uint64_t blocks = (in_dim + info->block_elems - 1) / info->block_elems;
    return n_experts * out_dim * blocks * info->block_bytes;
}

static void bench_fill_iq2_rows(block_iq2_xxs *rows, uint64_t n_rows, uint32_t blocks, uint64_t *rng) {
    for (uint64_t r = 0; r < n_rows; r++) {
        for (uint32_t b = 0; b < blocks; b++) {
            block_iq2_xxs *x = rows + r * blocks + b;
            x->d = f32_to_f16(0.02f + 0.08f * fabsf(bench_rng_f32(rng)));
            for (uint32_t i = 0; i < QK_K / 8; i++) {
                x->qs[i] = (uint16_t)bench_rng_next(rng);
            }
        }
    }
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

static void bench_make_synthetic_model(synthetic_decode_model *sm, uint32_t n_experts, uint64_t *rng) {
    memset(sm, 0, sizeof(*sm));
    if (n_experts == 0 || n_experts > DS4_N_EXPERT) ds4_die("invalid synthetic expert count");

    const uint64_t gate_bytes = bench_tensor_bytes(DS4_TENSOR_IQ2_XXS,
                                                   DS4_N_EMBD,
                                                   DS4_N_FF_EXP,
                                                   n_experts);
    const uint64_t up_bytes = gate_bytes;
    const uint64_t down_bytes = bench_tensor_bytes(DS4_TENSOR_Q2_K,
                                                   DS4_N_FF_EXP,
                                                   DS4_N_EMBD,
                                                   n_experts);
    const uint64_t total = gate_bytes + up_bytes + down_bytes;
    sm->map = xmalloc((size_t)total);
    sm->model.fd = -1;
    sm->model.map = sm->map;
    sm->model.size = total;

    sm->gate = (ds4_tensor){
        .ndim = 3,
        .dim = { DS4_N_EMBD, DS4_N_FF_EXP, n_experts },
        .type = DS4_TENSOR_IQ2_XXS,
        .abs_offset = 0,
        .bytes = gate_bytes,
    };
    sm->up = sm->gate;
    sm->up.abs_offset = gate_bytes;
    sm->down = (ds4_tensor){
        .ndim = 3,
        .dim = { DS4_N_FF_EXP, DS4_N_EMBD, n_experts },
        .type = DS4_TENSOR_Q2_K,
        .abs_offset = gate_bytes + up_bytes,
        .bytes = down_bytes,
    };
    sm->layer.ffn_gate_exps = &sm->gate;
    sm->layer.ffn_up_exps = &sm->up;
    sm->layer.ffn_down_exps = &sm->down;

    bench_fill_iq2_rows((block_iq2_xxs *)(sm->map + sm->gate.abs_offset),
                        n_experts * DS4_N_FF_EXP,
                        DS4_N_EMBD / QK_K,
                        rng);
    bench_fill_iq2_rows((block_iq2_xxs *)(sm->map + sm->up.abs_offset),
                        n_experts * DS4_N_FF_EXP,
                        DS4_N_EMBD / QK_K,
                        rng);
    bench_fill_q2_rows((block_q2_K *)(sm->map + sm->down.abs_offset),
                       n_experts * DS4_N_EMBD,
                       DS4_N_FF_EXP / QK_K,
                       rng);
}

static void bench_free_synthetic_model(synthetic_decode_model *sm) {
    free(sm->map);
    memset(sm, 0, sizeof(*sm));
}

static void bench_selected_one_profile(
        float                   *out,
        const ds4_model         *model,
        const ds4_layer_weights *layer,
        const float             *x,
        const int32_t           *selected_rows,
        const float             *weight_rows,
        uint32_t                 n_selected,
        float                   *mid_all,
        block_q8_K              *xq,
        block_q8_K              *midq,
        decode_stage_stats      *stats) {
    int selected[DS4_N_EXPERT_USED];
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t expert_out_dim = layer->ffn_gate_exps->dim[1];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t down_out_dim = layer->ffn_down_exps->dim[1];
    const double total_t0 = now_sec();
    const double setup_t0 = total_t0;

    if (expert_in_dim % QK_K != 0) ds4_die("decode microbench expert input is not QK_K aligned");
    if (down_in_dim % QK_K != 0) ds4_die("decode microbench down input is not QK_K aligned");
    if (expert_out_dim != down_in_dim || down_out_dim != DS4_N_EMBD) {
        ds4_die("decode microbench tensor layout is unexpected");
    }
    if (n_selected == 0 || n_selected > DS4_N_EXPERT_USED) {
        ds4_die("decode microbench selected expert count is invalid");
    }
    const bool is_q4 = layer->ffn_gate_exps->type == DS4_TENSOR_Q4_K;
    if (is_q4) {
        if (layer->ffn_up_exps->type != DS4_TENSOR_Q4_K ||
            layer->ffn_down_exps->type != DS4_TENSOR_Q4_K) {
            ds4_die("decode microbench expected all Q4_K routed expert tensors");
        }
    } else if (!(layer->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS &&
                 layer->ffn_up_exps->type == DS4_TENSOR_IQ2_XXS &&
                 layer->ffn_down_exps->type == DS4_TENSOR_Q2_K)) {
        ds4_die("decode microbench expected IQ2_XXS/IQ2_XXS/Q2_K routed expert tensors");
    }
    for (uint32_t i = 0; i < n_selected; i++) {
        const int32_t expert = selected_rows[i];
        if (expert < 0 || expert >= DS4_N_EXPERT) ds4_die("decode microbench expert id is outside range");
        selected[i] = (int)expert;
    }
    memset(out, 0, (size_t)DS4_N_EMBD * sizeof(out[0]));
    stats->setup_seconds += now_sec() - setup_t0;

    const double xq_t0 = now_sec();
    ds4_quantize_row_q8_K(x, xq, (int64_t)expert_in_dim);
    stats->xq_seconds += now_sec() - xq_t0;

    const double gate_up_t0 = now_sec();
    if (is_q4) {
        matvec_q4_k_experts_mid_prequant(mid_all, model,
                                         layer->ffn_gate_exps,
                                         layer->ffn_up_exps,
                                         xq,
                                         selected,
                                         weight_rows,
                                         (int)n_selected,
                                         DS4_SWIGLU_CLAMP_EXP);
    } else {
        matvec_iq2_xxs_experts_mid_prequant(mid_all, model,
                                            layer->ffn_gate_exps,
                                            layer->ffn_up_exps,
                                            xq,
                                            selected,
                                            weight_rows,
                                            (int)n_selected,
                                            DS4_SWIGLU_CLAMP_EXP);
    }
    stats->gate_up_seconds += now_sec() - gate_up_t0;

    const double midq_t0 = now_sec();
    for (uint32_t i = 0; i < n_selected; i++) {
        ds4_quantize_row_q8_K(mid_all + (uint64_t)i * down_in_dim,
                              midq + (uint64_t)i * (down_in_dim / QK_K),
                              (int64_t)down_in_dim);
    }
    stats->midq_seconds += now_sec() - midq_t0;

    const double down_t0 = now_sec();
    if (is_q4) {
        matvec_q4_k_experts_accum_prequant(out, model, layer->ffn_down_exps, midq, selected, (int)n_selected);
    } else {
        matvec_q2_k_experts_accum_prequant(out, model, layer->ffn_down_exps,
                                           midq, selected, (int)n_selected);
    }
    stats->down_seconds += now_sec() - down_t0;
    stats->total_seconds += now_sec() - total_t0;
}

static void bench_selected_batch_profile_staged(
        float                   *out,
        const ds4_model         *model,
        const ds4_layer_weights *layer,
        uint32_t                 il,
        const float             *norm,
        const int32_t           *selected_rows,
        const float             *weight_rows,
        uint32_t                 n_tok,
        float                   *mid,
        block_q8_K              *xq,
        block_q8_K              *midq,
        uint32_t                *pair_ids,
        decode_stage_stats      *stats) {
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t expert_out_dim = layer->ffn_gate_exps->dim[1];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t down_out_dim = layer->ffn_down_exps->dim[1];
    const double total_t0 = now_sec();
    const double setup_t0 = total_t0;

    if (layer->ffn_gate_exps->type != DS4_TENSOR_IQ2_XXS ||
        layer->ffn_up_exps->type != DS4_TENSOR_IQ2_XXS ||
        layer->ffn_down_exps->type != DS4_TENSOR_Q2_K) {
        ds4_die("batch microbench reference only supports IQ2_XXS/IQ2_XXS/Q2_K");
    }
    if (expert_in_dim % QK_K != 0 || down_in_dim % QK_K != 0 ||
        expert_out_dim != down_in_dim || down_out_dim != DS4_N_EMBD) {
        ds4_die("batch microbench tensor layout is unexpected");
    }

    const uint32_t total_pairs = n_tok * DS4_N_EXPERT_USED;
    uint32_t counts[DS4_N_EXPERT + 1] = {0};
    uint32_t cursor[DS4_N_EXPERT] = {0};
    uint32_t active_expert[DS4_N_EXPERT];
    uint32_t n_active = 0;
    const uint64_t xq_blocks = expert_in_dim / QK_K;
    const uint64_t midq_blocks = down_in_dim / QK_K;

    memset(out, 0, (size_t)((uint64_t)n_tok * down_out_dim) * sizeof(out[0]));
    for (uint32_t pair_id = 0; pair_id < total_pairs; pair_id++) {
        const int32_t expert = selected_rows[pair_id];
        if (expert < 0 || expert >= DS4_N_EXPERT) ds4_die("batch microbench expert id is outside range");
        counts[(uint32_t)expert + 1]++;
    }
    for (uint32_t e = 0; e < DS4_N_EXPERT; e++) {
        counts[e + 1] += counts[e];
        cursor[e] = counts[e];
        if (counts[e + 1] != counts[e]) active_expert[n_active++] = e;
    }
    for (uint32_t pair_id = 0; pair_id < total_pairs; pair_id++) {
        const uint32_t expert = (uint32_t)selected_rows[pair_id];
        pair_ids[cursor[expert]++] = pair_id;
    }
    stats->setup_seconds += now_sec() - setup_t0;

    const double xq_t0 = now_sec();
    for (uint32_t t = 0; t < n_tok; t++) {
        ds4_quantize_row_q8_K(norm + (uint64_t)t * expert_in_dim,
                              xq + (uint64_t)t * xq_blocks,
                              (int64_t)expert_in_dim);
    }
    stats->xq_seconds += now_sec() - xq_t0;

    const double gate_up_t0 = now_sec();
    matvec_iq2_xxs_batch_mid_ctx mid_ctx = {
        .mid = mid,
        .xq = xq,
        .pair_ids = pair_ids,
        .expert_offset = counts,
        .active_expert = active_expert,
        .pair_weight = weight_rows,
        .clamp = DS4_SWIGLU_CLAMP_EXP,
        .in_dim = expert_in_dim,
        .out_dim = expert_out_dim,
        .xq_blocks = xq_blocks,
    };
    for (uint32_t ai = 0; ai < n_active; ai++) {
        const uint32_t expert = active_expert[ai];
        uint64_t gate_in_dim, gate_out_dim;
        uint64_t up_in_dim, up_out_dim;
        mid_ctx.gate_base[expert] = tensor_expert_bytes(model, layer->ffn_gate_exps, expert,
                                                        &gate_in_dim, &gate_out_dim, &mid_ctx.gate_row_bytes[expert]);
        mid_ctx.up_base[expert] = tensor_expert_bytes(model, layer->ffn_up_exps, expert,
                                                      &up_in_dim, &up_out_dim, &mid_ctx.up_row_bytes[expert]);
        if (gate_in_dim != expert_in_dim || up_in_dim != expert_in_dim ||
            gate_out_dim != expert_out_dim || up_out_dim != expert_out_dim) {
            ds4_die("batch microbench gate/up tensor layout mismatch");
        }
    }
    ds4_cpu_moe_parallel_expert_rows(il,
                                      active_expert,
                                      n_active,
                                      expert_out_dim,
                                      ds4_cpu_moe_row_chunk(),
                                      matvec_iq2_xxs_batch_mid_expert_rows,
                                      &mid_ctx);
    stats->gate_up_seconds += now_sec() - gate_up_t0;

    const double midq_t0 = now_sec();
    quantize_mid_pairs_ctx quant_ctx = {
        .mid = mid,
        .midq = midq,
        .down_in_dim = down_in_dim,
        .down_blocks = midq_blocks,
    };
    ds4_parallel_for(total_pairs, quantize_mid_pairs_worker, &quant_ctx);
    stats->midq_seconds += now_sec() - midq_t0;

    const double down_t0 = now_sec();
    matvec_q2_k_batch_accum_rows_ctx down_ctx = {
        .moe = out,
        .midq = midq,
        .pair_ids = pair_ids,
        .expert_offset = counts,
        .active_expert = active_expert,
        .n_active = n_active,
        .n_pairs = total_pairs,
        .n_tok = n_tok,
        .in_dim = down_in_dim,
        .out_dim = down_out_dim,
        .midq_blocks = midq_blocks,
    };
    for (uint32_t ai = 0; ai < n_active; ai++) {
        const uint32_t expert = active_expert[ai];
        uint64_t in_dim, out_dim;
        down_ctx.base[expert] = tensor_expert_bytes(model, layer->ffn_down_exps, expert,
                                                    &in_dim, &out_dim, &down_ctx.row_bytes[expert]);
        if (in_dim != down_in_dim || out_dim != down_out_dim) {
            ds4_die("batch microbench down tensor layout mismatch");
        }
    }
    matvec_q2_k_batch_accum_down_auto(&down_ctx);
    stats->down_seconds += now_sec() - down_t0;
    stats->total_seconds += now_sec() - total_t0;
}

static double bench_production_loop(
        float                   *out,
        const ds4_model         *model,
        const ds4_layer_weights *layer,
        const float             *x,
        const int32_t           *selected,
        const float             *weights,
        uint32_t                 n_selected,
        float                   *mid,
        block_q8_K              *xq,
        block_q8_K              *midq,
        uint32_t                 iters) {
    const double t0 = now_sec();
    for (uint32_t it = 0; it < iters; it++) {
        layer_routed_moe_selected_one_n_prealloc(out,
                                                 model,
                                                 layer,
                                                 x,
                                                 selected,
                                                 weights,
                                                 n_selected,
                                                 DS4_SWIGLU_CLAMP_EXP,
                                                 mid,
                                                 xq,
                                                 midq);
    }
    return now_sec() - t0;
}

static decode_stage_stats bench_profile_loop(
        float                   *out,
        const ds4_model         *model,
        const ds4_layer_weights *layer,
        const float             *x,
        const int32_t           *selected,
        const float             *weights,
        uint32_t                 n_selected,
        float                   *mid,
        block_q8_K              *xq,
        block_q8_K              *midq,
        uint32_t                 iters) {
    decode_stage_stats stats = {0};
    for (uint32_t it = 0; it < iters; it++) {
        bench_selected_one_profile(out,
                                   model,
                                   layer,
                                   x,
                                   selected,
                                   weights,
                                   n_selected,
                                   mid,
                                   xq,
                                   midq,
                                   &stats);
    }
    return stats;
}

static double bench_batch_production_loop(
        float                   *out,
        const ds4_model         *model,
        const ds4_layer_weights *layer,
        uint32_t                 il,
        const float             *norm,
        const int32_t           *selected_rows,
        const float             *weights,
        uint32_t                 n_tok,
        float                   *mid,
        block_q8_K              *xq,
        block_q8_K              *midq,
        uint32_t                *pair_ids,
        uint32_t                 iters) {
    const double t0 = now_sec();
    for (uint32_t it = 0; it < iters; it++) {
        layer_routed_moe_selected_batch_prealloc(out,
                                                 model,
                                                 layer,
                                                 il,
                                                 norm,
                                                 selected_rows,
                                                 weights,
                                                 n_tok,
                                                 DS4_SWIGLU_CLAMP_EXP,
                                                 mid,
                                                 xq,
                                                 midq,
                                                 pair_ids);
    }
    return now_sec() - t0;
}

static decode_stage_stats bench_batch_profile_loop_staged(
        float                   *out,
        const ds4_model         *model,
        const ds4_layer_weights *layer,
        uint32_t                 il,
        const float             *norm,
        const int32_t           *selected_rows,
        const float             *weights,
        uint32_t                 n_tok,
        float                   *mid,
        block_q8_K              *xq,
        block_q8_K              *midq,
        uint32_t                *pair_ids,
        uint32_t                 iters) {
    decode_stage_stats stats = {0};
    for (uint32_t it = 0; it < iters; it++) {
        bench_selected_batch_profile_staged(out,
                                            model,
                                            layer,
                                            il,
                                            norm,
                                            selected_rows,
                                            weights,
                                            n_tok,
                                            mid,
                                            xq,
                                            midq,
                                            pair_ids,
                                            &stats);
    }
    return stats;
}

static double bench_checksum(const float *a, uint64_t n) {
    double sum = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        sum += (double)a[i] * 0.0000003;
    }
    return sum;
}

static double bench_max_abs_diff(const float *a, const float *b, uint64_t n) {
    double max_abs = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        const double d = fabs((double)a[i] - (double)b[i]);
        if (d > max_abs) max_abs = d;
    }
    return max_abs;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *selected_arg = NULL;
    uint32_t layer_id = 20;
    uint32_t iters = 64;
    uint32_t warmup = 4;
    uint32_t threads = 0;
    uint32_t n_tok = 1;
    uint32_t synthetic_experts = DS4_N_EXPERT_USED;
    int32_t selected[DS4_N_EXPERT_USED];
    uint32_t n_selected = DS4_N_EXPERT_USED;
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) selected[i] = (int32_t)i;

    for (int i = 1; i < argc; i++) {
        model_path = bench_parse_str(&i, argc, argv, "--model", model_path);
        selected_arg = bench_parse_str(&i, argc, argv, "--selected", selected_arg);
        layer_id = bench_parse_u32(&i, argc, argv, "--layer", layer_id);
        iters = bench_parse_u32(&i, argc, argv, "--iters", iters);
        warmup = bench_parse_u32(&i, argc, argv, "--warmup", warmup);
        threads = bench_parse_u32(&i, argc, argv, "--threads", threads);
        n_tok = bench_parse_u32(&i, argc, argv, "--tokens", n_tok);
        synthetic_experts = bench_parse_u32(&i, argc, argv, "--synthetic-experts", synthetic_experts);
        if (strcmp(argv[i], "--help") == 0) {
            fprintf(stderr,
                    "usage: %s [--model GGUF] [--layer N] [--selected e0,e1,...] [--tokens N] [--iters N] [--warmup N] [--threads N]\n",
                    argv[0]);
            return 0;
        }
    }
    bench_parse_selected(selected_arg, selected, &n_selected);
    if (layer_id >= DS4_N_LAYER) ds4_die("--layer is outside DS4 layer range");
    if (n_tok == 0) ds4_die("--tokens must be at least 1");
    if (n_tok > 1 && n_selected != DS4_N_EXPERT_USED) {
        ds4_die("batch microbench requires exactly DS4_N_EXPERT_USED selected experts");
    }
    if (threads != 0) g_requested_threads = threads;

    uint64_t rng = UINT64_C(0x9e3779b97f4a7c15);
    ds4_model loaded_model;
    ds4_weights loaded_weights;
    synthetic_decode_model synthetic;
    memset(&loaded_model, 0, sizeof(loaded_model));
    memset(&loaded_weights, 0, sizeof(loaded_weights));
    memset(&synthetic, 0, sizeof(synthetic));

    const ds4_model *model = NULL;
    const ds4_layer_weights *layer = NULL;
    if (model_path) {
        model_open(&loaded_model, model_path, false, false);
        weights_bind(&loaded_weights, &loaded_model);
        model = &loaded_model;
        layer = &loaded_weights.layer[layer_id];
    } else {
        if (synthetic_experts < n_selected) synthetic_experts = n_selected;
        bench_make_synthetic_model(&synthetic, synthetic_experts, &rng);
        model = &synthetic.model;
        layer = &synthetic.layer;
    }

    for (uint32_t i = 0; i < n_selected; i++) {
        if ((uint32_t)selected[i] >= layer->ffn_gate_exps->dim[2]) {
            fprintf(stderr,
                    "selected expert %d is outside tensor expert count %llu\n",
                    selected[i],
                    (unsigned long long)layer->ffn_gate_exps->dim[2]);
            exit(2);
        }
    }

    const bool batch_mode = n_tok > 1;
    const uint32_t total_pairs = batch_mode ? n_tok * DS4_N_EXPERT_USED : DS4_N_EXPERT_USED;
    const uint64_t output_elems = (uint64_t)n_tok * DS4_N_EMBD;
    float *x = xmalloc((size_t)((uint64_t)n_tok * DS4_N_EMBD) * sizeof(x[0]));
    float *weights = xmalloc((size_t)total_pairs * sizeof(weights[0]));
    int32_t *selected_rows = xmalloc((size_t)total_pairs * sizeof(selected_rows[0]));
    float *mid = xmalloc((size_t)((uint64_t)total_pairs * DS4_N_FF_EXP) * sizeof(mid[0]));
    block_q8_K *xq = xmalloc((size_t)((uint64_t)n_tok * (DS4_N_EMBD / QK_K)) * sizeof(xq[0]));
    block_q8_K *midq = xmalloc((size_t)((uint64_t)total_pairs * (DS4_N_FF_EXP / QK_K)) * sizeof(midq[0]));
    uint32_t *pair_ids = xmalloc((size_t)total_pairs * sizeof(pair_ids[0]));
    float *prod_out = xmalloc((size_t)output_elems * sizeof(prod_out[0]));
    float *profile_out = xmalloc((size_t)output_elems * sizeof(profile_out[0]));

    for (uint64_t i = 0; i < (uint64_t)n_tok * DS4_N_EMBD; i++) x[i] = bench_rng_f32(&rng);
    if (batch_mode) {
        for (uint32_t t = 0; t < n_tok; t++) {
            float sum_w = 0.0f;
            for (uint32_t slot = 0; slot < DS4_N_EXPERT_USED; slot++) {
                const uint32_t pair_id = t * DS4_N_EXPERT_USED + slot;
                selected_rows[pair_id] = selected[slot];
                weights[pair_id] = 0.5f + fabsf(bench_rng_f32(&rng));
                sum_w += weights[pair_id];
            }
            for (uint32_t slot = 0; slot < DS4_N_EXPERT_USED; slot++) {
                weights[t * DS4_N_EXPERT_USED + slot] /= sum_w;
            }
        }
    } else {
        float sum_w = 0.0f;
        for (uint32_t i = 0; i < n_selected; i++) {
            weights[i] = 0.5f + fabsf(bench_rng_f32(&rng));
            sum_w += weights[i];
        }
        for (uint32_t i = 0; i < n_selected; i++) weights[i] /= sum_w;
    }

    if (batch_mode) {
        for (uint32_t i = 0; i < warmup; i++) {
            layer_routed_moe_selected_batch_prealloc(prod_out,
                                                     model,
                                                     layer,
                                                     layer_id,
                                                     x,
                                                     selected_rows,
                                                     weights,
                                                     n_tok,
                                                     DS4_SWIGLU_CLAMP_EXP,
                                                     mid,
                                                     xq,
                                                     midq,
                                                     pair_ids);
        }
    } else {
        for (uint32_t i = 0; i < warmup; i++) {
            layer_routed_moe_selected_one_n_prealloc(prod_out,
                                                     model,
                                                     layer,
                                                     x,
                                                     selected,
                                                     weights,
                                                     n_selected,
                                                     DS4_SWIGLU_CLAMP_EXP,
                                                     mid,
                                                     xq,
                                                     midq);
        }
    }

    double prod_seconds;
    decode_stage_stats prof;
    if (batch_mode) {
        prod_seconds = bench_batch_production_loop(prod_out,
                                                   model,
                                                   layer,
                                                   layer_id,
                                                   x,
                                                   selected_rows,
                                                   weights,
                                                   n_tok,
                                                   mid,
                                                   xq,
                                                   midq,
                                                   pair_ids,
                                                   iters);
        prof = bench_batch_profile_loop_staged(profile_out,
                                               model,
                                               layer,
                                               layer_id,
                                               x,
                                               selected_rows,
                                               weights,
                                               n_tok,
                                               mid,
                                               xq,
                                               midq,
                                               pair_ids,
                                               iters);
    } else {
        prod_seconds = bench_production_loop(prod_out,
                                             model,
                                             layer,
                                             x,
                                             selected,
                                             weights,
                                             n_selected,
                                             mid,
                                             xq,
                                             midq,
                                             iters);
        prof = bench_profile_loop(profile_out,
                                  model,
                                  layer,
                                  x,
                                  selected,
                                  weights,
                                  n_selected,
                                  mid,
                                  xq,
                                  midq,
                                  iters);
    }
    const double max_abs = bench_max_abs_diff(prod_out, profile_out, output_elems);

    ds4_threads_init();
    const uint32_t active_threads = g_pool.n_threads ? g_pool.n_threads : 1;
    const double prod_us = 1.0e6 * prod_seconds / (double)iters;
    const double prof_us = 1.0e6 * prof.total_seconds / (double)iters;
    const double checksum = bench_checksum(prod_out, output_elems) +
                            bench_checksum(profile_out, output_elems);

    printf("%s source=%s layer=%u tokens=%u experts=%u threads=%u iters=%u warmup=%u cpu_isa=%s\n",
           batch_mode ? "decode-selected-batch" : "decode-selected-one",
           model_path ? "model" : "synthetic",
           layer_id,
           n_tok,
           n_selected,
           active_threads,
           iters,
           warmup,
#if defined(DS4_HAVE_AVX512_VNNI)
           "avx512_vnni"
#elif defined(__AVX2__)
           "avx2"
#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
           "neon_dotprod"
#elif defined(__ARM_NEON)
           "neon"
#else
           "scalar"
#endif
    );
    printf("selected=");
    for (uint32_t i = 0; i < n_selected; i++) {
        printf("%s%d", i ? "," : "", selected[i]);
    }
    printf("\n");
    printf("correctness max_abs_diff=%.9g checksum=%.9f\n", max_abs, checksum);
    printf("production seconds=%.6f us_per_call=%.3f calls_per_s=%.3f\n",
           prod_seconds,
           prod_us,
           (double)iters / prod_seconds);
    if (batch_mode) {
        printf("production us_per_token=%.3f tokens_per_s=%.3f\n",
               prod_us / (double)n_tok,
               (double)(iters * n_tok) / prod_seconds);
    }
    printf("profiled seconds=%.6f us_per_call=%.3f calls_per_s=%.3f\n",
           prof.total_seconds,
           prof_us,
           (double)iters / prof.total_seconds);
    if (batch_mode) {
        printf("profiled us_per_token=%.3f tokens_per_s=%.3f\n",
               prof_us / (double)n_tok,
               (double)(iters * n_tok) / prof.total_seconds);
    }
    printf("stage_us_per_call setup=%.3f xq=%.3f gate_up=%.3f midq=%.3f down=%.3f\n",
           1.0e6 * prof.setup_seconds / (double)iters,
           1.0e6 * prof.xq_seconds / (double)iters,
           1.0e6 * prof.gate_up_seconds / (double)iters,
           1.0e6 * prof.midq_seconds / (double)iters,
           1.0e6 * prof.down_seconds / (double)iters);
    printf("stage_pct setup=%.2f xq=%.2f gate_up=%.2f midq=%.2f down=%.2f\n",
           100.0 * prof.setup_seconds / prof.total_seconds,
           100.0 * prof.xq_seconds / prof.total_seconds,
           100.0 * prof.gate_up_seconds / prof.total_seconds,
           100.0 * prof.midq_seconds / prof.total_seconds,
           100.0 * prof.down_seconds / prof.total_seconds);

    ds4_threads_shutdown();
    free(profile_out);
    free(prod_out);
    free(pair_ids);
    free(midq);
    free(xq);
    free(mid);
    free(selected_rows);
    free(weights);
    free(x);
    if (model_path) {
        model_close(&loaded_model);
    } else {
        bench_free_synthetic_model(&synthetic);
    }
    return max_abs < 1.0e-3 ? 0 : 1;
}
