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

static void bench_fill_iq2_rows(block_iq2_xxs *rows, uint32_t n_rows, uint32_t blocks, uint64_t *rng) {
    for (uint32_t r = 0; r < n_rows; r++) {
        for (uint32_t b = 0; b < blocks; b++) {
            block_iq2_xxs *x = rows + (uint64_t)r * blocks + b;
            x->d = f32_to_f16(0.02f + 0.08f * fabsf(bench_rng_f32(rng)));
            for (uint32_t i = 0; i < QK_K / 8; i++) {
                x->qs[i] = (uint16_t)bench_rng_next(rng);
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

static DS4_MAYBE_UNUSED void bench_iq2_pair_panel4(
        int n,
        float out0[4],
        float out1[4],
        const block_iq2_xxs *x0,
        const block_iq2_xxs *x1,
        const block_q8_K *y0,
        const block_q8_K *y1,
        const block_q8_K *y2,
        const block_q8_K *y3) {
#if defined(__AVX2__)
    const int nb = n / QK_K;
    const block_q8_K *ys[4] = { y0, y1, y2, y3 };
    __m256 accum0[4] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(),
    };
    __m256 accum1[4] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(),
    };

    for (int i = 0; i < nb; i++) {
        const float x0d = f16_to_f32(x0[i].d);
        const float x1d = f16_to_f32(x1[i].d);
        const uint16_t *q20 = x0[i].qs;
        const uint16_t *q21 = x1[i].qs;
        const int8_t *q8[4] = { ys[0][i].qs, ys[1][i].qs, ys[2][i].qs, ys[3][i].qs };
        __m256i sumi01[4] = {
            _mm256_setzero_si256(), _mm256_setzero_si256(),
            _mm256_setzero_si256(), _mm256_setzero_si256(),
        };
        __m256i sumi02[4] = {
            _mm256_setzero_si256(), _mm256_setzero_si256(),
            _mm256_setzero_si256(), _mm256_setzero_si256(),
        };
        __m256i sumi11[4] = {
            _mm256_setzero_si256(), _mm256_setzero_si256(),
            _mm256_setzero_si256(), _mm256_setzero_si256(),
        };
        __m256i sumi12[4] = {
            _mm256_setzero_si256(), _mm256_setzero_si256(),
            _mm256_setzero_si256(), _mm256_setzero_si256(),
        };

        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            __m256i q8_1[4];
            __m256i q8_2[4];
            for (int p = 0; p < 4; p++) {
                q8_1[p] = _mm256_loadu_si256((const __m256i *)q8[p]);
                q8[p] += 32;
                q8_2[p] = _mm256_loadu_si256((const __m256i *)q8[p]);
                q8[p] += 32;
            }

#define BENCH_IQ2_PANEL_ACCUM(q2_ptr, accum_a, accum_b) do {                           \
                uint32_t aux32[4];                                                     \
                memcpy(aux32, (q2_ptr), sizeof(aux32));                                \
                (q2_ptr) += 8;                                                         \
                const uint8_t *aux8 = (const uint8_t *)aux32;                          \
                const __m256i q2_1 = _mm256_set_epi64x(                                \
                    iq2xxs_grid[aux8[3]], iq2xxs_grid[aux8[2]],                       \
                    iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);                      \
                const __m256i q2_2 = _mm256_set_epi64x(                                \
                    iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]],                     \
                    iq2xxs_grid[aux8[9]],  iq2xxs_grid[aux8[8]]);                     \
                const __m256i s2_1 = _mm256_set_epi64x(                                \
                    ds4_load_i8x8_as_i64(iq2xxs_signs[(aux32[1] >> 21) & 127]),        \
                    ds4_load_i8x8_as_i64(iq2xxs_signs[(aux32[1] >> 14) & 127]),        \
                    ds4_load_i8x8_as_i64(iq2xxs_signs[(aux32[1] >>  7) & 127]),        \
                    ds4_load_i8x8_as_i64(iq2xxs_signs[(aux32[1] >>  0) & 127]));       \
                const __m256i s2_2 = _mm256_set_epi64x(                                \
                    ds4_load_i8x8_as_i64(iq2xxs_signs[(aux32[3] >> 21) & 127]),        \
                    ds4_load_i8x8_as_i64(iq2xxs_signs[(aux32[3] >> 14) & 127]),        \
                    ds4_load_i8x8_as_i64(iq2xxs_signs[(aux32[3] >>  7) & 127]),        \
                    ds4_load_i8x8_as_i64(iq2xxs_signs[(aux32[3] >>  0) & 127]));       \
                const uint16_t ls1 = (uint16_t)(aux32[1] >> 28);                       \
                const uint16_t ls2 = (uint16_t)(aux32[3] >> 28);                       \
                const __m256i scale1 = _mm256_set1_epi16((int16_t)(2 * ls1 + 1));      \
                const __m256i scale2 = _mm256_set1_epi16((int16_t)(2 * ls2 + 1));      \
                for (int bp = 0; bp < 4; bp++) {                                      \
                    const __m256i dot1 = _mm256_maddubs_epi16(                         \
                        q2_1, _mm256_sign_epi8(q8_1[bp], s2_1));                      \
                    const __m256i dot2 = _mm256_maddubs_epi16(                         \
                        q2_2, _mm256_sign_epi8(q8_2[bp], s2_2));                      \
                    (accum_a)[bp] = _mm256_add_epi32((accum_a)[bp],                    \
                        _mm256_madd_epi16(dot1, scale1));                             \
                    (accum_b)[bp] = _mm256_add_epi32((accum_b)[bp],                    \
                        _mm256_madd_epi16(dot2, scale2));                             \
                }                                                                      \
            } while (0)

            BENCH_IQ2_PANEL_ACCUM(q20, sumi01, sumi02);
            BENCH_IQ2_PANEL_ACCUM(q21, sumi11, sumi12);

#undef BENCH_IQ2_PANEL_ACCUM
        }

        for (int p = 0; p < 4; p++) {
            accum0[p] = ds4_mm256_fmadd_ps(_mm256_set1_ps(x0d * ys[p][i].d),
                                           _mm256_cvtepi32_ps(_mm256_add_epi32(sumi01[p], sumi02[p])),
                                           accum0[p]);
            accum1[p] = ds4_mm256_fmadd_ps(_mm256_set1_ps(x1d * ys[p][i].d),
                                           _mm256_cvtepi32_ps(_mm256_add_epi32(sumi11[p], sumi12[p])),
                                           accum1[p]);
        }
    }

    for (int p = 0; p < 4; p++) {
        out0[p] = 0.125f * ds4_hsum_float_8(accum0[p]);
        out1[p] = 0.125f * ds4_hsum_float_8(accum1[p]);
    }
#else
    ds4_vec_dot_iq2_xxs_pair_q8_K(n, out0 + 0, out1 + 0, x0, x1, y0);
    ds4_vec_dot_iq2_xxs_pair_q8_K(n, out0 + 1, out1 + 1, x0, x1, y1);
    ds4_vec_dot_iq2_xxs_pair_q8_K(n, out0 + 2, out1 + 2, x0, x1, y2);
    ds4_vec_dot_iq2_xxs_pair_q8_K(n, out0 + 3, out1 + 3, x0, x1, y3);
#endif
}

static double bench_baseline(
        const block_iq2_xxs *gate,
        const block_iq2_xxs *up,
        const block_q8_K    *xq,
        float               *gate_out,
        float               *up_out,
        uint32_t             rows,
        uint32_t             pairs,
        uint32_t             blocks,
        uint32_t             iters) {
    const double t0 = now_sec();
    for (uint32_t it = 0; it < iters; it++) {
        for (uint32_t r = 0; r < rows; r++) {
            const block_iq2_xxs *gate_row = gate + (uint64_t)r * blocks;
            const block_iq2_xxs *up_row = up + (uint64_t)r * blocks;
            for (uint32_t p = 0; p < pairs; p++) {
                ds4_vec_dot_iq2_xxs_pair_q8_K((int)(blocks * QK_K),
                                              gate_out + (uint64_t)r * pairs + p,
                                              up_out + (uint64_t)r * pairs + p,
                                              gate_row,
                                              up_row,
                                              xq + (uint64_t)p * blocks);
            }
        }
    }
    return now_sec() - t0;
}

static double bench_panel4(
        const block_iq2_xxs *gate,
        const block_iq2_xxs *up,
        const block_q8_K    *xq,
        float               *gate_out,
        float               *up_out,
        uint32_t             rows,
        uint32_t             pairs,
        uint32_t             blocks,
        uint32_t             iters) {
    const double t0 = now_sec();
    for (uint32_t it = 0; it < iters; it++) {
        for (uint32_t r = 0; r < rows; r++) {
            const block_iq2_xxs *gate_row = gate + (uint64_t)r * blocks;
            const block_iq2_xxs *up_row = up + (uint64_t)r * blocks;
            uint32_t p = 0;
            for (; p + 3 < pairs; p += 4) {
                float g[4];
                float u[4];
                ds4_vec_dot_iq2_xxs_pair_panel4_q8_K((int)(blocks * QK_K),
                                                      g,
                                                      u,
                                                      gate_row,
                                                      up_row,
                                                      xq + (uint64_t)(p + 0) * blocks,
                                                      xq + (uint64_t)(p + 1) * blocks,
                                                      xq + (uint64_t)(p + 2) * blocks,
                                                      xq + (uint64_t)(p + 3) * blocks);
                for (uint32_t k = 0; k < 4; k++) {
                    gate_out[(uint64_t)r * pairs + p + k] = g[k];
                    up_out[(uint64_t)r * pairs + p + k] = u[k];
                }
            }
            for (; p < pairs; p++) {
                ds4_vec_dot_iq2_xxs_pair_q8_K((int)(blocks * QK_K),
                                              gate_out + (uint64_t)r * pairs + p,
                                              up_out + (uint64_t)r * pairs + p,
                                              gate_row,
                                              up_row,
                                              xq + (uint64_t)p * blocks);
            }
        }
    }
    return now_sec() - t0;
}

static double bench_checksum(const float *a, const float *b, uint64_t n) {
    double sum = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        sum += (double)a[i] * 0.0000003;
        sum += (double)b[i] * 0.0000007;
    }
    return sum;
}

int main(int argc, char **argv) {
    uint32_t rows = 512;
    uint32_t pairs = 16;
    uint32_t dim = DS4_N_EMBD;
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

    pthread_once(&iq2xxs_signed_grid_once, iq2xxs_signed_grid_init);

    const uint32_t blocks = dim / QK_K;
    const uint64_t out_elems = (uint64_t)rows * pairs;
    uint64_t rng = UINT64_C(0x9e3779b97f4a7c15);

    block_iq2_xxs *gate = xmalloc((size_t)((uint64_t)rows * blocks) * sizeof(gate[0]));
    block_iq2_xxs *up = xmalloc((size_t)((uint64_t)rows * blocks) * sizeof(up[0]));
    block_q8_K *xq = xmalloc((size_t)((uint64_t)pairs * blocks) * sizeof(xq[0]));
    float *base_g = xmalloc((size_t)out_elems * sizeof(base_g[0]));
    float *base_u = xmalloc((size_t)out_elems * sizeof(base_u[0]));
    float *panel_g = xmalloc((size_t)out_elems * sizeof(panel_g[0]));
    float *panel_u = xmalloc((size_t)out_elems * sizeof(panel_u[0]));

    bench_fill_iq2_rows(gate, rows, blocks, &rng);
    bench_fill_iq2_rows(up, rows, blocks, &rng);
    bench_fill_q8_rows(xq, pairs, dim, &rng);

    (void)bench_baseline(gate, up, xq, base_g, base_u, rows, pairs, blocks, 1);
    (void)bench_panel4(gate, up, xq, panel_g, panel_u, rows, pairs, blocks, 1);

    double max_abs = 0.0;
    for (uint64_t i = 0; i < out_elems; i++) {
        const double dg = fabs((double)base_g[i] - (double)panel_g[i]);
        const double du = fabs((double)base_u[i] - (double)panel_u[i]);
        if (dg > max_abs) max_abs = dg;
        if (du > max_abs) max_abs = du;
    }

    const double tb = bench_baseline(gate, up, xq, base_g, base_u, rows, pairs, blocks, iters);
    const double tp = bench_panel4(gate, up, xq, panel_g, panel_u, rows, pairs, blocks, iters);
    const double dots = (double)rows * (double)pairs * (double)iters;
    const double checksum = bench_checksum(base_g, base_u, out_elems) +
                            bench_checksum(panel_g, panel_u, out_elems);

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

    free(panel_u);
    free(panel_g);
    free(base_u);
    free(base_g);
    free(xq);
    free(up);
    free(gate);
    return max_abs < 1.0e-3 ? 0 : 1;
}
