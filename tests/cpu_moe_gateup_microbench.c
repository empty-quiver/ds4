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

typedef struct {
    uint8_t q2_1[32];
    uint8_t q2_2[32];
    int8_t  s2_1[32];
    int8_t  s2_2[32];
    int16_t scale1;
    int16_t scale2;
} bench_iq2_packed_seg;

typedef struct {
    uint16_t d;
    bench_iq2_packed_seg seg[QK_K / 64];
} bench_iq2_xxs_packed;

typedef struct {
    uint8_t  q2_1[4];
    uint8_t  q2_2[4];
    uint8_t  s2_1[4];
    uint8_t  s2_2[4];
    int16_t  scale1;
    int16_t  scale2;
} bench_iq2_meta_seg;

typedef struct {
    uint16_t d;
    bench_iq2_meta_seg seg[QK_K / 64];
} bench_iq2_xxs_meta;

#if defined(__AVX2__)
static void bench_pack_iq2_rows(
        bench_iq2_xxs_packed *dst,
        const block_iq2_xxs  *src,
        uint32_t              n_rows,
        uint32_t              blocks) {
    for (uint32_t r = 0; r < n_rows; r++) {
        for (uint32_t b = 0; b < blocks; b++) {
            const block_iq2_xxs *x = src + (uint64_t)r * blocks + b;
            bench_iq2_xxs_packed *p = dst + (uint64_t)r * blocks + b;
            const uint16_t *q2 = x->qs;

            p->d = x->d;
            for (uint32_t s = 0; s < QK_K / 64; s++) {
                uint32_t aux32[4];
                memcpy(aux32, q2, sizeof(aux32));
                q2 += 8;
                const uint8_t *aux8 = (const uint8_t *)aux32;
                bench_iq2_packed_seg *seg = p->seg + s;

                memcpy(seg->q2_1 +  0, iq2xxs_grid + aux8[0], 8);
                memcpy(seg->q2_1 +  8, iq2xxs_grid + aux8[1], 8);
                memcpy(seg->q2_1 + 16, iq2xxs_grid + aux8[2], 8);
                memcpy(seg->q2_1 + 24, iq2xxs_grid + aux8[3], 8);
                memcpy(seg->q2_2 +  0, iq2xxs_grid + aux8[8], 8);
                memcpy(seg->q2_2 +  8, iq2xxs_grid + aux8[9], 8);
                memcpy(seg->q2_2 + 16, iq2xxs_grid + aux8[10], 8);
                memcpy(seg->q2_2 + 24, iq2xxs_grid + aux8[11], 8);

                memcpy(seg->s2_1 +  0, iq2xxs_signs[(aux32[1] >>  0) & 127], 8);
                memcpy(seg->s2_1 +  8, iq2xxs_signs[(aux32[1] >>  7) & 127], 8);
                memcpy(seg->s2_1 + 16, iq2xxs_signs[(aux32[1] >> 14) & 127], 8);
                memcpy(seg->s2_1 + 24, iq2xxs_signs[(aux32[1] >> 21) & 127], 8);
                memcpy(seg->s2_2 +  0, iq2xxs_signs[(aux32[3] >>  0) & 127], 8);
                memcpy(seg->s2_2 +  8, iq2xxs_signs[(aux32[3] >>  7) & 127], 8);
                memcpy(seg->s2_2 + 16, iq2xxs_signs[(aux32[3] >> 14) & 127], 8);
                memcpy(seg->s2_2 + 24, iq2xxs_signs[(aux32[3] >> 21) & 127], 8);

                seg->scale1 = (int16_t)(2 * (aux32[1] >> 28) + 1);
                seg->scale2 = (int16_t)(2 * (aux32[3] >> 28) + 1);
            }
        }
    }
}

static void bench_pack_iq2_meta_rows(
        bench_iq2_xxs_meta *dst,
        const block_iq2_xxs *src,
        uint32_t n_rows,
        uint32_t blocks) {
    for (uint32_t r = 0; r < n_rows; r++) {
        for (uint32_t b = 0; b < blocks; b++) {
            const block_iq2_xxs *x = src + (uint64_t)r * blocks + b;
            bench_iq2_xxs_meta *p = dst + (uint64_t)r * blocks + b;
            const uint16_t *q2 = x->qs;

            p->d = x->d;
            for (uint32_t s = 0; s < QK_K / 64; s++) {
                uint32_t aux32[4];
                memcpy(aux32, q2, sizeof(aux32));
                q2 += 8;
                const uint8_t *aux8 = (const uint8_t *)aux32;
                bench_iq2_meta_seg *seg = p->seg + s;

                seg->q2_1[0] = aux8[0];
                seg->q2_1[1] = aux8[1];
                seg->q2_1[2] = aux8[2];
                seg->q2_1[3] = aux8[3];
                seg->q2_2[0] = aux8[8];
                seg->q2_2[1] = aux8[9];
                seg->q2_2[2] = aux8[10];
                seg->q2_2[3] = aux8[11];
                seg->s2_1[0] = (uint8_t)((aux32[1] >>  0) & 127);
                seg->s2_1[1] = (uint8_t)((aux32[1] >>  7) & 127);
                seg->s2_1[2] = (uint8_t)((aux32[1] >> 14) & 127);
                seg->s2_1[3] = (uint8_t)((aux32[1] >> 21) & 127);
                seg->s2_2[0] = (uint8_t)((aux32[3] >>  0) & 127);
                seg->s2_2[1] = (uint8_t)((aux32[3] >>  7) & 127);
                seg->s2_2[2] = (uint8_t)((aux32[3] >> 14) & 127);
                seg->s2_2[3] = (uint8_t)((aux32[3] >> 21) & 127);
                seg->scale1 = (int16_t)(2 * (aux32[1] >> 28) + 1);
                seg->scale2 = (int16_t)(2 * (aux32[3] >> 28) + 1);
            }
        }
    }
}

static void bench_iq2_packed_pair_q8_K(
        int n,
        float *s0,
        float *s1,
        const bench_iq2_xxs_packed *x0,
        const bench_iq2_xxs_packed *x1,
        const block_q8_K *y) {
    const int nb = n / QK_K;
    __m256 accum0 = _mm256_setzero_ps();
    __m256 accum1 = _mm256_setzero_ps();

    for (int i = 0; i < nb; i++) {
        const float d0 = f16_to_f32(x0[i].d) * y[i].d;
        const float d1 = f16_to_f32(x1[i].d) * y[i].d;
        const int8_t *q8 = y[i].qs;
        __m256i sumi01 = _mm256_setzero_si256();
        __m256i sumi02 = _mm256_setzero_si256();
        __m256i sumi11 = _mm256_setzero_si256();
        __m256i sumi12 = _mm256_setzero_si256();

        for (uint32_t seg_idx = 0; seg_idx < QK_K / 64; seg_idx++) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;

#define BENCH_IQ2_PACKED_ACCUM(seg, accum_a, accum_b) do {                              \
                const __m256i q2_1 = _mm256_loadu_si256((const __m256i *)(seg)->q2_1);  \
                const __m256i q2_2 = _mm256_loadu_si256((const __m256i *)(seg)->q2_2);  \
                const __m256i s2_1 = _mm256_loadu_si256((const __m256i *)(seg)->s2_1);  \
                const __m256i s2_2 = _mm256_loadu_si256((const __m256i *)(seg)->s2_2);  \
                const __m256i dot1 = _mm256_maddubs_epi16(                              \
                    q2_1, _mm256_sign_epi8(q8_1, s2_1));                                \
                const __m256i dot2 = _mm256_maddubs_epi16(                              \
                    q2_2, _mm256_sign_epi8(q8_2, s2_2));                                \
                (accum_a) = _mm256_add_epi32((accum_a), _mm256_madd_epi16(              \
                    dot1, _mm256_set1_epi16((seg)->scale1)));                           \
                (accum_b) = _mm256_add_epi32((accum_b), _mm256_madd_epi16(              \
                    dot2, _mm256_set1_epi16((seg)->scale2)));                           \
            } while (0)

            BENCH_IQ2_PACKED_ACCUM(x0[i].seg + seg_idx, sumi01, sumi02);
            BENCH_IQ2_PACKED_ACCUM(x1[i].seg + seg_idx, sumi11, sumi12);

#undef BENCH_IQ2_PACKED_ACCUM
        }

        accum0 = ds4_mm256_fmadd_ps(_mm256_set1_ps(d0),
                                    _mm256_cvtepi32_ps(_mm256_add_epi32(sumi01, sumi02)),
                                    accum0);
        accum1 = ds4_mm256_fmadd_ps(_mm256_set1_ps(d1),
                                    _mm256_cvtepi32_ps(_mm256_add_epi32(sumi11, sumi12)),
                                    accum1);
    }

    *s0 = 0.125f * ds4_hsum_float_8(accum0);
    *s1 = 0.125f * ds4_hsum_float_8(accum1);
}

static void bench_iq2_meta_pair_q8_K(
        int n,
        float *s0,
        float *s1,
        const bench_iq2_xxs_meta *x0,
        const bench_iq2_xxs_meta *x1,
        const block_q8_K *y) {
    const int nb = n / QK_K;
    __m256 accum0 = _mm256_setzero_ps();
    __m256 accum1 = _mm256_setzero_ps();

    for (int i = 0; i < nb; i++) {
        const float d0 = f16_to_f32(x0[i].d) * y[i].d;
        const float d1 = f16_to_f32(x1[i].d) * y[i].d;
        const int8_t *q8 = y[i].qs;
        __m256i sumi01 = _mm256_setzero_si256();
        __m256i sumi02 = _mm256_setzero_si256();
        __m256i sumi11 = _mm256_setzero_si256();
        __m256i sumi12 = _mm256_setzero_si256();

        for (uint32_t seg_idx = 0; seg_idx < QK_K / 64; seg_idx++) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;

#define BENCH_IQ2_META_ACCUM(seg, accum_a, accum_b) do {                                \
                const __m256i q2_1 = _mm256_set_epi64x(                                \
                    iq2xxs_grid[(seg)->q2_1[3]], iq2xxs_grid[(seg)->q2_1[2]],          \
                    iq2xxs_grid[(seg)->q2_1[1]], iq2xxs_grid[(seg)->q2_1[0]]);         \
                const __m256i q2_2 = _mm256_set_epi64x(                                \
                    iq2xxs_grid[(seg)->q2_2[3]], iq2xxs_grid[(seg)->q2_2[2]],          \
                    iq2xxs_grid[(seg)->q2_2[1]], iq2xxs_grid[(seg)->q2_2[0]]);         \
                const __m256i s2_1 = _mm256_set_epi64x(                                \
                    ds4_load_i8x8_as_i64(iq2xxs_signs[(seg)->s2_1[3]]),                \
                    ds4_load_i8x8_as_i64(iq2xxs_signs[(seg)->s2_1[2]]),                \
                    ds4_load_i8x8_as_i64(iq2xxs_signs[(seg)->s2_1[1]]),                \
                    ds4_load_i8x8_as_i64(iq2xxs_signs[(seg)->s2_1[0]]));               \
                const __m256i s2_2 = _mm256_set_epi64x(                                \
                    ds4_load_i8x8_as_i64(iq2xxs_signs[(seg)->s2_2[3]]),                \
                    ds4_load_i8x8_as_i64(iq2xxs_signs[(seg)->s2_2[2]]),                \
                    ds4_load_i8x8_as_i64(iq2xxs_signs[(seg)->s2_2[1]]),                \
                    ds4_load_i8x8_as_i64(iq2xxs_signs[(seg)->s2_2[0]]));               \
                const __m256i dot1 = _mm256_maddubs_epi16(                             \
                    q2_1, _mm256_sign_epi8(q8_1, s2_1));                               \
                const __m256i dot2 = _mm256_maddubs_epi16(                             \
                    q2_2, _mm256_sign_epi8(q8_2, s2_2));                               \
                (accum_a) = _mm256_add_epi32((accum_a), _mm256_madd_epi16(             \
                    dot1, _mm256_set1_epi16((seg)->scale1)));                          \
                (accum_b) = _mm256_add_epi32((accum_b), _mm256_madd_epi16(             \
                    dot2, _mm256_set1_epi16((seg)->scale2)));                          \
            } while (0)

            BENCH_IQ2_META_ACCUM(x0[i].seg + seg_idx, sumi01, sumi02);
            BENCH_IQ2_META_ACCUM(x1[i].seg + seg_idx, sumi11, sumi12);

#undef BENCH_IQ2_META_ACCUM
        }

        accum0 = ds4_mm256_fmadd_ps(_mm256_set1_ps(d0),
                                    _mm256_cvtepi32_ps(_mm256_add_epi32(sumi01, sumi02)),
                                    accum0);
        accum1 = ds4_mm256_fmadd_ps(_mm256_set1_ps(d1),
                                    _mm256_cvtepi32_ps(_mm256_add_epi32(sumi11, sumi12)),
                                    accum1);
    }

    *s0 = 0.125f * ds4_hsum_float_8(accum0);
    *s1 = 0.125f * ds4_hsum_float_8(accum1);
}
#endif

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

static DS4_MAYBE_UNUSED void bench_iq2_pair_row2(
        int n,
        float out0[2],
        float out1[2],
        const block_iq2_xxs *x00,
        const block_iq2_xxs *x01,
        const block_iq2_xxs *x10,
        const block_iq2_xxs *x11,
        const block_q8_K *y) {
#if defined(__AVX2__)
    const int nb = n / QK_K;
    __m256 accum00 = _mm256_setzero_ps();
    __m256 accum01 = _mm256_setzero_ps();
    __m256 accum10 = _mm256_setzero_ps();
    __m256 accum11 = _mm256_setzero_ps();

    for (int i = 0; i < nb; i++) {
        const float d00 = f16_to_f32(x00[i].d) * y[i].d;
        const float d01 = f16_to_f32(x01[i].d) * y[i].d;
        const float d10 = f16_to_f32(x10[i].d) * y[i].d;
        const float d11 = f16_to_f32(x11[i].d) * y[i].d;
        const uint16_t *q200 = x00[i].qs;
        const uint16_t *q201 = x01[i].qs;
        const uint16_t *q210 = x10[i].qs;
        const uint16_t *q211 = x11[i].qs;
        const int8_t *q8 = y[i].qs;
        __m256i sumi001 = _mm256_setzero_si256();
        __m256i sumi002 = _mm256_setzero_si256();
        __m256i sumi011 = _mm256_setzero_si256();
        __m256i sumi012 = _mm256_setzero_si256();
        __m256i sumi101 = _mm256_setzero_si256();
        __m256i sumi102 = _mm256_setzero_si256();
        __m256i sumi111 = _mm256_setzero_si256();
        __m256i sumi112 = _mm256_setzero_si256();

        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;

#define BENCH_IQ2_ROW2_ACCUM(q2_ptr, accum_a, accum_b) do {                             \
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
                const __m256i dot1 = _mm256_maddubs_epi16(                             \
                    q2_1, _mm256_sign_epi8(q8_1, s2_1));                               \
                const __m256i dot2 = _mm256_maddubs_epi16(                             \
                    q2_2, _mm256_sign_epi8(q8_2, s2_2));                               \
                const uint16_t ls1 = (uint16_t)(aux32[1] >> 28);                       \
                const uint16_t ls2 = (uint16_t)(aux32[3] >> 28);                       \
                (accum_a) = _mm256_add_epi32((accum_a), _mm256_madd_epi16(             \
                    dot1, _mm256_set1_epi16((int16_t)(2 * ls1 + 1))));                 \
                (accum_b) = _mm256_add_epi32((accum_b), _mm256_madd_epi16(             \
                    dot2, _mm256_set1_epi16((int16_t)(2 * ls2 + 1))));                 \
            } while (0)

            BENCH_IQ2_ROW2_ACCUM(q200, sumi001, sumi002);
            BENCH_IQ2_ROW2_ACCUM(q201, sumi011, sumi012);
            BENCH_IQ2_ROW2_ACCUM(q210, sumi101, sumi102);
            BENCH_IQ2_ROW2_ACCUM(q211, sumi111, sumi112);

#undef BENCH_IQ2_ROW2_ACCUM
        }

        accum00 = ds4_mm256_fmadd_ps(_mm256_set1_ps(d00),
                                     _mm256_cvtepi32_ps(_mm256_add_epi32(sumi001, sumi002)),
                                     accum00);
        accum01 = ds4_mm256_fmadd_ps(_mm256_set1_ps(d01),
                                     _mm256_cvtepi32_ps(_mm256_add_epi32(sumi011, sumi012)),
                                     accum01);
        accum10 = ds4_mm256_fmadd_ps(_mm256_set1_ps(d10),
                                     _mm256_cvtepi32_ps(_mm256_add_epi32(sumi101, sumi102)),
                                     accum10);
        accum11 = ds4_mm256_fmadd_ps(_mm256_set1_ps(d11),
                                     _mm256_cvtepi32_ps(_mm256_add_epi32(sumi111, sumi112)),
                                     accum11);
    }

    out0[0] = 0.125f * ds4_hsum_float_8(accum00);
    out1[0] = 0.125f * ds4_hsum_float_8(accum01);
    out0[1] = 0.125f * ds4_hsum_float_8(accum10);
    out1[1] = 0.125f * ds4_hsum_float_8(accum11);
#else
    ds4_vec_dot_iq2_xxs_pair_q8_K(n, out0 + 0, out1 + 0, x00, x01, y);
    ds4_vec_dot_iq2_xxs_pair_q8_K(n, out0 + 1, out1 + 1, x10, x11, y);
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
            for (; p + 1 < pairs; p += 2) {
                float g[2];
                float u[2];
                ds4_vec_dot_iq2_xxs_pair_panel2_q8_K((int)(blocks * QK_K),
                                                      g,
                                                      u,
                                                      gate_row,
                                                      up_row,
                                                      xq + (uint64_t)(p + 0) * blocks,
                                                      xq + (uint64_t)(p + 1) * blocks);
                for (uint32_t k = 0; k < 2; k++) {
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

static double bench_row2(
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
        for (uint32_t p = 0; p < pairs; p++) {
            const block_q8_K *xq_row = xq + (uint64_t)p * blocks;
            uint32_t r = 0;
            for (; r + 1 < rows; r += 2) {
                const block_iq2_xxs *gate0 = gate + (uint64_t)r * blocks;
                const block_iq2_xxs *up0 = up + (uint64_t)r * blocks;
                const block_iq2_xxs *gate1 = gate + (uint64_t)(r + 1) * blocks;
                const block_iq2_xxs *up1 = up + (uint64_t)(r + 1) * blocks;
                float g[2];
                float u[2];
                bench_iq2_pair_row2((int)(blocks * QK_K),
                                    g,
                                    u,
                                    gate0,
                                    up0,
                                    gate1,
                                    up1,
                                    xq_row);
                gate_out[(uint64_t)r * pairs + p] = g[0];
                up_out[(uint64_t)r * pairs + p] = u[0];
                gate_out[(uint64_t)(r + 1) * pairs + p] = g[1];
                up_out[(uint64_t)(r + 1) * pairs + p] = u[1];
            }
            if (r < rows) {
                const block_iq2_xxs *gate_row = gate + (uint64_t)r * blocks;
                const block_iq2_xxs *up_row = up + (uint64_t)r * blocks;
                ds4_vec_dot_iq2_xxs_pair_q8_K((int)(blocks * QK_K),
                                              gate_out + (uint64_t)r * pairs + p,
                                              up_out + (uint64_t)r * pairs + p,
                                              gate_row,
                                              up_row,
                                              xq_row);
            }
        }
    }
    return now_sec() - t0;
}

static double bench_packed(
        const block_iq2_xxs        *gate,
        const block_iq2_xxs        *up,
        const bench_iq2_xxs_packed *gate_packed,
        const bench_iq2_xxs_packed *up_packed,
        const block_q8_K           *xq,
        float                      *gate_out,
        float                      *up_out,
        uint32_t                    rows,
        uint32_t                    pairs,
        uint32_t                    blocks,
        uint32_t                    iters) {
#if defined(__AVX2__)
    (void)gate;
    (void)up;
    const double t0 = now_sec();
    for (uint32_t it = 0; it < iters; it++) {
        for (uint32_t r = 0; r < rows; r++) {
            const bench_iq2_xxs_packed *gate_row = gate_packed + (uint64_t)r * blocks;
            const bench_iq2_xxs_packed *up_row = up_packed + (uint64_t)r * blocks;
            for (uint32_t p = 0; p < pairs; p++) {
                bench_iq2_packed_pair_q8_K((int)(blocks * QK_K),
                                           gate_out + (uint64_t)r * pairs + p,
                                           up_out + (uint64_t)r * pairs + p,
                                           gate_row,
                                           up_row,
                                           xq + (uint64_t)p * blocks);
            }
        }
    }
    return now_sec() - t0;
#else
    (void)gate_packed;
    (void)up_packed;
    return bench_baseline(gate, up, xq, gate_out, up_out, rows, pairs, blocks, iters);
#endif
}

static double bench_meta(
        const block_iq2_xxs      *gate,
        const block_iq2_xxs      *up,
        const bench_iq2_xxs_meta *gate_meta,
        const bench_iq2_xxs_meta *up_meta,
        const block_q8_K         *xq,
        float                    *gate_out,
        float                    *up_out,
        uint32_t                  rows,
        uint32_t                  pairs,
        uint32_t                  blocks,
        uint32_t                  iters) {
#if defined(__AVX2__)
    (void)gate;
    (void)up;
    const double t0 = now_sec();
    for (uint32_t it = 0; it < iters; it++) {
        for (uint32_t r = 0; r < rows; r++) {
            const bench_iq2_xxs_meta *gate_row = gate_meta + (uint64_t)r * blocks;
            const bench_iq2_xxs_meta *up_row = up_meta + (uint64_t)r * blocks;
            for (uint32_t p = 0; p < pairs; p++) {
                bench_iq2_meta_pair_q8_K((int)(blocks * QK_K),
                                         gate_out + (uint64_t)r * pairs + p,
                                         up_out + (uint64_t)r * pairs + p,
                                         gate_row,
                                         up_row,
                                         xq + (uint64_t)p * blocks);
            }
        }
    }
    return now_sec() - t0;
#else
    (void)gate_meta;
    (void)up_meta;
    return bench_baseline(gate, up, xq, gate_out, up_out, rows, pairs, blocks, iters);
#endif
}

typedef enum {
    BENCH_PARALLEL_BASELINE = 0,
    BENCH_PARALLEL_PACKED = 1,
    BENCH_PARALLEL_META = 2,
} bench_parallel_path;

typedef struct {
    const block_iq2_xxs        *gate;
    const block_iq2_xxs        *up;
    const bench_iq2_xxs_packed *gate_packed;
    const bench_iq2_xxs_packed *up_packed;
    const bench_iq2_xxs_meta   *gate_meta;
    const bench_iq2_xxs_meta   *up_meta;
    const block_q8_K           *xq;
    float                      *gate_out;
    float                      *up_out;
    uint32_t                    pairs;
    uint32_t                    blocks;
    bench_parallel_path         path;
} bench_parallel_ctx;

static void bench_parallel_worker(void *vctx, uint64_t task0, uint64_t task1) {
    bench_parallel_ctx *ctx = vctx;
    for (uint64_t task = task0; task < task1; task++) {
        const uint32_t r = (uint32_t)(task / ctx->pairs);
        const uint32_t p = (uint32_t)(task - (uint64_t)r * ctx->pairs);
        const block_q8_K *xq = ctx->xq + (uint64_t)p * ctx->blocks;
        float *gate_out = ctx->gate_out + (uint64_t)r * ctx->pairs + p;
        float *up_out = ctx->up_out + (uint64_t)r * ctx->pairs + p;

#if defined(__AVX2__)
        if (ctx->path == BENCH_PARALLEL_PACKED) {
            const bench_iq2_xxs_packed *gate_row = ctx->gate_packed + (uint64_t)r * ctx->blocks;
            const bench_iq2_xxs_packed *up_row = ctx->up_packed + (uint64_t)r * ctx->blocks;
            bench_iq2_packed_pair_q8_K((int)(ctx->blocks * QK_K),
                                       gate_out,
                                       up_out,
                                       gate_row,
                                       up_row,
                                       xq);
            continue;
        }
        if (ctx->path == BENCH_PARALLEL_META) {
            const bench_iq2_xxs_meta *gate_row = ctx->gate_meta + (uint64_t)r * ctx->blocks;
            const bench_iq2_xxs_meta *up_row = ctx->up_meta + (uint64_t)r * ctx->blocks;
            bench_iq2_meta_pair_q8_K((int)(ctx->blocks * QK_K),
                                     gate_out,
                                     up_out,
                                     gate_row,
                                     up_row,
                                     xq);
            continue;
        }
#endif

        const block_iq2_xxs *gate_row = ctx->gate + (uint64_t)r * ctx->blocks;
        const block_iq2_xxs *up_row = ctx->up + (uint64_t)r * ctx->blocks;
        ds4_vec_dot_iq2_xxs_pair_q8_K((int)(ctx->blocks * QK_K),
                                      gate_out,
                                      up_out,
                                      gate_row,
                                      up_row,
                                      xq);
    }
}

static double bench_parallel(
        const block_iq2_xxs        *gate,
        const block_iq2_xxs        *up,
        const bench_iq2_xxs_packed *gate_packed,
        const bench_iq2_xxs_packed *up_packed,
        const bench_iq2_xxs_meta   *gate_meta,
        const bench_iq2_xxs_meta   *up_meta,
        const block_q8_K           *xq,
        float                      *gate_out,
        float                      *up_out,
        uint32_t                    rows,
        uint32_t                    pairs,
        uint32_t                    blocks,
        uint32_t                    iters,
        bench_parallel_path         path) {
    bench_parallel_ctx ctx = {
        .gate = gate,
        .up = up,
        .gate_packed = gate_packed,
        .up_packed = up_packed,
        .gate_meta = gate_meta,
        .up_meta = up_meta,
        .xq = xq,
        .gate_out = gate_out,
        .up_out = up_out,
        .pairs = pairs,
        .blocks = blocks,
        .path = path,
    };
    const uint64_t tasks = (uint64_t)rows * pairs;
    const double t0 = now_sec();
    for (uint32_t it = 0; it < iters; it++) {
        ds4_parallel_for(tasks, bench_parallel_worker, &ctx);
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
    uint32_t threads = 1;

    for (int i = 1; i < argc; i++) {
        rows = bench_parse_u32(&i, argc, argv, "--rows", rows);
        pairs = bench_parse_u32(&i, argc, argv, "--pairs", pairs);
        dim = bench_parse_u32(&i, argc, argv, "--dim", dim);
        iters = bench_parse_u32(&i, argc, argv, "--iters", iters);
        threads = bench_parse_u32(&i, argc, argv, "--threads", threads);
        if (strcmp(argv[i], "--help") == 0) {
            fprintf(stderr, "usage: %s [--rows N] [--pairs N] [--dim N] [--iters N] [--threads N]\n", argv[0]);
            return 0;
        }
    }
    if (dim % QK_K != 0) ds4_die("microbench dim must be QK_K aligned");
    g_requested_threads = threads;

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
    float *row2_g = xmalloc((size_t)out_elems * sizeof(row2_g[0]));
    float *row2_u = xmalloc((size_t)out_elems * sizeof(row2_u[0]));
    float *packed_g = xmalloc((size_t)out_elems * sizeof(packed_g[0]));
    float *packed_u = xmalloc((size_t)out_elems * sizeof(packed_u[0]));
    float *meta_g = xmalloc((size_t)out_elems * sizeof(meta_g[0]));
    float *meta_u = xmalloc((size_t)out_elems * sizeof(meta_u[0]));
    bench_iq2_xxs_packed *gate_packed = NULL;
    bench_iq2_xxs_packed *up_packed = NULL;
    bench_iq2_xxs_meta *gate_meta = NULL;
    bench_iq2_xxs_meta *up_meta = NULL;
    double pack_seconds = 0.0;
    double meta_pack_seconds = 0.0;

    bench_fill_iq2_rows(gate, rows, blocks, &rng);
    bench_fill_iq2_rows(up, rows, blocks, &rng);
    bench_fill_q8_rows(xq, pairs, dim, &rng);

#if defined(__AVX2__)
    gate_packed = xmalloc((size_t)((uint64_t)rows * blocks) * sizeof(gate_packed[0]));
    up_packed = xmalloc((size_t)((uint64_t)rows * blocks) * sizeof(up_packed[0]));
    gate_meta = xmalloc((size_t)((uint64_t)rows * blocks) * sizeof(gate_meta[0]));
    up_meta = xmalloc((size_t)((uint64_t)rows * blocks) * sizeof(up_meta[0]));
    const double pack_t0 = now_sec();
    bench_pack_iq2_rows(gate_packed, gate, rows, blocks);
    bench_pack_iq2_rows(up_packed, up, rows, blocks);
    pack_seconds = now_sec() - pack_t0;
    const double meta_pack_t0 = now_sec();
    bench_pack_iq2_meta_rows(gate_meta, gate, rows, blocks);
    bench_pack_iq2_meta_rows(up_meta, up, rows, blocks);
    meta_pack_seconds = now_sec() - meta_pack_t0;
#endif

    (void)bench_baseline(gate, up, xq, base_g, base_u, rows, pairs, blocks, 1);
    (void)bench_panel4(gate, up, xq, panel_g, panel_u, rows, pairs, blocks, 1);
    (void)bench_row2(gate, up, xq, row2_g, row2_u, rows, pairs, blocks, 1);
    (void)bench_packed(gate, up, gate_packed, up_packed, xq, packed_g, packed_u,
                       rows, pairs, blocks, 1);
    (void)bench_meta(gate, up, gate_meta, up_meta, xq, meta_g, meta_u,
                     rows, pairs, blocks, 1);

    double panel_max_abs = 0.0;
    double row2_max_abs = 0.0;
    double packed_max_abs = 0.0;
    double meta_max_abs = 0.0;
    for (uint64_t i = 0; i < out_elems; i++) {
        const double pdg = fabs((double)base_g[i] - (double)panel_g[i]);
        const double pdu = fabs((double)base_u[i] - (double)panel_u[i]);
        const double rdg = fabs((double)base_g[i] - (double)row2_g[i]);
        const double rdu = fabs((double)base_u[i] - (double)row2_u[i]);
        const double xdg = fabs((double)base_g[i] - (double)packed_g[i]);
        const double xdu = fabs((double)base_u[i] - (double)packed_u[i]);
        const double mdg = fabs((double)base_g[i] - (double)meta_g[i]);
        const double mdu = fabs((double)base_u[i] - (double)meta_u[i]);
        if (pdg > panel_max_abs) panel_max_abs = pdg;
        if (pdu > panel_max_abs) panel_max_abs = pdu;
        if (rdg > row2_max_abs) row2_max_abs = rdg;
        if (rdu > row2_max_abs) row2_max_abs = rdu;
        if (xdg > packed_max_abs) packed_max_abs = xdg;
        if (xdu > packed_max_abs) packed_max_abs = xdu;
        if (mdg > meta_max_abs) meta_max_abs = mdg;
        if (mdu > meta_max_abs) meta_max_abs = mdu;
    }

    const double tb = bench_baseline(gate, up, xq, base_g, base_u, rows, pairs, blocks, iters);
    const double tp = bench_panel4(gate, up, xq, panel_g, panel_u, rows, pairs, blocks, iters);
    const double tr = bench_row2(gate, up, xq, row2_g, row2_u, rows, pairs, blocks, iters);
    const double tx = bench_packed(gate, up, gate_packed, up_packed, xq, packed_g, packed_u,
                                   rows, pairs, blocks, iters);
    const double tm = bench_meta(gate, up, gate_meta, up_meta, xq, meta_g, meta_u,
                                 rows, pairs, blocks, iters);
    double tpb = 0.0;
    double tpx = 0.0;
    double tpm = 0.0;
    if (threads > 1) {
        tpb = bench_parallel(gate, up, gate_packed, up_packed, gate_meta, up_meta, xq,
                             base_g, base_u, rows, pairs, blocks, iters,
                             BENCH_PARALLEL_BASELINE);
        tpm = bench_parallel(gate, up, gate_packed, up_packed, gate_meta, up_meta, xq,
                             meta_g, meta_u, rows, pairs, blocks, iters,
                             BENCH_PARALLEL_META);
        tpx = bench_parallel(gate, up, gate_packed, up_packed, gate_meta, up_meta, xq,
                             packed_g, packed_u, rows, pairs, blocks, iters,
                             BENCH_PARALLEL_PACKED);
    }
    const double dots = (double)rows * (double)pairs * (double)iters;
    const double checksum = bench_checksum(base_g, base_u, out_elems) +
                            bench_checksum(panel_g, panel_u, out_elems) +
                            bench_checksum(row2_g, row2_u, out_elems) +
                            bench_checksum(packed_g, packed_u, out_elems) +
                            bench_checksum(meta_g, meta_u, out_elems);
    const double original_mb = 2.0 * (double)((uint64_t)rows * blocks * sizeof(block_iq2_xxs)) /
                               (1024.0 * 1024.0);
    const double packed_mb =
#if defined(__AVX2__)
        2.0 * (double)((uint64_t)rows * blocks * sizeof(bench_iq2_xxs_packed)) /
        (1024.0 * 1024.0);
#else
        0.0;
#endif
    const double meta_mb =
#if defined(__AVX2__)
        2.0 * (double)((uint64_t)rows * blocks * sizeof(bench_iq2_xxs_meta)) /
        (1024.0 * 1024.0);
#else
        0.0;
#endif

    ds4_threads_init();
    const uint32_t active_threads = g_pool.n_threads ? g_pool.n_threads : 1;
    printf("rows=%u pairs=%u dim=%u iters=%u threads=%u avx2=%s\n",
           rows, pairs, dim, iters, threads > 1 ? active_threads : 1,
#if defined(__AVX2__)
           "yes"
#else
           "no"
#endif
    );
    printf("correctness panel_max_abs_diff=%.9g row2_max_abs_diff=%.9g packed_max_abs_diff=%.9g meta_max_abs_diff=%.9g checksum=%.9f\n",
           panel_max_abs,
           row2_max_abs,
           packed_max_abs,
           meta_max_abs,
           checksum);
    printf("packed_iq2 original_mb=%.3f packed_mb=%.3f expansion=%.3fx pack_seconds=%.6f\n",
           original_mb,
           packed_mb,
           original_mb > 0.0 ? packed_mb / original_mb : 0.0,
           pack_seconds);
    printf("meta_iq2 original_mb=%.3f meta_mb=%.3f expansion=%.3fx pack_seconds=%.6f\n",
           original_mb,
           meta_mb,
           original_mb > 0.0 ? meta_mb / original_mb : 0.0,
           meta_pack_seconds);
    printf("baseline seconds=%.6f dots_per_s=%.3f\n", tb, dots / tb);
    printf("panel4+2 seconds=%.6f dots_per_s=%.3f speedup=%.3fx\n", tp, dots / tp, tb / tp);
    printf("row2 seconds=%.6f dots_per_s=%.3f speedup=%.3fx\n", tr, dots / tr, tb / tr);
    printf("packed seconds=%.6f dots_per_s=%.3f speedup=%.3fx\n", tx, dots / tx, tb / tx);
    printf("meta seconds=%.6f dots_per_s=%.3f speedup=%.3fx\n", tm, dots / tm, tb / tm);
    if (threads > 1) {
        printf("parallel_baseline seconds=%.6f dots_per_s=%.3f\n", tpb, dots / tpb);
        printf("parallel_meta seconds=%.6f dots_per_s=%.3f speedup=%.3fx\n",
               tpm, dots / tpm, tpb / tpm);
        printf("parallel_packed seconds=%.6f dots_per_s=%.3f speedup=%.3fx\n",
               tpx, dots / tpx, tpb / tpx);
    }

    ds4_threads_shutdown();
    free(up_meta);
    free(gate_meta);
    free(up_packed);
    free(gate_packed);
    free(meta_u);
    free(meta_g);
    free(packed_u);
    free(packed_g);
    free(row2_u);
    free(row2_g);
    free(panel_u);
    free(panel_g);
    free(base_u);
    free(base_g);
    free(xq);
    free(up);
    free(gate);
    return panel_max_abs < 1.0e-3 &&
           row2_max_abs < 1.0e-3 &&
           packed_max_abs < 1.0e-3 &&
           meta_max_abs < 1.0e-3 ? 0 : 1;
}
