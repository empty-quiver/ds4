#include "ds4_warm_client.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    const char *model_path;
    const char *host;
    const char *port;
    int threads;
    int timeout_ms;
    uint32_t layer;
    uint32_t n_selected;
    int iters;
    int warmup;
    uint32_t tok_counts[32];
    uint32_t n_tok_counts;
    bool local_only;
    bool remote_only;
} bench_config;

typedef struct {
    double min_s;
    double avg_s;
    double max_s;
} time_summary;

static void usage(FILE *fp) {
    fprintf(fp,
            "usage: tests/warm_worker_microbench --model FILE [options]\n"
            "\n"
            "Options:\n"
            "  --host HOST          Warm worker host (default: 127.0.0.1).\n"
            "  --port PORT          Warm worker port (default: 9044).\n"
            "  --threads N          Local CPU baseline threads.\n"
            "  --timeout-ms N       Socket timeout (default: 30000).\n"
            "  --layer N            Layer to test (default: 0).\n"
            "  --n-selected N       Selected experts per token (default: 6).\n"
            "  --tok-counts LIST    Comma list, e.g. 1,2,4,8,16,64.\n"
            "  --iters N            Timed iterations (default: 5).\n"
            "  --warmup N           Warmup iterations (default: 1).\n"
            "  --local-only         Skip remote calls.\n"
            "  --remote-only        Skip local baseline.\n"
            "  -h, --help           Show this help.\n");
}

static const char *need_arg(int *i, int argc, char **argv, const char *arg) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "%s requires a value\n", arg);
        exit(2);
    }
    return argv[++(*i)];
}

static uint32_t parse_u32(const char *s, const char *arg) {
    char *end = NULL;
    errno = 0;
    unsigned long v = strtoul(s, &end, 10);
    if (errno || !end || *end || v > UINT32_MAX) {
        fprintf(stderr, "invalid %s: %s\n", arg, s);
        exit(2);
    }
    return (uint32_t)v;
}

static int parse_int(const char *s, const char *arg) {
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || !end || *end || v < 0 || v > INT32_MAX) {
        fprintf(stderr, "invalid %s: %s\n", arg, s);
        exit(2);
    }
    return (int)v;
}

static void parse_tok_counts(bench_config *c, const char *s) {
    c->n_tok_counts = 0;
    const char *p = s;
    while (*p) {
        if (c->n_tok_counts >= sizeof(c->tok_counts) / sizeof(c->tok_counts[0])) {
            fprintf(stderr, "too many --tok-counts values\n");
            exit(2);
        }
        char *end = NULL;
        errno = 0;
        unsigned long v = strtoul(p, &end, 10);
        if (errno || end == p || v == 0 || v > UINT32_MAX) {
            fprintf(stderr, "invalid --tok-counts: %s\n", s);
            exit(2);
        }
        c->tok_counts[c->n_tok_counts++] = (uint32_t)v;
        if (*end == ',') p = end + 1;
        else if (*end == '\0') break;
        else {
            fprintf(stderr, "invalid --tok-counts: %s\n", s);
            exit(2);
        }
    }
}

static bench_config parse_args(int argc, char **argv) {
    bench_config c = {
        .host = "127.0.0.1",
        .port = "9044",
        .timeout_ms = 30000,
        .layer = 0,
        .n_selected = 6,
        .iters = 5,
        .warmup = 1,
    };
    parse_tok_counts(&c, "1,2,4,8,16,64");

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--model")) {
            c.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--host")) {
            c.host = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--port")) {
            c.port = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--threads")) {
            c.threads = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--timeout-ms")) {
            c.timeout_ms = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--layer")) {
            c.layer = parse_u32(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--n-selected")) {
            c.n_selected = parse_u32(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--tok-counts")) {
            parse_tok_counts(&c, need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--iters")) {
            c.iters = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--warmup")) {
            c.warmup = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--local-only")) {
            c.local_only = true;
        } else if (!strcmp(arg, "--remote-only")) {
            c.remote_only = true;
        } else if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            exit(0);
        } else {
            fprintf(stderr, "unknown argument: %s\n", arg);
            usage(stderr);
            exit(2);
        }
    }
    if (!c.model_path && !c.remote_only) {
        fprintf(stderr, "missing --model FILE\n");
        usage(stderr);
        exit(2);
    }
    if (c.local_only && c.remote_only) {
        fprintf(stderr, "--local-only and --remote-only are mutually exclusive\n");
        exit(2);
    }
    if (c.iters <= 0) c.iters = 1;
    return c;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static void *xmalloc_bench(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "out of memory allocating %zu bytes\n", n);
        exit(1);
    }
    return p;
}

static uint32_t lcg_next(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static void fill_inputs(float *x, uint32_t n_tok, uint32_t n_embd, uint32_t seed) {
    uint32_t rng = seed ? seed : 1;
    for (uint64_t i = 0; i < (uint64_t)n_tok * n_embd; i++) {
        const uint32_t v = lcg_next(&rng);
        const int32_t centered = (int32_t)((v >> 8) & 0xffffu) - 32768;
        x[i] = (float)centered * (1.0f / 32768.0f) * 0.125f;
    }
}

static void fill_routes(
        int32_t *selected,
        float   *weights,
        uint32_t n_tok,
        uint32_t n_selected,
        uint32_t n_expert,
        uint32_t layer) {
    const float denom = (float)(n_selected * (n_selected + 1u)) * 0.5f;
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t i = 0; i < n_selected; i++) {
            selected[(uint64_t)t * n_selected + i] =
                (int32_t)((layer * 17u + t * 29u + i * 37u) % n_expert);
            weights[(uint64_t)t * n_selected + i] = (float)(n_selected - i) / denom;
        }
    }
}

static uint32_t build_unique_experts(
        ds4_warm_expert_id *ids,
        const int32_t      *selected,
        uint32_t            n_tok,
        uint32_t            n_selected,
        uint32_t            layer,
        uint32_t            n_expert) {
    bool seen[256];
    memset(seen, 0, sizeof(seen));
    uint32_t count = 0;
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t i = 0; i < n_selected; i++) {
            const int32_t expert_i = selected[(uint64_t)t * n_selected + i];
            if (expert_i < 0 || (uint32_t)expert_i >= n_expert) continue;
            const uint32_t expert = (uint32_t)expert_i;
            if (!seen[expert]) {
                seen[expert] = true;
                ids[count++] = (ds4_warm_expert_id){ .layer = layer, .expert = expert };
            }
        }
    }
    return count;
}

static double checksum(const float *x, uint64_t n) {
    double s = 0.0;
    for (uint64_t i = 0; i < n; i++) s += x[i];
    return s;
}

static double max_abs_diff(const float *a, const float *b, uint64_t n) {
    double m = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        const double d = fabs((double)a[i] - (double)b[i]);
        if (d > m) m = d;
    }
    return m;
}

static time_summary summarize(const double *times, int n) {
    time_summary s = { .min_s = times[0], .avg_s = 0.0, .max_s = times[0] };
    for (int i = 0; i < n; i++) {
        if (times[i] < s.min_s) s.min_s = times[i];
        if (times[i] > s.max_s) s.max_s = times[i];
        s.avg_s += times[i];
    }
    s.avg_s /= (double)n;
    return s;
}

static bool model_info_compatible(const ds4_warm_model_info *a, const ds4_warm_model_info *b) {
    return a->n_layer == b->n_layer &&
           a->n_embd == b->n_embd &&
           a->n_expert == b->n_expert &&
           a->n_expert_used == b->n_expert_used &&
           a->n_ff_exp == b->n_ff_exp &&
           a->gate_type == b->gate_type &&
           a->up_type == b->up_type &&
           a->down_type == b->down_type &&
           a->gate_expert_bytes == b->gate_expert_bytes &&
           a->up_expert_bytes == b->up_expert_bytes &&
           a->down_expert_bytes == b->down_expert_bytes &&
           a->model_size == b->model_size &&
           a->model_fingerprint == b->model_fingerprint;
}

int main(int argc, char **argv) {
    signal(SIGPIPE, SIG_IGN);
    const bench_config cfg = parse_args(argc, argv);

    ds4_engine *engine = NULL;
    ds4_warm_model_info local_info;
    memset(&local_info, 0, sizeof(local_info));

    if (!cfg.remote_only) {
        ds4_engine_options opt;
        memset(&opt, 0, sizeof(opt));
        opt.model_path = cfg.model_path;
        opt.backend = DS4_BACKEND_CPU;
        opt.n_threads = cfg.threads;

        if (ds4_engine_open(&engine, &opt) != 0) return 1;
        if (ds4_engine_warm_model_info(engine, &local_info) != 0) {
            fprintf(stderr, "failed to read local model info\n");
            ds4_engine_close(engine);
            return 1;
        }
    }

    ds4_warm_client *client = NULL;
    ds4_warm_model_info remote_info;
    memset(&remote_info, 0, sizeof(remote_info));
    if (!cfg.local_only) {
        if (ds4_warm_client_connect(&client, cfg.host, cfg.port, cfg.timeout_ms) != 0) {
            fprintf(stderr, "failed to connect to warm worker at %s:%s\n", cfg.host, cfg.port);
            ds4_engine_close(engine);
            return 1;
        }
        if (ds4_warm_client_hello(client, &remote_info) != 0) {
            fprintf(stderr, "HELLO failed: %s\n", ds4_warm_client_error(client));
            ds4_warm_client_close(client);
            if (engine) ds4_engine_close(engine);
            return 1;
        }
        if (cfg.remote_only) {
            local_info = remote_info;
        } else if (!model_info_compatible(&local_info, &remote_info)) {
            fprintf(stderr, "remote model layout does not match local model\n");
            ds4_warm_client_close(client);
            if (engine) ds4_engine_close(engine);
            return 1;
        }
    }
    if (cfg.layer >= local_info.n_layer || cfg.n_selected > local_info.n_expert_used) {
        fprintf(stderr, "layer/n-selected outside model shape\n");
        if (client) ds4_warm_client_close(client);
        if (engine) ds4_engine_close(engine);
        return 2;
    }

    printf("n_tok,slots,unique_experts,local_avg_ms,local_min_ms,local_max_ms,remote_avg_ms,remote_min_ms,remote_max_ms,remote_req_bytes,remote_resp_bytes,max_abs_diff,local_checksum,remote_checksum\n");

    for (uint32_t ti = 0; ti < cfg.n_tok_counts; ti++) {
        const uint32_t n_tok = cfg.tok_counts[ti];
        const uint64_t slots = (uint64_t)n_tok * cfg.n_selected;
        const uint64_t out_elems = (uint64_t)n_tok * local_info.n_embd;
        float *x = xmalloc_bench(out_elems * sizeof(x[0]));
        int32_t *selected = xmalloc_bench(slots * sizeof(selected[0]));
        float *weights = xmalloc_bench(slots * sizeof(weights[0]));
        float *local_out = xmalloc_bench(out_elems * sizeof(local_out[0]));
        float *remote_out = xmalloc_bench(out_elems * sizeof(remote_out[0]));
        double *local_times = xmalloc_bench((size_t)cfg.iters * sizeof(local_times[0]));
        double *remote_times = xmalloc_bench((size_t)cfg.iters * sizeof(remote_times[0]));

        fill_inputs(x, n_tok, local_info.n_embd, 1234u + n_tok);
        fill_routes(selected, weights, n_tok, cfg.n_selected, local_info.n_expert, cfg.layer);

        ds4_warm_expert_id ids[256];
        const uint32_t unique = build_unique_experts(ids, selected, n_tok, cfg.n_selected,
                                                     cfg.layer, local_info.n_expert);
        if (client) {
            ds4_warm_expert_list_response load_resp;
            if (ds4_warm_client_load_experts(client, ids, unique, &load_resp) != 0) {
                fprintf(stderr, "LOAD_EXPERTS failed: %s\n", ds4_warm_client_error(client));
                return 1;
            }
        }

        if (!cfg.remote_only) {
            for (int i = 0; i < cfg.warmup; i++) {
                if (ds4_engine_warm_run_layer_f32(engine, cfg.layer, x, n_tok, selected,
                                                  weights, cfg.n_selected, local_out) != 0) {
                    fprintf(stderr, "local warm helper failed\n");
                    return 1;
                }
            }
            for (int i = 0; i < cfg.iters; i++) {
                const double t0 = now_sec();
                if (ds4_engine_warm_run_layer_f32(engine, cfg.layer, x, n_tok, selected,
                                                  weights, cfg.n_selected, local_out) != 0) {
                    fprintf(stderr, "local warm helper failed\n");
                    return 1;
                }
                local_times[i] = now_sec() - t0;
            }
        } else {
            memset(local_out, 0, out_elems * sizeof(local_out[0]));
            for (int i = 0; i < cfg.iters; i++) local_times[i] = 0.0;
        }

        uint64_t req_bytes = 0;
        uint64_t resp_bytes = 0;
        if (client) {
            ds4_warm_client_timing timing;
            for (int i = 0; i < cfg.warmup; i++) {
                if (ds4_warm_client_run_layer_f32(client, cfg.layer, x, n_tok, selected,
                                                  weights, cfg.n_selected, remote_out,
                                                  local_info.n_embd, &timing) != 0) {
                    fprintf(stderr, "remote RUN_LAYER failed: %s\n", ds4_warm_client_error(client));
                    return 1;
                }
            }
            for (int i = 0; i < cfg.iters; i++) {
                if (ds4_warm_client_run_layer_f32(client, cfg.layer, x, n_tok, selected,
                                                  weights, cfg.n_selected, remote_out,
                                                  local_info.n_embd, &timing) != 0) {
                    fprintf(stderr, "remote RUN_LAYER failed: %s\n", ds4_warm_client_error(client));
                    return 1;
                }
                remote_times[i] = timing.elapsed_seconds;
                req_bytes = timing.request_bytes;
                resp_bytes = timing.response_bytes;
            }
        } else {
            memset(remote_out, 0, out_elems * sizeof(remote_out[0]));
            for (int i = 0; i < cfg.iters; i++) remote_times[i] = 0.0;
        }

        const time_summary local = summarize(local_times, cfg.iters);
        const time_summary remote = summarize(remote_times, cfg.iters);
        const double local_sum = checksum(local_out, out_elems);
        const double remote_sum = checksum(remote_out, out_elems);
        const double diff = (!cfg.remote_only && client) ? max_abs_diff(local_out, remote_out, out_elems) : 0.0;

        printf("%u,%" PRIu64 ",%u,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%" PRIu64 ",%" PRIu64 ",%.9g,%.9g,%.9g\n",
               n_tok,
               slots,
               unique,
               local.avg_s * 1000.0,
               local.min_s * 1000.0,
               local.max_s * 1000.0,
               remote.avg_s * 1000.0,
               remote.min_s * 1000.0,
               remote.max_s * 1000.0,
               req_bytes,
               resp_bytes,
               diff,
               local_sum,
               remote_sum);
        fflush(stdout);

        free(remote_times);
        free(local_times);
        free(remote_out);
        free(local_out);
        free(weights);
        free(selected);
        free(x);
    }

    if (client) {
        ds4_warm_stats_response stats;
        if (ds4_warm_client_stats(client, &stats) == 0) {
            fprintf(stderr,
                    "warm_worker_stats requests=%" PRIu64
                    " run_layer=%" PRIu64
                    " tokens=%" PRIu64
                    " selected_slots=%" PRIu64
                    " compute_s=%.6f\n",
                    stats.requests,
                    stats.run_layer_requests,
                    stats.tokens,
                    stats.selected_slots,
                    stats.compute_seconds);
        }
        ds4_warm_client_close(client);
    }
    if (engine) ds4_engine_close(engine);
    return 0;
}
