#include "ds4_warm.h"

#include <errno.h>
#include <inttypes.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    const char *model_path;
    const char *host;
    const char *port;
    const char *backend;
    int threads;
    bool once;
    bool require_resident;
} worker_config;

typedef struct {
    ds4_engine *engine;
    ds4_warm_model_info model;
    bool resident[43][256];
    uint32_t resident_count;
    bool require_resident;
    bool use_metal;
    ds4_warm_stats_response stats;
} worker_state;

static void usage(FILE *fp) {
    fprintf(fp,
            "usage: ds4-warm-worker --model FILE [options]\n"
            "\n"
            "Options:\n"
            "  --host ADDR              Listen address (default: 0.0.0.0).\n"
            "  --port PORT              Listen port (default: 9044).\n"
            "  --backend cpu|metal      Expert execution backend (default: cpu).\n"
            "  --threads N              CPU reference backend threads.\n"
            "  --require-resident       Reject RUN_LAYER for experts not loaded first.\n"
            "  --once                   Handle one client, then exit.\n"
            "  -h, --help               Show this help.\n");
}

static const char *need_arg(int *i, int argc, char **argv, const char *arg) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "%s requires a value\n", arg);
        exit(2);
    }
    return argv[++(*i)];
}

static worker_config parse_args(int argc, char **argv) {
    worker_config c = {
        .host = "0.0.0.0",
        .port = "9044",
        .backend = "cpu",
        .threads = 0,
    };

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--model")) {
            c.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--host")) {
            c.host = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--port")) {
            c.port = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--backend")) {
            c.backend = need_arg(&i, argc, argv, arg);
            if (strcmp(c.backend, "cpu") && strcmp(c.backend, "metal")) {
                fprintf(stderr, "--backend must be cpu or metal\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--threads")) {
            c.threads = atoi(need_arg(&i, argc, argv, arg));
            if (c.threads < 0) c.threads = 0;
        } else if (!strcmp(arg, "--require-resident")) {
            c.require_resident = true;
        } else if (!strcmp(arg, "--once")) {
            c.once = true;
        } else if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            exit(0);
        } else {
            fprintf(stderr, "unknown argument: %s\n", arg);
            usage(stderr);
            exit(2);
        }
    }

    if (!c.model_path) {
        fprintf(stderr, "missing --model FILE\n");
        usage(stderr);
        exit(2);
    }
    return c;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static void *xmalloc_worker(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "ds4-warm-worker: out of memory allocating %zu bytes\n", n);
        exit(1);
    }
    return p;
}

static int read_exact(int fd, void *buf, size_t n) {
    uint8_t *p = buf;
    while (n != 0) {
        ssize_t r = read(fd, p, n);
        if (r == 0) return 0;
        if (r < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        p += (size_t)r;
        n -= (size_t)r;
    }
    return 1;
}

static int write_exact(int fd, const void *buf, size_t n) {
    const uint8_t *p = buf;
    while (n != 0) {
        ssize_t r = write(fd, p, n);
        if (r < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        p += (size_t)r;
        n -= (size_t)r;
    }
    return 0;
}

static int send_frame(int fd, uint16_t opcode, uint32_t sequence, const void *payload, uint32_t payload_len) {
    ds4_warm_frame_header h = {
        .magic = DS4_WARM_MAGIC,
        .version = DS4_WARM_PROTOCOL_VERSION,
        .opcode = opcode,
        .sequence = sequence,
        .payload_len = payload_len,
    };
    if (write_exact(fd, &h, sizeof(h)) != 0) return -1;
    if (payload_len != 0 && write_exact(fd, payload, payload_len) != 0) return -1;
    return 0;
}

static int send_status_run(int fd, uint32_t sequence, uint32_t status) {
    ds4_warm_run_layer_response resp = {
        .status = status,
        .n_tok = 0,
        .output_dim = 0,
    };
    return send_frame(fd, DS4_WARM_OP_RUN_LAYER, sequence, &resp, sizeof(resp));
}

static uint32_t mark_experts(worker_state *st, const ds4_warm_expert_id *ids, uint32_t count, bool resident) {
    uint32_t accepted = 0;
    for (uint32_t i = 0; i < count; i++) {
        const uint32_t layer = ids[i].layer;
        const uint32_t expert = ids[i].expert;
        if (layer >= st->model.n_layer || expert >= st->model.n_expert) continue;
        if (st->resident[layer][expert] != resident) {
            st->resident[layer][expert] = resident;
            if (resident) st->resident_count++;
            else st->resident_count--;
        }
        accepted++;
    }
    return accepted;
}

static bool all_selected_resident(worker_state *st, const int32_t *selected, uint32_t n_tok, uint32_t n_selected, uint32_t layer) {
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t i = 0; i < n_selected; i++) {
            const int32_t expert = selected[(uint64_t)t * n_selected + i];
            if (expert < 0 || (uint32_t)expert >= st->model.n_expert) return false;
            if (!st->resident[layer][expert]) return false;
        }
    }
    return true;
}

static int handle_hello(worker_state *st, int fd, const ds4_warm_frame_header *h) {
    ds4_warm_hello_response resp = {
        .status = DS4_WARM_STATUS_OK,
        .protocol_version = DS4_WARM_PROTOCOL_VERSION,
        .model = st->model,
    };
    return send_frame(fd, h->opcode, h->sequence, &resp, sizeof(resp));
}

static int handle_expert_list(worker_state *st, int fd, const ds4_warm_frame_header *h, const uint8_t *payload) {
    if (h->payload_len < sizeof(ds4_warm_expert_list_header)) {
        ds4_warm_expert_list_response resp = { .status = DS4_WARM_STATUS_BAD_REQUEST };
        return send_frame(fd, h->opcode, h->sequence, &resp, sizeof(resp));
    }
    const ds4_warm_expert_list_header *req = (const ds4_warm_expert_list_header *)payload;
    const uint64_t need = sizeof(*req) + (uint64_t)req->count * sizeof(ds4_warm_expert_id);
    if (need > h->payload_len) {
        ds4_warm_expert_list_response resp = { .status = DS4_WARM_STATUS_BAD_REQUEST };
        return send_frame(fd, h->opcode, h->sequence, &resp, sizeof(resp));
    }

    const ds4_warm_expert_id *ids = (const ds4_warm_expert_id *)(payload + sizeof(*req));
    const bool resident = h->opcode == DS4_WARM_OP_LOAD_EXPERTS;
    const uint32_t accepted = mark_experts(st, ids, req->count, resident);
    ds4_warm_expert_list_response resp = {
        .status = DS4_WARM_STATUS_OK,
        .accepted = accepted,
        .resident = st->resident_count,
    };
    return send_frame(fd, h->opcode, h->sequence, &resp, sizeof(resp));
}

static int handle_stats(worker_state *st, int fd, const ds4_warm_frame_header *h) {
    return send_frame(fd, h->opcode, h->sequence, &st->stats, sizeof(st->stats));
}

static int handle_run_layer(worker_state *st, int fd, const ds4_warm_frame_header *h, const uint8_t *payload) {
    if (h->payload_len < sizeof(ds4_warm_run_layer_header)) {
        return send_status_run(fd, h->sequence, DS4_WARM_STATUS_BAD_REQUEST);
    }

    const ds4_warm_run_layer_header *req = (const ds4_warm_run_layer_header *)payload;
    if (req->layer >= st->model.n_layer ||
        req->n_tok == 0 ||
        req->n_selected > st->model.n_expert_used ||
        req->input_format != DS4_WARM_INPUT_F32 ||
        req->output_format != DS4_WARM_OUTPUT_F32) {
        return send_status_run(fd, h->sequence, DS4_WARM_STATUS_UNSUPPORTED);
    }

    const uint64_t slots = (uint64_t)req->n_tok * req->n_selected;
    const uint64_t selected_bytes = slots * sizeof(int32_t);
    const uint64_t weight_bytes = slots * sizeof(float);
    const uint64_t x_bytes = (uint64_t)req->n_tok * st->model.n_embd * sizeof(float);
    const uint64_t need = sizeof(*req) + selected_bytes + weight_bytes + x_bytes;
    if (need != h->payload_len || need > SIZE_MAX) {
        return send_status_run(fd, h->sequence, DS4_WARM_STATUS_BAD_REQUEST);
    }

    const int32_t *selected = (const int32_t *)(payload + sizeof(*req));
    const float *weights = (const float *)(payload + sizeof(*req) + selected_bytes);
    const float *x = (const float *)(payload + sizeof(*req) + selected_bytes + weight_bytes);

    if (st->require_resident && !all_selected_resident(st, selected, req->n_tok, req->n_selected, req->layer)) {
        st->stats.resident_misses += slots;
        return send_status_run(fd, h->sequence, DS4_WARM_STATUS_NOT_RESIDENT);
    }

    const uint64_t out_bytes_u64 = (uint64_t)req->n_tok * st->model.n_embd * sizeof(float);
    if (out_bytes_u64 > UINT32_MAX || out_bytes_u64 > SIZE_MAX - sizeof(ds4_warm_run_layer_response)) {
        return send_status_run(fd, h->sequence, DS4_WARM_STATUS_BAD_REQUEST);
    }

    const size_t out_bytes = (size_t)out_bytes_u64;
    const size_t resp_payload_len = sizeof(ds4_warm_run_layer_response) + out_bytes;
    uint8_t *resp_payload = xmalloc_worker(resp_payload_len);
    ds4_warm_run_layer_response *resp = (ds4_warm_run_layer_response *)resp_payload;
    float *out = (float *)(resp_payload + sizeof(*resp));
    memset(resp, 0, sizeof(*resp));

    const double t0 = now_sec();
    const int rc = st->use_metal
        ? ds4_engine_warm_run_layer_metal_f32(
                st->engine,
                req->layer,
                x,
                req->n_tok,
                selected,
                weights,
                req->n_selected,
                out)
        : ds4_engine_warm_run_layer_f32(
                st->engine,
                req->layer,
                x,
                req->n_tok,
                selected,
                weights,
                req->n_selected,
                out);
    st->stats.compute_seconds += now_sec() - t0;

    if (rc != 0) {
        free(resp_payload);
        return send_status_run(fd, h->sequence, DS4_WARM_STATUS_RUNTIME_ERROR);
    }

    resp->status = DS4_WARM_STATUS_OK;
    resp->n_tok = req->n_tok;
    resp->output_dim = st->model.n_embd;

    st->stats.run_layer_requests++;
    st->stats.tokens += req->n_tok;
    st->stats.selected_slots += slots;
    st->stats.resident_hits += slots;

    const int wr = send_frame(fd, h->opcode, h->sequence, resp_payload, (uint32_t)resp_payload_len);
    free(resp_payload);
    return wr;
}

static int handle_frame(worker_state *st, int fd, const ds4_warm_frame_header *h, const uint8_t *payload) {
    st->stats.requests++;
    switch ((ds4_warm_opcode)h->opcode) {
    case DS4_WARM_OP_HELLO:
        return handle_hello(st, fd, h);
    case DS4_WARM_OP_LOAD_EXPERTS:
    case DS4_WARM_OP_EVICT_EXPERTS:
        return handle_expert_list(st, fd, h, payload);
    case DS4_WARM_OP_RUN_LAYER:
        return handle_run_layer(st, fd, h, payload);
    case DS4_WARM_OP_STATS:
        return handle_stats(st, fd, h);
    default: {
        ds4_warm_expert_list_response resp = { .status = DS4_WARM_STATUS_UNSUPPORTED };
        return send_frame(fd, h->opcode, h->sequence, &resp, sizeof(resp));
    }
    }
}

static void serve_client(worker_state *st, int fd) {
    int one = 1;
    (void)setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    for (;;) {
        ds4_warm_frame_header h;
        const int r = read_exact(fd, &h, sizeof(h));
        if (r <= 0) break;
        if (h.magic != DS4_WARM_MAGIC || h.version != DS4_WARM_PROTOCOL_VERSION) break;
        if (h.payload_len > UINT32_C(512) * 1024u * 1024u) break;

        uint8_t *payload = NULL;
        if (h.payload_len != 0) {
            payload = xmalloc_worker(h.payload_len);
            if (read_exact(fd, payload, h.payload_len) != 1) {
                free(payload);
                break;
            }
        }

        const int rc = handle_frame(st, fd, &h, payload);
        free(payload);
        if (rc != 0) break;
    }
}

static int listen_socket(const char *host, const char *port) {
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;

    struct addrinfo *res = NULL;
    const int gai = getaddrinfo(host, port, &hints, &res);
    if (gai != 0) {
        fprintf(stderr, "getaddrinfo(%s,%s): %s\n", host, port, gai_strerror(gai));
        return -1;
    }

    int fd = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        int one = 1;
        (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        if (bind(fd, ai->ai_addr, ai->ai_addrlen) == 0 && listen(fd, 16) == 0) break;
        close(fd);
        fd = -1;
    }

    freeaddrinfo(res);
    return fd;
}

int main(int argc, char **argv) {
    signal(SIGPIPE, SIG_IGN);
    const worker_config cfg = parse_args(argc, argv);

    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = cfg.model_path;
    opt.backend = DS4_BACKEND_CPU;
    opt.n_threads = cfg.threads;

    worker_state st;
    memset(&st, 0, sizeof(st));
    st.require_resident = cfg.require_resident;
    st.use_metal = !strcmp(cfg.backend, "metal");
    if (ds4_engine_open(&st.engine, &opt) != 0) return 1;
    if (ds4_engine_warm_model_info(st.engine, &st.model) != 0) {
        fprintf(stderr, "ds4-warm-worker: failed to read warm model info\n");
        ds4_engine_close(st.engine);
        return 1;
    }

    fprintf(stderr,
            "ds4-warm-worker: model loaded layers=%u experts=%u top=%u embd=%u "
            "gate_type=%u up_type=%u down_type=%u backend=%s fingerprint=%016" PRIx64 "\n",
            st.model.n_layer,
            st.model.n_expert,
            st.model.n_expert_used,
            st.model.n_embd,
            st.model.gate_type,
            st.model.up_type,
            st.model.down_type,
            cfg.backend,
            st.model.model_fingerprint);

    const int lfd = listen_socket(cfg.host, cfg.port);
    if (lfd < 0) {
        ds4_engine_close(st.engine);
        return 1;
    }
    fprintf(stderr, "ds4-warm-worker: listening on %s:%s%s\n",
            cfg.host, cfg.port, st.require_resident ? " require_resident=1" : "");

    for (;;) {
        struct sockaddr_storage ss;
        socklen_t slen = sizeof(ss);
        int cfd = accept(lfd, (struct sockaddr *)&ss, &slen);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            perror("accept");
            break;
        }
        serve_client(&st, cfd);
        close(cfd);
        if (cfg.once) break;
    }

    close(lfd);
    ds4_engine_close(st.engine);
    return 0;
}
