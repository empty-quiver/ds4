#include "ds4_warm_client.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

struct ds4_warm_client {
    int fd;
    uint32_t seq;
    int timeout_ms;
    char error[256];
};

static double warm_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static void warm_set_error(ds4_warm_client *c, const char *msg) {
    if (!c) return;
    snprintf(c->error, sizeof(c->error), "%s", msg ? msg : "unknown error");
}

static void *warm_malloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) return NULL;
    return p;
}

static float warm_bf16_to_f32(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

const char *ds4_warm_client_error(ds4_warm_client *c) {
    if (!c || !c->error[0]) return "";
    return c->error;
}

static int wait_fd(int fd, bool writeable, int timeout_ms) {
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(fd, &fds);
    struct timeval tv;
    struct timeval *ptv = NULL;
    if (timeout_ms >= 0) {
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
        ptv = &tv;
    }
    int rc;
    do {
        rc = select(fd + 1,
                    writeable ? NULL : &fds,
                    writeable ? &fds : NULL,
                    NULL,
                    ptv);
    } while (rc < 0 && errno == EINTR);
    return rc;
}

static int set_blocking(int fd, bool blocking) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return -1;
    if (blocking) flags &= ~O_NONBLOCK;
    else flags |= O_NONBLOCK;
    return fcntl(fd, F_SETFL, flags);
}

static int connect_one(const struct addrinfo *ai, int timeout_ms) {
    int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (fd < 0) return -1;

    int one = 1;
    (void)setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    if (timeout_ms >= 0 && set_blocking(fd, false) != 0) {
        close(fd);
        return -1;
    }

    int rc = connect(fd, ai->ai_addr, ai->ai_addrlen);
    if (rc != 0 && timeout_ms >= 0 && errno == EINPROGRESS) {
        rc = wait_fd(fd, true, timeout_ms);
        if (rc > 0) {
            int err = 0;
            socklen_t len = sizeof(err);
            if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len) != 0 || err != 0) {
                rc = -1;
                errno = err ? err : errno;
            } else {
                rc = 0;
            }
        } else if (rc == 0) {
            rc = -1;
            errno = ETIMEDOUT;
        } else {
            rc = -1;
        }
    }
    if (rc != 0) {
        close(fd);
        return -1;
    }
    if (timeout_ms >= 0 && set_blocking(fd, true) != 0) {
        close(fd);
        return -1;
    }

    if (timeout_ms >= 0) {
        struct timeval tv = {
            .tv_sec = timeout_ms / 1000,
            .tv_usec = (timeout_ms % 1000) * 1000,
        };
        (void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        (void)setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    }
    return fd;
}

int ds4_warm_client_connect(
        ds4_warm_client **out,
        const char       *host,
        const char       *port,
        int               timeout_ms) {
    if (!out || !host || !port) return -1;
    *out = NULL;

    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    struct addrinfo *res = NULL;
    const int gai = getaddrinfo(host, port, &hints, &res);
    if (gai != 0) return -1;

    int fd = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        fd = connect_one(ai, timeout_ms);
        if (fd >= 0) break;
    }
    freeaddrinfo(res);
    if (fd < 0) return -1;

    ds4_warm_client *c = warm_malloc(sizeof(*c));
    if (!c) {
        close(fd);
        return -1;
    }
    memset(c, 0, sizeof(*c));
    c->fd = fd;
    c->timeout_ms = timeout_ms;
    *out = c;
    return 0;
}

void ds4_warm_client_close(ds4_warm_client *c) {
    if (!c) return;
    if (c->fd >= 0) close(c->fd);
    free(c);
}

static int read_exact(ds4_warm_client *c, void *buf, size_t n) {
    uint8_t *p = buf;
    while (n != 0) {
        ssize_t r = read(c->fd, p, n);
        if (r == 0) {
            warm_set_error(c, "connection closed");
            return -1;
        }
        if (r < 0) {
            if (errno == EINTR) continue;
            warm_set_error(c, strerror(errno));
            return -1;
        }
        p += (size_t)r;
        n -= (size_t)r;
    }
    return 0;
}

static int write_exact(ds4_warm_client *c, const void *buf, size_t n) {
    const uint8_t *p = buf;
    while (n != 0) {
        ssize_t r = write(c->fd, p, n);
        if (r < 0) {
            if (errno == EINTR) continue;
            warm_set_error(c, strerror(errno));
            return -1;
        }
        p += (size_t)r;
        n -= (size_t)r;
    }
    return 0;
}

static int transact(
        ds4_warm_client *c,
        uint16_t         opcode,
        const void      *payload,
        uint32_t         payload_len,
        void           **out_payload,
        uint32_t        *out_payload_len,
        ds4_warm_client_timing *timing) {
    if (!c || !out_payload || !out_payload_len) return -1;
    *out_payload = NULL;
    *out_payload_len = 0;

    const uint32_t seq = ++c->seq;
    ds4_warm_frame_header h = {
        .magic = DS4_WARM_MAGIC,
        .version = DS4_WARM_PROTOCOL_VERSION,
        .opcode = opcode,
        .sequence = seq,
        .payload_len = payload_len,
    };

    const double t0 = warm_now_sec();
    if (write_exact(c, &h, sizeof(h)) != 0) return -1;
    if (payload_len != 0 && write_exact(c, payload, payload_len) != 0) return -1;

    ds4_warm_frame_header rh;
    if (read_exact(c, &rh, sizeof(rh)) != 0) return -1;
    if (rh.magic != DS4_WARM_MAGIC ||
        rh.version != DS4_WARM_PROTOCOL_VERSION ||
        rh.opcode != opcode ||
        rh.sequence != seq) {
        warm_set_error(c, "bad response header");
        return -1;
    }
    if (rh.payload_len > UINT32_C(512) * 1024u * 1024u) {
        warm_set_error(c, "response is too large");
        return -1;
    }
    void *p = NULL;
    if (rh.payload_len != 0) {
        p = warm_malloc(rh.payload_len);
        if (!p) {
            warm_set_error(c, "out of memory");
            return -1;
        }
        if (read_exact(c, p, rh.payload_len) != 0) {
            free(p);
            return -1;
        }
    }
    if (timing) {
        timing->request_bytes = sizeof(h) + payload_len;
        timing->response_bytes = sizeof(rh) + rh.payload_len;
        timing->elapsed_seconds = warm_now_sec() - t0;
    }
    *out_payload = p;
    *out_payload_len = rh.payload_len;
    return 0;
}

int ds4_warm_client_hello(
        ds4_warm_client      *c,
        ds4_warm_model_info  *out) {
    void *payload = NULL;
    uint32_t payload_len = 0;
    if (transact(c, DS4_WARM_OP_HELLO, NULL, 0, &payload, &payload_len, NULL) != 0) return -1;
    if (payload_len != sizeof(ds4_warm_hello_response)) {
        free(payload);
        warm_set_error(c, "bad HELLO response length");
        return -1;
    }
    ds4_warm_hello_response resp;
    memcpy(&resp, payload, sizeof(resp));
    free(payload);
    if (resp.status != DS4_WARM_STATUS_OK || resp.protocol_version != DS4_WARM_PROTOCOL_VERSION) {
        warm_set_error(c, "HELLO rejected");
        return -1;
    }
    if (out) *out = resp.model;
    return 0;
}

static int expert_list(
        ds4_warm_client          *c,
        uint16_t                  opcode,
        const ds4_warm_expert_id *ids,
        uint32_t                  count,
        ds4_warm_expert_list_response *out) {
    if (count != 0 && !ids) return -1;
    const uint64_t payload_len_u64 = sizeof(ds4_warm_expert_list_header) +
                                     (uint64_t)count * sizeof(ds4_warm_expert_id);
    if (payload_len_u64 > UINT32_MAX) {
        warm_set_error(c, "expert list too large");
        return -1;
    }
    uint8_t *payload = warm_malloc((size_t)payload_len_u64);
    if (!payload) {
        warm_set_error(c, "out of memory");
        return -1;
    }
    ds4_warm_expert_list_header hdr = { .count = count };
    memcpy(payload, &hdr, sizeof(hdr));
    if (count != 0) memcpy(payload + sizeof(hdr), ids, (size_t)count * sizeof(ids[0]));

    void *resp_payload = NULL;
    uint32_t resp_len = 0;
    const int rc = transact(c, opcode, payload, (uint32_t)payload_len_u64,
                            &resp_payload, &resp_len, NULL);
    free(payload);
    if (rc != 0) return -1;
    if (resp_len != sizeof(ds4_warm_expert_list_response)) {
        free(resp_payload);
        warm_set_error(c, "bad expert-list response length");
        return -1;
    }
    ds4_warm_expert_list_response resp;
    memcpy(&resp, resp_payload, sizeof(resp));
    free(resp_payload);
    if (out) *out = resp;
    return resp.status == DS4_WARM_STATUS_OK ? 0 : -1;
}

int ds4_warm_client_load_experts(
        ds4_warm_client          *c,
        const ds4_warm_expert_id *ids,
        uint32_t                  count,
        ds4_warm_expert_list_response *out) {
    return expert_list(c, DS4_WARM_OP_LOAD_EXPERTS, ids, count, out);
}

int ds4_warm_client_evict_experts(
        ds4_warm_client          *c,
        const ds4_warm_expert_id *ids,
        uint32_t                  count,
        ds4_warm_expert_list_response *out) {
    return expert_list(c, DS4_WARM_OP_EVICT_EXPERTS, ids, count, out);
}

int ds4_warm_client_run_routed_experts_q8_bf16(
        ds4_warm_client       *c,
        uint32_t               layer,
        const void            *xq,
        uint32_t               n_tok,
        const int32_t         *selected,
        const float           *weights,
        uint32_t               n_selected,
        float                 *out,
        uint32_t               out_dim,
        ds4_warm_client_timing *timing) {
    if (!c || !xq || !selected || !weights || !out || n_tok == 0) return -1;
    const uint64_t slots = (uint64_t)n_tok * n_selected;
    const uint64_t selected_bytes = slots * sizeof(int32_t);
    const uint64_t weight_bytes = slots * sizeof(float);
    uint64_t x_bytes = 0;
    if (ds4_warm_q8_k_bytes(n_tok, out_dim, &x_bytes) != 0) {
        warm_set_error(c, "bad Q8_K input shape");
        return -1;
    }
    const uint64_t payload_len_u64 = sizeof(ds4_warm_run_routed_experts_header) +
                                     selected_bytes + weight_bytes + x_bytes;
    if (payload_len_u64 > UINT32_MAX) {
        warm_set_error(c, "RUN_ROUTED_EXPERTS payload too large");
        return -1;
    }

    uint8_t *payload = warm_malloc((size_t)payload_len_u64);
    if (!payload) {
        warm_set_error(c, "out of memory");
        return -1;
    }
    ds4_warm_run_routed_experts_header req = {
        .layer = layer,
        .n_tok = n_tok,
        .n_selected = n_selected,
        .input_format = DS4_WARM_INPUT_Q8_K,
        .output_format = DS4_WARM_OUTPUT_BF16,
    };
    uint8_t *p = payload;
    memcpy(p, &req, sizeof(req)); p += sizeof(req);
    memcpy(p, selected, (size_t)selected_bytes); p += selected_bytes;
    memcpy(p, weights, (size_t)weight_bytes); p += weight_bytes;
    memcpy(p, xq, (size_t)x_bytes);

    void *resp_payload = NULL;
    uint32_t resp_len = 0;
    ds4_warm_client_timing local_timing;
    const int rc = transact(c, DS4_WARM_OP_RUN_ROUTED_EXPERTS, payload, (uint32_t)payload_len_u64,
                            &resp_payload, &resp_len, &local_timing);
    free(payload);
    if (rc != 0) return -1;
    if (resp_len < sizeof(ds4_warm_run_routed_experts_response)) {
        free(resp_payload);
        warm_set_error(c, "bad RUN_ROUTED_EXPERTS response length");
        return -1;
    }
    ds4_warm_run_routed_experts_response resp;
    memcpy(&resp, resp_payload, sizeof(resp));
    if (resp.status != DS4_WARM_STATUS_OK) {
        free(resp_payload);
        snprintf(c->error, sizeof(c->error), "RUN_ROUTED_EXPERTS status=%u", resp.status);
        return -1;
    }
    const uint64_t out_elems = (uint64_t)resp.n_tok * resp.output_dim;
    const uint64_t out_bytes = out_elems * sizeof(uint16_t);
    if (resp.n_tok != n_tok || resp.output_dim != out_dim ||
        out_elems > UINT64_MAX / sizeof(float) ||
        sizeof(resp) + out_bytes != resp_len) {
        free(resp_payload);
        warm_set_error(c, "RUN_ROUTED_EXPERTS response shape mismatch");
        return -1;
    }
    const uint8_t *out_payload = (const uint8_t *)resp_payload + sizeof(resp);
    const uint16_t *bf16 = (const uint16_t *)out_payload;
    for (uint64_t i = 0; i < out_elems; i++) out[i] = warm_bf16_to_f32(bf16[i]);
    free(resp_payload);
    if (timing) *timing = local_timing;
    return 0;
}

int ds4_warm_client_stats(
        ds4_warm_client          *c,
        ds4_warm_stats_response  *out) {
    void *payload = NULL;
    uint32_t payload_len = 0;
    if (transact(c, DS4_WARM_OP_STATS, NULL, 0, &payload, &payload_len, NULL) != 0) return -1;
    if (payload_len != sizeof(ds4_warm_stats_response)) {
        free(payload);
        warm_set_error(c, "bad STATS response length");
        return -1;
    }
    if (out) memcpy(out, payload, sizeof(*out));
    free(payload);
    return 0;
}
