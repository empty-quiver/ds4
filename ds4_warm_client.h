#ifndef DS4_WARM_CLIENT_H
#define DS4_WARM_CLIENT_H

#include <stdint.h>

#include "ds4_warm.h"

typedef struct ds4_warm_client ds4_warm_client;

typedef struct {
    uint64_t request_bytes;
    uint64_t response_bytes;
    double elapsed_seconds;
} ds4_warm_client_timing;

int ds4_warm_client_connect(
        ds4_warm_client **out,
        const char       *host,
        const char       *port,
        int               timeout_ms);

void ds4_warm_client_close(ds4_warm_client *c);

int ds4_warm_client_hello(
        ds4_warm_client      *c,
        ds4_warm_model_info  *out);

int ds4_warm_client_load_experts(
        ds4_warm_client          *c,
        const ds4_warm_expert_id *ids,
        uint32_t                  count,
        ds4_warm_expert_list_response *out);

int ds4_warm_client_evict_experts(
        ds4_warm_client          *c,
        const ds4_warm_expert_id *ids,
        uint32_t                  count,
        ds4_warm_expert_list_response *out);

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
        ds4_warm_client_timing *timing);

int ds4_warm_client_stats(
        ds4_warm_client          *c,
        ds4_warm_stats_response  *out);

const char *ds4_warm_client_error(ds4_warm_client *c);

#endif
