#ifndef DS4_WARM_H
#define DS4_WARM_H

#include <stdint.h>

#include "ds4.h"

#define DS4_WARM_MAGIC 0x57345344u /* DS4W, little-endian on the wire. */
#define DS4_WARM_PROTOCOL_VERSION 1u

#define DS4_WARM_INPUT_F32 1u
#define DS4_WARM_OUTPUT_F32 1u

typedef enum {
    DS4_WARM_OP_HELLO = 1,
    DS4_WARM_OP_LOAD_EXPERTS = 2,
    DS4_WARM_OP_EVICT_EXPERTS = 3,
    DS4_WARM_OP_RUN_LAYER = 4,
    DS4_WARM_OP_STATS = 5,
} ds4_warm_opcode;

typedef enum {
    DS4_WARM_STATUS_OK = 0,
    DS4_WARM_STATUS_BAD_REQUEST = 1,
    DS4_WARM_STATUS_UNSUPPORTED = 2,
    DS4_WARM_STATUS_NOT_RESIDENT = 3,
    DS4_WARM_STATUS_RUNTIME_ERROR = 4,
} ds4_warm_status;

typedef struct {
    uint32_t magic;
    uint16_t version;
    uint16_t opcode;
    uint32_t sequence;
    uint32_t payload_len;
} ds4_warm_frame_header;

typedef struct {
    uint32_t layer;
    uint32_t expert;
} ds4_warm_expert_id;

typedef struct {
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t n_ff_exp;
    uint32_t gate_type;
    uint32_t up_type;
    uint32_t down_type;
    uint64_t gate_expert_bytes;
    uint64_t up_expert_bytes;
    uint64_t down_expert_bytes;
    uint64_t model_size;
    uint64_t model_fingerprint;
} ds4_warm_model_info;

typedef struct {
    uint32_t status;
    uint32_t protocol_version;
    ds4_warm_model_info model;
} ds4_warm_hello_response;

typedef struct {
    uint32_t count;
} ds4_warm_expert_list_header;

typedef struct {
    uint32_t status;
    uint32_t accepted;
    uint32_t resident;
    uint32_t reserved;
} ds4_warm_expert_list_response;

typedef struct {
    uint32_t layer;
    uint32_t n_tok;
    uint32_t n_selected;
    uint32_t input_format;
    uint32_t output_format;
    uint32_t flags;
} ds4_warm_run_layer_header;

typedef struct {
    uint32_t status;
    uint32_t n_tok;
    uint32_t output_dim;
    uint32_t reserved;
} ds4_warm_run_layer_response;

typedef struct {
    uint64_t requests;
    uint64_t run_layer_requests;
    uint64_t tokens;
    uint64_t selected_slots;
    uint64_t resident_hits;
    uint64_t resident_misses;
    double compute_seconds;
} ds4_warm_stats_response;

int ds4_engine_warm_model_info(ds4_engine *e, ds4_warm_model_info *out);
int ds4_engine_warm_run_layer_f32(
        ds4_engine    *e,
        uint32_t       layer,
        const float   *x,
        uint32_t       n_tok,
        const int32_t *selected,
        const float   *weights,
        uint32_t       n_selected,
        float         *out);
int ds4_engine_warm_run_layer_metal_f32(
        ds4_engine    *e,
        uint32_t       layer,
        const float   *x,
        uint32_t       n_tok,
        const int32_t *selected,
        const float   *weights,
        uint32_t       n_selected,
        float         *out);

#endif
