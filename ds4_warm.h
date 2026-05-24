#ifndef DS4_WARM_H
#define DS4_WARM_H

#include <stdint.h>

#include "ds4.h"

#define DS4_WARM_MAGIC 0x57345344u /* DS4W, little-endian on the wire. */
#define DS4_WARM_PROTOCOL_VERSION 2u

#define DS4_WARM_INPUT_Q8_K 1u
#define DS4_WARM_OUTPUT_BF16 1u

#define DS4_WARM_Q8_K_BLOCK_ELEMS 256u
#define DS4_WARM_Q8_K_BLOCK_BYTES 292u

typedef enum {
    DS4_WARM_OP_HELLO = 1,
    DS4_WARM_OP_LOAD_EXPERTS = 2,
    DS4_WARM_OP_EVICT_EXPERTS = 3,
    DS4_WARM_OP_RUN_ROUTED_EXPERTS = 4,
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

/* Run only the selected routed MoE experts for one transformer layer.
 *
 * This request does not execute the whole layer: attention, layer norms,
 * router computation, shared experts, residuals, KV/cache work, and logits
 * remain owned by the Vector runtime. The worker receives activations plus
 * selected expert IDs/route weights, computes the routed expert contribution,
 * and returns an n_tok x n_embd BF16 tensor to be summed by the caller.
 */
typedef struct {
    uint32_t layer;
    uint32_t n_tok;
    uint32_t n_selected;
    uint32_t input_format;
    uint32_t output_format;
    uint32_t flags;
} ds4_warm_run_routed_experts_header;

typedef struct {
    uint32_t status;
    uint32_t n_tok;
    uint32_t output_dim;
    uint32_t reserved;
} ds4_warm_run_routed_experts_response;

typedef struct {
    uint64_t requests;
    uint64_t run_routed_expert_requests;
    uint64_t tokens;
    uint64_t selected_slots;
    uint64_t resident_hits;
    uint64_t resident_misses;
    double compute_seconds;
} ds4_warm_stats_response;

int ds4_engine_warm_model_info(ds4_engine *e, ds4_warm_model_info *out);
int ds4_engine_warm_run_routed_experts_q8_f32(
        ds4_engine    *e,
        uint32_t       layer,
        const void    *xq,
        uint32_t       n_tok,
        const int32_t *selected,
        const float   *weights,
        uint32_t       n_selected,
        float         *out);
int ds4_engine_warm_run_routed_experts_metal_q8_bf16(
        ds4_engine    *e,
        uint32_t       layer,
        const void    *xq,
        uint32_t       n_tok,
        const int32_t *selected,
        const float   *weights,
        uint32_t       n_selected,
        uint16_t      *out_bf16);
int ds4_engine_warm_load_experts_metal(
        ds4_engine                *e,
        const ds4_warm_expert_id  *ids,
        uint32_t                   count,
        uint32_t                  *accepted,
        uint32_t                  *resident);
int ds4_engine_warm_evict_experts_metal(
        ds4_engine                *e,
        const ds4_warm_expert_id  *ids,
        uint32_t                   count,
        uint32_t                  *accepted,
        uint32_t                  *resident);

int ds4_warm_q8_k_bytes(uint32_t n_rows, uint32_t row_dim, uint64_t *out_bytes);
int ds4_warm_quantize_f32_to_q8_k(const float *x, void *out_q8, uint32_t n_rows, uint32_t row_dim);

#endif
