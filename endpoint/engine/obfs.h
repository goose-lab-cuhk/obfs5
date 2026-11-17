/*
 * obfs.h — the C interface of the obfsgen engine: one recurrent step of the
 * sequence model and the sampling decision, for programs without Python
 * (engine.go is the Go binding).
 *
 * obfs_step.c implements the step and the decision, obfs_load.c reads a
 * weight file written by model/export.py.
 *
 * Shapes (all row-major, float32 unless stated; B = streams in one call):
 *   tokens  int64 (B, 2)  = (size bin, time bin) of the token being fed
 *   pos     int64 (B)     = its position in its stream (0, 1, 2, ...)
 *   roles   int64 (B)     = 0 client / 1 server (async model), NULL for the
 *                           sync model (n_roles == 1)
 *   conv    (n_layers, B, d_inner, d_conv)   the streams' conv states, zeroed
 *   ssm     (n_layers, B, d_inner, d_state)  the streams' SSM states, zeroed
 *           at the start of a stream; obfs_step updates both in place
 *   h_out   (B, d_model)  the hidden vector after the token
 *   size_logits (B, size_out) the next token's size logits (size_out = B_s + 1:
 *           the last slot is [EOF])
 */
#ifndef OBFS_H
#define OBFS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OBFS_ABI 3

#define OBFS_W_FP32 0
#define OBFS_W_INT8 2

typedef struct {
    int32_t d_model, d_inner, d_conv, d_state, dt_rank, n_layers;
    int32_t size_out, time_out, d_cond, n_roles, max_seq_len, x_out;
    int32_t emb_Bs, emb_Bt, abi, wtype, B_s, B_t;
    const float *emb_w, *emb_b, *pe, *role_emb;
    const float *norm_w, *norm_b;
    const void *in_proj_w;
    const float *conv_w, *conv_b;
    const void *x_proj_w;
    const void *dt_proj_w;
    const float *dt_proj_b;
    const float *A, *D;
    const void *out_proj_w;
    const float *fnorm_w, *fnorm_b;
    const void *size_w;
    const float *size_b;
    const void *t1_wh;
    const float *t1_tab;
    const void *t2_w;
    const float *t2_b;
    /* one scale per row of the int8 matrices; NULL for fp32 */
    const float *in_proj_s, *x_proj_s, *dt_proj_s, *out_proj_s;
    const float *size_s, *t1_s, *t2_s;
} obfs_model;

/* ---- obfs_step.c ------------------------------------------------------- */

int obfs_abi(void);  /* OBFS_ABI */
int obfs_simd(void); /* the kernels built: 0 plain C, 1 AVX2, 2 AVX-512, 3 NEON */

/* One recurrent step for B streams at once (one new token each). */
void obfs_step(const obfs_model *m, int B, const int64_t *tokens,
               const int64_t *pos, const int64_t *roles, float *conv,
               float *ssm, float *h_out, float *size_logits);

/* The time head: N rows of time logits (time_out each) for the size bins
 * `bins`; shared = 1 uses the single row h for every bin. */
void obfs_time_logits(const obfs_model *m, int N, const float *h, int shared,
                      const int64_t *bins, float *out);

/* The decisions for one stream at (h, size_logits). q holds i.i.d. Exp(1)
 * draws, one per bin: the bin is argmax(p / q), a draw from the softmax p, so
 * the caller owns the randomness. Masks are byte arrays, 1 = allowed. */

/* The next size bin (B_s = [EOF]). smask (size_out) restricts the size bins.
 * Without smask, tmask (B_t) conditions the draw on a time bin in tmask: each
 * regular size bin is weighted by P(time in tmask | size). Both may be NULL. */
int obfs_decide_size(const obfs_model *m, const float *h, const float *size_logits,
                     const uint8_t *smask, const uint8_t *tmask, const float *q);

/* The time bin (0..B_t-1) for a regular size bin; tmask (B_t) or NULL. */
int obfs_decide_time(const obfs_model *m, const float *h, int size_bin,
                     const uint8_t *tmask, const float *q);

/* The distribution of the gap before the next received record (B_t doubles
 * into pmf): the size softmax restricted to the receive bins recv (B_s, 1 =
 * receive), each one's time softmax, mixed, the bins below min_time_bin cut,
 * renormalised. Returns the receive bins' probability mass. */
float obfs_receive_gap_pmf(const obfs_model *m, const float *h, const float *size_logits,
                           int min_time_bin, const uint8_t *recv, double *pmf);

/* ---- obfs_load.c ------------------------------------------------------- */

/* Load a weight file written by model/export.py (format OBFSW001).
 * Returns NULL and an error message in err (errlen bytes) on failure. The
 * model and every buffer it points to live in one allocation; free it with
 * obfs_free. The model is read-only and can be shared by any number of
 * threads, each with its own states. */
obfs_model *obfs_load(const char *path, char *err, int errlen);
/* The same from a buffer already in memory (copied; the caller keeps data). */
obfs_model *obfs_load_mem(const void *data, size_t size, char *err, int errlen);
void obfs_free(obfs_model *m);

/* Floats of ONE stream's conv / ssm state (n_layers · d_inner · d_conv and
 * n_layers · d_inner · d_state). A call with B streams takes B times that,
 * laid out (n_layers, B, d_inner, ·): the stream index is the SECOND axis, so
 * streams that are batched together must share one state block (or be copied
 * in and out of it); with B = 1 each stream simply owns its two buffers. */
long obfs_conv_state_floats(const obfs_model *m);
long obfs_ssm_state_floats(const obfs_model *m);

#ifdef __cplusplus
}
#endif

#endif /* OBFS_H */
