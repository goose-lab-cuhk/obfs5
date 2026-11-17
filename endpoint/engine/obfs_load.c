/*
 * obfs_load.c — read a weight file written by model/export.py into an
 * obfs_model, so the C step runs without Python or torch.
 *
 * File format OBFSW001 (little-endian):
 *   char     magic[8]   = "OBFSW001"
 *   uint32   version    = 1
 *   uint32   n_ints     = 18
 *   int32    ints[18]   the struct's integer fields in declaration order
 *                       (d_model ... B_t; abi must equal OBFS_ABI)
 *   uint32   n_bufs
 *   n_bufs × { char name[32]; uint64 offset; uint64 nbytes; }
 *                       name = the struct's pointer field (NUL-padded),
 *                       offset from the start of the file, a multiple of 64
 *   the buffers' raw bytes at their offsets (float32, except the matrices of
 *   an int8 file: int8 rows)
 * The whole file is read into one 64-byte aligned allocation and the struct's
 * pointers point into it.
 */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "obfs.h"

#define OBFW_MAGIC "OBFSW001"
#define OBFW_VERSION 1u
#define OBFW_NAME 32
#define OBFW_ALIGN 64

typedef struct {
    obfs_model m; /* first member: the pointer handed to the caller */
    void *blob;   /* the file's bytes; every pointer of m points into it */
} obfw_holder;

static const size_t INT_OFFSETS[18] = {
    offsetof(obfs_model, d_model),  offsetof(obfs_model, d_inner),
    offsetof(obfs_model, d_conv),   offsetof(obfs_model, d_state),
    offsetof(obfs_model, dt_rank),  offsetof(obfs_model, n_layers),
    offsetof(obfs_model, size_out), offsetof(obfs_model, time_out),
    offsetof(obfs_model, d_cond),   offsetof(obfs_model, n_roles),
    offsetof(obfs_model, max_seq_len), offsetof(obfs_model, x_out),
    offsetof(obfs_model, emb_Bs),   offsetof(obfs_model, emb_Bt),
    offsetof(obfs_model, abi),      offsetof(obfs_model, wtype),
    offsetof(obfs_model, B_s),      offsetof(obfs_model, B_t),
};

/* required: 1 always, 2 only in an int8 file (the row scales) */
static const struct {
    const char *name;
    size_t off;
    int required;
} FIELDS[] = {
    {"emb_w", offsetof(obfs_model, emb_w), 1},
    {"emb_b", offsetof(obfs_model, emb_b), 1},
    {"pe", offsetof(obfs_model, pe), 1},
    {"role_emb", offsetof(obfs_model, role_emb), 1},
    {"norm_w", offsetof(obfs_model, norm_w), 1},
    {"norm_b", offsetof(obfs_model, norm_b), 1},
    {"in_proj_w", offsetof(obfs_model, in_proj_w), 1},
    {"conv_w", offsetof(obfs_model, conv_w), 1},
    {"conv_b", offsetof(obfs_model, conv_b), 1},
    {"x_proj_w", offsetof(obfs_model, x_proj_w), 1},
    {"dt_proj_w", offsetof(obfs_model, dt_proj_w), 1},
    {"dt_proj_b", offsetof(obfs_model, dt_proj_b), 1},
    {"A", offsetof(obfs_model, A), 1},
    {"D", offsetof(obfs_model, D), 1},
    {"out_proj_w", offsetof(obfs_model, out_proj_w), 1},
    {"fnorm_w", offsetof(obfs_model, fnorm_w), 1},
    {"fnorm_b", offsetof(obfs_model, fnorm_b), 1},
    {"size_w", offsetof(obfs_model, size_w), 1},
    {"size_b", offsetof(obfs_model, size_b), 1},
    {"t1_wh", offsetof(obfs_model, t1_wh), 1},
    {"t1_tab", offsetof(obfs_model, t1_tab), 1},
    {"t2_w", offsetof(obfs_model, t2_w), 1},
    {"t2_b", offsetof(obfs_model, t2_b), 1},
    {"in_proj_s", offsetof(obfs_model, in_proj_s), 2},
    {"x_proj_s", offsetof(obfs_model, x_proj_s), 2},
    {"dt_proj_s", offsetof(obfs_model, dt_proj_s), 2},
    {"out_proj_s", offsetof(obfs_model, out_proj_s), 2},
    {"size_s", offsetof(obfs_model, size_s), 2},
    {"t1_s", offsetof(obfs_model, t1_s), 2},
    {"t2_s", offsetof(obfs_model, t2_s), 2},
};
#define N_FIELDS ((int)(sizeof(FIELDS) / sizeof(FIELDS[0])))

static uint32_t rd_u32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static uint64_t rd_u64(const unsigned char *p) {
    return (uint64_t)rd_u32(p) | (uint64_t)rd_u32(p + 4) << 32;
}

static obfs_model *fail(char *err, int errlen, obfw_holder *h, const char *msg,
                        const char *arg) {
    if (err && errlen > 0) snprintf(err, (size_t)errlen, "%s%s", msg, arg ? arg : "");
    if (h) {
        free(h->blob);
        free(h);
    }
    return NULL;
}

obfs_model *obfs_load_mem(const void *data, size_t size, char *err, int errlen) {
    obfw_holder *h = calloc(1, sizeof *h);
    if (!h) return fail(err, errlen, NULL, "out of memory", NULL);
    const size_t cap = (size + OBFW_ALIGN - 1) / OBFW_ALIGN * OBFW_ALIGN;
    /* posix_memalign, not aligned_alloc: Android's libc has the latter only
     * from API 28 */
    if (posix_memalign(&h->blob, OBFW_ALIGN, cap > 0 ? cap : OBFW_ALIGN) != 0) {
        h->blob = NULL;
        return fail(err, errlen, h, "out of memory", NULL);
    }
    memcpy(h->blob, data, size);
    const char *path = "(memory)";
    const unsigned char *b = (const unsigned char *)h->blob;
    const size_t head = 8 + 4 + 4 + 18 * 4 + 4;
    if ((size_t)size < head || memcmp(b, OBFW_MAGIC, 8) != 0)
        return fail(err, errlen, h, "not an OBFSW001 weight file: ", path);
    if (rd_u32(b + 8) != OBFW_VERSION) return fail(err, errlen, h, "unsupported version", NULL);
    if (rd_u32(b + 12) != 18u) return fail(err, errlen, h, "unexpected integer count", NULL);
    for (int i = 0; i < 18; i++) {
        const int32_t v = (int32_t)rd_u32(b + 16 + 4 * i);
        memcpy((char *)&h->m + INT_OFFSETS[i], &v, sizeof v);
    }
    if (h->m.abi != OBFS_ABI) return fail(err, errlen, h, "ABI mismatch with obfs_step.c", NULL);
    if (h->m.wtype != OBFS_W_FP32 && h->m.wtype != OBFS_W_INT8)
        return fail(err, errlen, h, "unknown weight format", NULL);
    const uint32_t n_bufs = rd_u32(b + 16 + 18 * 4);
    const size_t entry = OBFW_NAME + 16;
    if (n_bufs > 64 || head + (size_t)n_bufs * entry > (size_t)size)
        return fail(err, errlen, h, "corrupt buffer table", NULL);
    int seen[64] = {0};
    for (uint32_t k = 0; k < n_bufs; k++) {
        const unsigned char *e = b + head + (size_t)k * entry;
        char name[OBFW_NAME + 1];
        memcpy(name, e, OBFW_NAME);
        name[OBFW_NAME] = '\0';
        const uint64_t off = rd_u64(e + OBFW_NAME), nbytes = rd_u64(e + OBFW_NAME + 8);
        if (off % OBFW_ALIGN != 0 || off > (uint64_t)size || nbytes > (uint64_t)size - off)
            return fail(err, errlen, h, "buffer out of range: ", name);
        int j = 0;
        while (j < N_FIELDS && strcmp(FIELDS[j].name, name) != 0) j++;
        if (j == N_FIELDS) return fail(err, errlen, h, "unknown buffer: ", name);
        if (seen[j]) return fail(err, errlen, h, "duplicate buffer: ", name);
        seen[j] = 1;
        const void *ptr = b + off;
        memcpy((char *)&h->m + FIELDS[j].off, &ptr, sizeof ptr);
    }
    const int quantized = h->m.wtype == OBFS_W_INT8;
    for (int j = 0; j < N_FIELDS; j++) {
        if (FIELDS[j].required == 1 && !seen[j])
            return fail(err, errlen, h, "missing buffer: ", FIELDS[j].name);
        if (FIELDS[j].required == 2 && seen[j] != quantized)
            return fail(err, errlen, h,
                        quantized ? "missing row scale: " : "row scale in an fp32 file: ",
                        FIELDS[j].name);
    }
    return &h->m;
}

obfs_model *obfs_load(const char *path, char *err, int errlen) {
    FILE *f = fopen(path, "rb");
    if (!f) return fail(err, errlen, NULL, "cannot open ", path);
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return fail(err, errlen, NULL, "cannot seek ", path);
    }
    const long size = ftell(f);
    rewind(f);
    void *buf = malloc(size > 0 ? (size_t)size : 1);
    if (!buf) {
        fclose(f);
        return fail(err, errlen, NULL, "out of memory", NULL);
    }
    const size_t got = fread(buf, 1, (size_t)size, f);
    fclose(f);
    if (got != (size_t)size) {
        free(buf);
        return fail(err, errlen, NULL, "short read ", path);
    }
    obfs_model *m = obfs_load_mem(buf, (size_t)size, err, errlen);
    free(buf);
    return m;
}

void obfs_free(obfs_model *m) {
    if (!m) return;
    obfw_holder *h = (obfw_holder *)m; /* m is the holder's first member */
    free(h->blob);
    free(h);
}

long obfs_conv_state_floats(const obfs_model *m) {
    return (long)m->n_layers * m->d_inner * m->d_conv;
}

long obfs_ssm_state_floats(const obfs_model *m) {
    return (long)m->n_layers * m->d_inner * m->d_state;
}
