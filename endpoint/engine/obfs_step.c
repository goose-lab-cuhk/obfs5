/*
 * obfs_step.c — one recurrent step of the obfsgen model (embedding, Mamba
 * blocks, final norm, size head), the time head, and the sampling decision.
 * Plain C with optional AVX2 / AVX-512 / NEON paths (-DOBFS_NO_SIMD for none).
 *
 * Matrices are fp32 or int8 rows (wtype); an int8 row carries one float
 * scale, and a product is (sum q * x) * scale. Everything else is fp32. The
 * time head's first layer is split into its h part (t1_wh) and a per-size-bin
 * table t1_tab = W1[:, d_model:] e(s) + b1, made at export.
 * States are laid out (n_layers, B, d_inner, d_conv | d_state).
 */

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "obfs.h"

#if !defined(OBFS_NO_SIMD) && defined(__AVX512F__)
#define OBFS_SIMD 2
#include <immintrin.h>
#elif !defined(OBFS_NO_SIMD) && defined(__AVX2__) && defined(__FMA__)
#define OBFS_SIMD 1
#include <immintrin.h>
#elif !defined(OBFS_NO_SIMD) && defined(__aarch64__) && defined(__ARM_NEON)
#define OBFS_SIMD 3 /* ARM64 NEON (Apple silicon, Android arm64-v8a) */
#include <arm_neon.h>
#else
#define OBFS_SIMD 0
#endif

typedef float v8f __attribute__((vector_size(32)));
typedef int32_t v8i __attribute__((vector_size(32)));

int obfs_abi(void) { return OBFS_ABI; }
int obfs_simd(void) { return OBFS_SIMD; }

/* ------------------------------------------------------------------------ */
/* eight-lane helpers                                                        */
/* ------------------------------------------------------------------------ */

static inline v8f ld8(const float *p) {
    v8f v;
    memcpy(&v, p, sizeof v);
    return v;
}

static inline void st8(float *p, v8f v) { memcpy(p, &v, sizeof v); }

static inline v8f splat8(float x) {
    v8f v = {x, x, x, x, x, x, x, x};
    return v;
}

static inline float hsum8(v8f v) {
    float t[8];
    memcpy(t, &v, sizeof t);
    return ((t[0] + t[4]) + (t[1] + t[5])) + ((t[2] + t[6]) + (t[3] + t[7]));
}

/* select(mask, a, b): a where the lane mask is set, b elsewhere */
static inline v8f sel8(v8i m, v8f a, v8f b) {
    return (v8f)(((v8i)a & m) | ((v8i)b & ~m));
}

/* exp, Cephes expf: clamp, range-reduce by ln 2, degree-6 polynomial, scale */
static inline v8f exp8(v8f x) {
    const v8f hi = splat8(88.0f), lo = splat8(-87.0f);
    x = sel8(x > hi, hi, x);
    x = sel8(x < lo, lo, x);
    const v8f magic = splat8(12582912.0f); /* 1.5 * 2^23: round to nearest */
    v8f fx = (x * splat8(1.44269504088896341f) + magic) - magic;
    v8f r = x - fx * splat8(0.693359375f);
    r = r - fx * splat8(-2.12194440e-4f);
    v8f p = splat8(1.9875691500E-4f);
    p = p * r + splat8(1.3981999507E-3f);
    p = p * r + splat8(8.3334519073E-3f);
    p = p * r + splat8(4.1665795894E-2f);
    p = p * r + splat8(1.6666665459E-1f);
    p = p * r + splat8(5.0000001201E-1f);
    p = p * (r * r) + r + splat8(1.0f);
    v8i n = __builtin_convertvector(fx, v8i);
    v8f pow2 = (v8f)((n + 127) << 23);
    return p * pow2;
}

/* log, Cephes logf (x > 0): frexp by the bits, degree-9 polynomial */
static inline v8f log8(v8f x) {
    v8i bits = (v8i)x;
    v8i e = ((bits >> 23) & 0xff) - 126;
    v8f m = (v8f)((bits & 0x807fffff) | 0x3f000000); /* in [0.5, 1) */
    v8i lt = m < splat8(0.707106781186547524f);
    e = e + lt; /* lt is -1 where true */
    v8f y = sel8(lt, m + m - splat8(1.0f), m - splat8(1.0f));
    v8f z = y * y;
    v8f p = splat8(7.0376836292E-2f);
    p = p * y + splat8(-1.1514610310E-1f);
    p = p * y + splat8(1.1676998740E-1f);
    p = p * y + splat8(-1.2420140846E-1f);
    p = p * y + splat8(1.4249322787E-1f);
    p = p * y + splat8(-1.6668057665E-1f);
    p = p * y + splat8(2.0000714765E-1f);
    p = p * y + splat8(-2.4999993993E-1f);
    p = p * y + splat8(3.3333331174E-1f);
    p = p * y * z;
    v8f fe = __builtin_convertvector(e, v8f);
    p = p + fe * splat8(-2.12194440e-4f);
    p = p - z * splat8(0.5f);
    return y + p + fe * splat8(0.693359375f);
}

static inline v8f silu8(v8f x) { return x / (splat8(1.0f) + exp8(-x)); }

/* softplus(x) = max(x, 0) + log(1 + exp(-|x|)) */
static inline v8f softplus8(v8f x) {
    v8f ax = (v8f)((v8i)x & 0x7fffffff);
    v8f pos = sel8(x > splat8(0.0f), x, splat8(0.0f));
    return pos + log8(splat8(1.0f) + exp8(-ax));
}

static inline float silu1(float x) { return x / (1.0f + expf(-x)); }

static inline float softplus1(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

/* ------------------------------------------------------------------------ */
/* the weight formats                                                        */
/* ------------------------------------------------------------------------ */

/* bytes of one row of n elements in each format */
static inline size_t row_bytes(int wtype, int n) {
    return wtype == OBFS_W_INT8 ? (size_t)n : (size_t)n * 4;
}

static inline const void *wrow(const void *W, int wtype, size_t row, int n) {
    return (const char *)W + row * row_bytes(wtype, n);
}

static inline const float *srow(const float *S, size_t row) {
    return S ? S + row : NULL;
}

#if OBFS_SIMD == 2
static inline __m512 cvt16_i8(const int8_t *p) {
    return _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)p)));
}
#endif

#if OBFS_SIMD == 1
static inline __m256 cvt8_i8(const int8_t *p) {
    return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)p)));
}
static inline float hsum256(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}
#endif

/* ------------------------------------------------------------------------ */
/* vector kernels                                                            */
/* ------------------------------------------------------------------------ */

/* The lanes of the product kernel, one definition per SIMD level: ``vec`` =
 * OBFS_LANES floats, ck_<format>(w, i) = the chunk of OBFS_LANES weights from
 * element i of a row (i a multiple of OBFS_LANES) converted to fp32 lanes,
 * el_<format>(w, i) = one weight as fp32 (the scalar tail). The plain-C level
 * is one float per chunk and leaves the vectorising to the compiler. */
static inline float el_f32(const void *w, int i) { return ((const float *)w)[i]; }
static inline float el_i8(const void *w, int i) { return (float)((const int8_t *)w)[i]; }

#if OBFS_SIMD == 2
#define OBFS_LANES 16
typedef __m512 vec;
#define VZERO() _mm512_setzero_ps()
#define VLOAD(p) _mm512_loadu_ps(p)
#define VFMA(w, x, c) _mm512_fmadd_ps(w, x, c)
#define VADD(a, b) _mm512_add_ps(a, b)
#define VSUM(c) _mm512_reduce_add_ps(c)
static inline vec ck_f32(const void *w, int i) { return _mm512_loadu_ps((const float *)w + i); }
static inline vec ck_i8(const void *w, int i) { return cvt16_i8((const int8_t *)w + i); }
#elif OBFS_SIMD == 1
#define OBFS_LANES 8
typedef __m256 vec;
#define VZERO() _mm256_setzero_ps()
#define VLOAD(p) _mm256_loadu_ps(p)
#define VFMA(w, x, c) _mm256_fmadd_ps(w, x, c)
#define VADD(a, b) _mm256_add_ps(a, b)
#define VSUM(c) hsum256(c)
static inline vec ck_f32(const void *w, int i) { return _mm256_loadu_ps((const float *)w + i); }
static inline vec ck_i8(const void *w, int i) { return cvt8_i8((const int8_t *)w + i); }
#elif OBFS_SIMD == 3
/* NEON: a chunk is 8 lanes held in two q registers, so a block keeps K·U ≤ 8
 * accumulators (16 registers) and U converted chunks in the 32 NEON registers. */
#define OBFS_LANES 8
typedef struct {
    float32x4_t a, b;
} vec;
static inline vec VZERO(void) {
    vec v = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    return v;
}
static inline vec VLOAD(const float *p) {
    vec v = {vld1q_f32(p), vld1q_f32(p + 4)};
    return v;
}
static inline vec VFMA(vec w, vec x, vec c) {
    vec v = {vfmaq_f32(c.a, w.a, x.a), vfmaq_f32(c.b, w.b, x.b)};
    return v;
}
static inline vec VADD(vec x, vec y) {
    vec v = {vaddq_f32(x.a, y.a), vaddq_f32(x.b, y.b)};
    return v;
}
static inline float VSUM(vec c) { return vaddvq_f32(vaddq_f32(c.a, c.b)); }
static inline vec ck_f32(const void *w, int i) { return VLOAD((const float *)w + i); }
static inline vec i16x8_to_vec(int16x8_t s) {
    vec v = {vcvtq_f32_s32(vmovl_s16(vget_low_s16(s))), vcvtq_f32_s32(vmovl_high_s16(s))};
    return v;
}
static inline vec ck_i8(const void *w, int i) {
    return i16x8_to_vec(vmovl_s8(vld1_s8((const int8_t *)w + i)));
}
#else
#define OBFS_LANES 1
typedef float vec;
#define VZERO() 0.0f
#define VLOAD(p) (*(p))
#define VFMA(w, x, c) ((w) * (x) + (c))
#define VADD(a, b) ((a) + (b))
#define VSUM(c) (c)
#define ck_f32 el_f32
#define ck_i8 el_i8
#endif

/* The partial sums s[0..K) of K streams against one weight row, the streams'
 * rows sx floats apart: U chunks per iteration, so K·U accumulators are in
 * flight and the latency of the fused multiply-add is hidden at every K; each
 * converted chunk serves the K streams and is never stored. One instance per
 * weight format and block size (K·U ≤ 8 accumulators + U chunks in registers;
 * the loops over them are unrolled and the block is forced inline into the row
 * loop so the compiler keeps them there — left to its heuristics, gcc 11 calls
 * the block per row, which costs 0.1 ms per step at B = 1). */
#define OBFS_MV_BLOCK(FMT, K, U)                                                     \
    static inline __attribute__((always_inline)) void mv_##FMT##_##K(                \
        const void *w, int n, const float *X, int sx, float *s) {                   \
        vec acc[K * U], wv[U];                                                       \
        _Pragma("GCC unroll 8") for (int a = 0; a < K * U; a++) acc[a] = VZERO();    \
        int i = 0;                                                                   \
        for (; i + U * OBFS_LANES <= n; i += U * OBFS_LANES) {                       \
            _Pragma("GCC unroll 4") for (int u = 0; u < U; u++)                      \
                wv[u] = ck_##FMT(w, i + u * OBFS_LANES);                             \
            _Pragma("GCC unroll 8") for (int b = 0; b < K; b++)                      \
                _Pragma("GCC unroll 4") for (int u = 0; u < U; u++)                  \
                    acc[b * U + u] = VFMA(                                           \
                        wv[u], VLOAD(X + (size_t)b * sx + i + u * OBFS_LANES),       \
                        acc[b * U + u]);                                             \
        }                                                                            \
        for (; i + OBFS_LANES <= n; i += OBFS_LANES) {                               \
            const vec wv0 = ck_##FMT(w, i);                                          \
            _Pragma("GCC unroll 8") for (int b = 0; b < K; b++)                      \
                acc[b * U] = VFMA(wv0, VLOAD(X + (size_t)b * sx + i), acc[b * U]);   \
        }                                                                            \
        _Pragma("GCC unroll 8") for (int b = 0; b < K; b++) {                        \
            vec t = acc[b * U];                                                      \
            _Pragma("GCC unroll 4") for (int u = 1; u < U; u++)                      \
                t = VADD(t, acc[b * U + u]);                                         \
            s[b] = VSUM(t);                                                          \
        }                                                                            \
        for (; i < n; i++) {                                                         \
            const float wi = el_##FMT(w, i);                                         \
            _Pragma("GCC unroll 8") for (int b = 0; b < K; b++)                      \
                s[b] += wi * X[(size_t)b * sx + i];                                  \
        }                                                                            \
    }

/* Y[b, r] = scale[r] · (W[r, :] · X[b, :]) (+ bias[r]) for the rows of W in
 * one format, rb bytes per row; the streams' rows of X and Y are sx / sy
 * floats apart. Each weight row is read once and its chunks are converted
 * once per block of up to 8 streams (blocks of 8, 4, 2, 1 cover any B), so a
 * step costs a + c·B with the same kernel at every batch size and B = 1 is
 * the single-stream kernel itself. */
#define OBFS_MV_FORMAT(FMT)                                                          \
    OBFS_MV_BLOCK(FMT, 1, 4)                                                         \
    OBFS_MV_BLOCK(FMT, 2, 2)                                                         \
    OBFS_MV_BLOCK(FMT, 4, 2)                                                         \
    OBFS_MV_BLOCK(FMT, 8, 1)                                                         \
    static void matvec_##FMT(const void *W, size_t rb, const float *scale,           \
                             const float *bias, int rows, int n, const float *X,     \
                             int B, int sx, float *Y, int sy) {                      \
        for (int r = 0; r < rows; r++) {                                             \
            const void *w = (const char *)W + (size_t)r * rb;                        \
            const float add = bias ? bias[r] : 0.0f;                                 \
            const float sc = scale ? scale[r] : 1.0f;                                \
            float s[8];                                                              \
            int b = 0;                                                               \
            for (; b + 8 <= B; b += 8) {                                             \
                mv_##FMT##_8(w, n, X + (size_t)b * sx, sx, s);                       \
                for (int k = 0; k < 8; k++) Y[(size_t)(b + k) * sy + r] = sc * s[k] + add; \
            }                                                                        \
            if (b + 4 <= B) {                                                        \
                mv_##FMT##_4(w, n, X + (size_t)b * sx, sx, s);                       \
                for (int k = 0; k < 4; k++) Y[(size_t)(b + k) * sy + r] = sc * s[k] + add; \
                b += 4;                                                              \
            }                                                                        \
            if (b + 2 <= B) {                                                        \
                mv_##FMT##_2(w, n, X + (size_t)b * sx, sx, s);                       \
                for (int k = 0; k < 2; k++) Y[(size_t)(b + k) * sy + r] = sc * s[k] + add; \
                b += 2;                                                              \
            }                                                                        \
            if (b < B) {                                                             \
                mv_##FMT##_1(w, n, X + (size_t)b * sx, sx, s);                       \
                Y[(size_t)b * sy + r] = sc * s[0] + add;                             \
            }                                                                        \
        }                                                                            \
    }
OBFS_MV_FORMAT(f32)
OBFS_MV_FORMAT(i8)

/* the product in the format wtype (one path for every B) */
static void matvec(int wtype, const void *W, const float *scale, const float *bias,
                   int rows, int n, const float *X, int B, int sx, float *Y, int sy) {
    const size_t rb = row_bytes(wtype, n);
    if (wtype == OBFS_W_INT8)
        matvec_i8(W, rb, scale, bias, rows, n, X, B, sx, Y, sy);
    else
        matvec_f32(W, rb, scale, bias, rows, n, X, B, sx, Y, sy);
}

static void layernorm(const float *x, float *y, const float *w,
                      const float *bias, int n) {
    v8f acc = splat8(0.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) acc += ld8(x + i);
    float s = hsum8(acc);
    for (; i < n; i++) s += x[i];
    const float mean = s / (float)n;
    v8f vm = splat8(mean), acc2 = splat8(0.0f);
    for (i = 0; i + 8 <= n; i += 8) {
        v8f d = ld8(x + i) - vm;
        acc2 += d * d;
    }
    float ss = hsum8(acc2);
    for (; i < n; i++) {
        float d = x[i] - mean;
        ss += d * d;
    }
    const float rstd = 1.0f / sqrtf(ss / (float)n + 1e-5f);
    v8f vr = splat8(rstd);
    for (i = 0; i + 8 <= n; i += 8)
        st8(y + i, (ld8(x + i) - vm) * vr * ld8(w + i) + ld8(bias + i));
    for (; i < n; i++) y[i] = (x[i] - mean) * rstd * w[i] + bias[i];
}

static void silu_vec(float *x, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) st8(x + i, silu8(ld8(x + i)));
    for (; i < n; i++) x[i] = silu1(x[i]);
}

static void softplus_vec(float *x, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) st8(x + i, softplus8(ld8(x + i)));
    for (; i < n; i++) x[i] = softplus1(x[i]);
}

/* ------------------------------------------------------------------------ */
/* the step                                                                   */
/* ------------------------------------------------------------------------ */

/*
 * One token per stream through the whole model.
 *   tokens (B, 2) int64, pos (B,) int64, roles (B,) int64 or NULL (then no
 *   role term: the sync model);
 *   conv / ssm: the stacked states, updated in place;
 *   h_out (B, d_model) = the final-norm output, size_logits (B, size_out).
 */
void obfs_step(const obfs_model *m, int B, const int64_t *tokens,
               const int64_t *pos, const int64_t *roles, float *conv,
               float *ssm, float *h_out, float *size_logits) {
    const int dm = m->d_model, di = m->d_inner, dc = m->d_conv;
    const int ds = m->d_state, dr = m->dt_rank, xo = m->x_out;
    const int wt = m->wtype;
    /* per stream: x, xn, uz (2 di), uc, xdbl, dt, y */
    const int per = 2 * dm + 2 * di + di + xo + di + di;
    float work[(size_t)B * per];
#define X_(b) (work + (size_t)(b) * per)
#define XN_(b) (X_(b) + dm)
#define UZ_(b) (XN_(b) + dm)
#define UC_(b) (UZ_(b) + 2 * di)
#define XD_(b) (UC_(b) + di)
#define DT_(b) (XD_(b) + xo)
#define Y_(b) (DT_(b) + di)

    /* the embedding: Linear(2 → d_model) + PE[pos] + role_emb[role] */
    for (int b = 0; b < B; b++) {
        const float sf = (float)tokens[2 * b] / (float)m->emb_Bs;
        const float tf = (float)tokens[2 * b + 1] / (float)m->emb_Bt;
        const float *pe = m->pe + (size_t)pos[b] * dm;
        const float *re = roles ? m->role_emb + (size_t)roles[b] * dm : NULL;
        float *x = X_(b);
        for (int j = 0; j < dm; j++) {
            float v = (m->emb_w[2 * j] * sf + m->emb_w[2 * j + 1] * tf) + m->emb_b[j];
            v += pe[j];
            if (re) v += re[j];
            x[j] = v;
        }
    }

    for (int l = 0; l < m->n_layers; l++) {
        const float *nw = m->norm_w + (size_t)l * dm;
        const float *nb = m->norm_b + (size_t)l * dm;
        const void *in_w = wrow(m->in_proj_w, wt, (size_t)l * 2 * di, dm);
        const float *in_s = srow(m->in_proj_s, (size_t)l * 2 * di);
        const float *cw = m->conv_w + (size_t)l * di * dc;
        const float *cb = m->conv_b + (size_t)l * di;
        const void *xw = wrow(m->x_proj_w, wt, (size_t)l * xo, di);
        const float *xs = srow(m->x_proj_s, (size_t)l * xo);
        const void *dw = wrow(m->dt_proj_w, wt, (size_t)l * di, dr);
        const float *dsc = srow(m->dt_proj_s, (size_t)l * di);
        const float *db = m->dt_proj_b + (size_t)l * di;
        const float *A = m->A + (size_t)l * di * ds;
        const float *D = m->D + (size_t)l * di;
        const void *ow = wrow(m->out_proj_w, wt, (size_t)l * dm, di);
        const float *os = srow(m->out_proj_s, (size_t)l * dm);

        for (int b = 0; b < B; b++) layernorm(X_(b), XN_(b), nw, nb, dm);
        matvec(wt, in_w, in_s, NULL, 2 * di, dm, XN_(0), B, per, UZ_(0), per);

        /* the depthwise causal conv on the shifted state, then silu */
        for (int b = 0; b < B; b++) {
            float *cs = conv + ((size_t)l * B + b) * di * dc;
            const float *u = UZ_(b);
            float *uc = UC_(b);
            for (int c = 0; c < di; c++) {
                float *s = cs + (size_t)c * dc;
                for (int k = 0; k + 1 < dc; k++) s[k] = s[k + 1];
                s[dc - 1] = u[c];
                float acc = 0.0f;
                for (int k = 0; k < dc; k++) acc += s[k] * cw[(size_t)c * dc + k];
                uc[c] = acc + cb[c];
            }
            silu_vec(uc, di);
        }

        matvec(wt, xw, xs, NULL, xo, di, UC_(0), B, per, XD_(0), per);
        matvec(wt, dw, dsc, db, di, dr, XD_(0), B, per, DT_(0), per);
        for (int b = 0; b < B; b++) softplus_vec(DT_(b), di);

        /* the selective scan: s = s·exp(dt·A) + (dt·u) B;  y = <s, C> + D·u */
        for (int b = 0; b < B; b++) {
            float *st = ssm + ((size_t)l * B + b) * di * ds;
            const float *u = UC_(b), *z = UZ_(b) + di, *dt = DT_(b);
            const float *Bv = XD_(b) + dr, *Cv = XD_(b) + dr + ds;
            float *y = Y_(b);
            for (int c = 0; c < di; c++) {
                float *s = st + (size_t)c * ds;
                const float *a = A + (size_t)c * ds;
                const float dtc = dt[c], dtu = dtc * u[c];
                v8f vdt = splat8(dtc), vdtu = splat8(dtu), acc = splat8(0.0f);
                int n = 0;
                for (; n + 8 <= ds; n += 8) {
                    v8f sn = ld8(s + n) * exp8(vdt * ld8(a + n)) + vdtu * ld8(Bv + n);
                    st8(s + n, sn);
                    acc += sn * ld8(Cv + n);
                }
                float yc = hsum8(acc);
                for (; n < ds; n++) {
                    float sn = s[n] * expf(dtc * a[n]) + dtu * Bv[n];
                    s[n] = sn;
                    yc += sn * Cv[n];
                }
                y[c] = yc + D[c] * u[c];
            }
            /* the gate */
            int c = 0;
            for (; c + 8 <= di; c += 8) st8(y + c, ld8(y + c) * silu8(ld8(z + c)));
            for (; c < di; c++) y[c] *= silu1(z[c]);
        }

        /* out_proj into the norm buffer, then the residual */
        matvec(wt, ow, os, NULL, dm, di, Y_(0), B, per, XN_(0), per);
        for (int b = 0; b < B; b++) {
            float *x = X_(b);
            const float *o = XN_(b);
            int j = 0;
            for (; j + 8 <= dm; j += 8) st8(x + j, ld8(x + j) + ld8(o + j));
            for (; j < dm; j++) x[j] += o[j];
        }
    }

    for (int b = 0; b < B; b++)
        layernorm(X_(b), h_out + (size_t)b * dm, m->fnorm_w, m->fnorm_b, dm);
    matvec(wt, m->size_w, m->size_s, m->size_b, m->size_out, dm, h_out, B, dm,
           size_logits, m->size_out);
#undef X_
#undef XN_
#undef UZ_
#undef UC_
#undef XD_
#undef DT_
#undef Y_
}

/*
 * The size-conditioned time head for N rows: out (N, time_out). With
 * shared != 0, h is one row (d_model) used by every bin (as the decisions
 * below use it); else h is (N, d_model).
 */
void obfs_time_logits(const obfs_model *m, int N, const float *h, int shared,
                      const int64_t *bins, float *out) {
    const int dm = m->d_model, to = m->time_out, wt = m->wtype;
    float pre[dm], act[dm];
    if (shared) matvec(wt, m->t1_wh, m->t1_s, NULL, dm, dm, h, 1, dm, pre, dm);
    for (int i = 0; i < N; i++) {
        if (!shared)
            matvec(wt, m->t1_wh, m->t1_s, NULL, dm, dm, h + (size_t)i * dm, 1, dm, pre, dm);
        const float *tab = m->t1_tab + (size_t)bins[i] * dm;
        for (int j = 0; j < dm; j++) {
            float v = pre[j] + tab[j];
            act[j] = v > 0.0f ? v : 0.0f;
        }
        matvec(wt, m->t2_w, m->t2_s, m->t2_b, to, dm, act, 1, dm, out + (size_t)i * to, to);
    }
}

/* ------------------------------------------------------------------------ */
/* the decision                                                              */
/* ------------------------------------------------------------------------ */

/*
 * The random numbers come from the caller: q is a vector of i.i.d. Exp(1)
 * draws, and the bin is argmax(p / q), which is a draw from the softmax p.
 * The byte values inside a bin (record size, gap) are picked by the caller
 * from the tokenizer's intra-bin tables. Masks are byte arrays (1 = allowed).
 */

/* softmax of y (n entries, -inf allowed) into p: e = exp(y - max), p = e / Σe */
static void softmax_vec(const float *y, int n, float *p) {
    float mx = -INFINITY;
    for (int i = 0; i < n; i++)
        if (y[i] > mx) mx = y[i];
    const v8f vmx = splat8(mx);
    int i = 0;
    for (; i + 8 <= n; i += 8) st8(p + i, exp8(ld8(y + i) - vmx));
    for (; i < n; i++) p[i] = expf(y[i] - mx);
    float s = 0.0f;
    for (i = 0; i < n; i++) {
        if (y[i] == -INFINITY) p[i] = 0.0f; /* exp8 clamps; a masked entry is 0 */
        s += p[i];
    }
    for (i = 0; i < n; i++) p[i] /= s;
}

/* one draw from the softmax of y: argmax(p / q) */
static int draw(const float *y, int n, const float *q) {
    float p[n];
    softmax_vec(y, n, p);
    int best = 0;
    float bv = -INFINITY;
    for (int i = 0; i < n; i++) {
        const float r = p[i] / q[i];
        if (r != r) return i;
        if (r > bv) {
            bv = r;
            best = i;
        }
    }
    return best;
}

int obfs_decide_size(const obfs_model *m, const float *h, const float *size_logits,
                     const uint8_t *smask, const uint8_t *tmask, const float *q) {
    const int n = m->size_out, Bs = m->B_s, Bt = m->B_t, to = m->time_out;
    float y[n];
    if (smask) {
        for (int i = 0; i < n; i++) y[i] = smask[i] ? size_logits[i] : -INFINITY;
    } else {
        memcpy(y, size_logits, (size_t)n * sizeof(float));
        if (tmask) { /* + log Σ_{t ∈ tmask} P(t | size) on the regular bins */
            float tl[(size_t)Bs * to], pr[Bt];
            int64_t bins[Bs];
            for (int b = 0; b < Bs; b++) bins[b] = b;
            obfs_time_logits(m, Bs, h, 1, bins, tl);
            for (int b = 0; b < Bs; b++) {
                softmax_vec(tl + (size_t)b * to, Bt, pr);
                float mass = 0.0f;
                for (int t = 0; t < Bt; t++)
                    if (tmask[t]) mass += pr[t];
                y[b] += mass <= 0.0f ? -INFINITY : logf(mass < 1e-30f ? 1e-30f : mass);
            }
        }
    }
    return draw(y, n, q);
}

int obfs_decide_time(const obfs_model *m, const float *h, int size_bin,
                     const uint8_t *tmask, const float *q) {
    const int Bt = m->B_t, to = m->time_out;
    float tl[to], y[Bt];
    const int64_t bin = size_bin;
    obfs_time_logits(m, 1, h, 1, &bin, tl);
    for (int t = 0; t < Bt; t++) y[t] = tmask && !tmask[t] ? -INFINITY : tl[t];
    return draw(y, Bt, q);
}

float obfs_receive_gap_pmf(const obfs_model *m, const float *h, const float *size_logits,
                           int min_time_bin, const uint8_t *recv, double *pmf) {
    const int n = m->size_out, Bs = m->B_s, Bt = m->B_t, to = m->time_out;
    float ps[n];
    softmax_vec(size_logits, n, ps);
    int64_t bins[Bs];
    int nr = 0;
    for (int b = 0; b < Bs; b++)
        if (recv[b]) bins[nr++] = b;
    float tl[(size_t)(nr > 0 ? nr : 1) * to], pr[Bt], mix[Bt];
    if (nr > 0) obfs_time_logits(m, nr, h, 1, bins, tl);
    for (int t = 0; t < Bt; t++) mix[t] = 0.0f;
    float p_recv = 0.0f;
    for (int i = 0; i < nr; i++) {
        const float w = ps[bins[i]];
        p_recv += w;
        softmax_vec(tl + (size_t)i * to, Bt, pr);
        for (int t = 0; t < Bt; t++) mix[t] += w * pr[t];
    }
    const float denom = p_recv > 1e-30f ? p_recv : 1e-30f;
    double total = 0.0;
    for (int t = 0; t < Bt; t++) {
        pmf[t] = t < min_time_bin ? 0.0 : (double)(mix[t] / denom);
        total += pmf[t];
    }
    if (total <= 0.0) {
        for (int t = 0; t < Bt; t++) pmf[t] = 0.0;
        pmf[Bt - 1] = 1.0;
    } else {
        for (int t = 0; t < Bt; t++) pmf[t] /= total;
    }
    return p_recv;
}
