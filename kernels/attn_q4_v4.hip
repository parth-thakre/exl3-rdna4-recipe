// Split-decode paged attention over a 4-bit quantized KV cache, hand-written for gfx12 (RDNA4) WMMA.
// Drop-in for triton_paged._paged_attn_decode_split_kernel (QCK = QCV = 4, head_dim 256, no window/softcap/sinks):
// same partial_o / partial_ml layout, so the Triton combine kernel finishes the reduction (and the inverse H32
// rotation).
//
// gfx12 v_wmma_f32_16x16x16_f16 (A, B) layouts, probed on gfx1201:
//   A: lane L holds row L%16, k = 8(L/16) .. +7        B: lane L holds column L%16, k = 8(L/16) .. +7
//   D: lane L holds column L%16, rows 8(L/16) .. +7
// One workgroup per (kv-head program, split) with 2 * RT waves: wave (rt, wv) owns query rows 16 rt .. +15 and head
// dims 128 wv .. +127. Each 16-token tile is loaded and decoded once, cooperatively, into LDS: K token-major (a K
// fragment, 8 consecutive head dims of one token, is one packed word, so a 16-byte LDS read), V dim-major (the one
// transpose). Scores are computed transposed, S^T (tokens x rows) = K . Q^T, each wave over its 128 dims, and the
// two partials of a row tile are summed through LDS. S^T's D layout puts a row's 8 consecutive tokens in one lane,
// which is exactly the A layout of P for O = P . V, so softmax output feeds the second WMMA without any shuffle.
// The next tile's raw words are loaded into registers before the current tile's math.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

typedef _Float16 h8 __attribute__((ext_vector_type(8)));
typedef _Float16 h4 __attribute__((ext_vector_type(4)));
typedef float f8 __attribute__((ext_vector_type(8)));

#define HD 256          // head dim
#define WPH 32          // 4-bit words per head per token
#define GPH 8           // 32-value scale groups per head
#define TILE 16         // tokens per tile

__device__ __forceinline__ _Float16 dq(uint32_t w, int j, float s8)
{
    // (nibble j - 7.5) * scale / 8, nibble j at bits 4j (matches _qc_load_kt, BITS = 4)
    return (_Float16) (((float) ((w >> (4 * j)) & 15u) - 7.5f) * s8);
}

template <int RT>
__global__ __launch_bounds__(RT * 64)
void attn_q4_split_kernel
(
    const half* __restrict__ q,             // (bsz, q_len, n_q_heads, HD)
    const uint32_t* __restrict__ k_cache,   // (pages, page, n_kv_heads * WPH)
    const uint32_t* __restrict__ v_cache,
    const int* __restrict__ block_table,    // (bsz, num_pages_per_seq)
    const int* __restrict__ cache_seqlens,  // (bsz)
    float* __restrict__ partial_o,          // (programs, splits, BLOCK_ROWS, HD)
    float* __restrict__ partial_ml,         // (programs, splits, BLOCK_ROWS, 2)
    const half* __restrict__ k_scales,      // (pages, page, n_kv_heads * GPH)
    const half* __restrict__ v_scales,
    const half* __restrict__ h32,           // 32 x 32, H / sqrt(32), symmetric
    int split_len,
    int num_pages_per_seq,
    int num_splits,
    int q_len,
    int kv_append_len,
    int n_q_heads,
    int n_kv_heads,
    int page_size,
    float scale,
    int block_m,                            // next_pow2(q_len)
    int block_h,                            // heads per program (whole GQA group, padded)
    int causal
)
{
#if defined(__gfx1200__) || defined(__gfx1201__)
    constexpr int NT = RT * 64;
    __shared__ __align__(16) _Float16 ks[TILE * HD];        // K tile, token-major (8 KB)
    __shared__ __align__(16) _Float16 vt[HD * TILE];        // V tile, dim-major (8 KB)
    __shared__ __align__(16) f8 sx[RT][2][32];              // partial S^T per (row tile, dim half)

    const int pid = blockIdx.x;
    const int split = blockIdx.y;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int rt = wave >> 1;               // row tile
    const int wv = wave & 1;                // dim half
    const int lane = tid & 31;
    const int lr = lane & 15;
    const int lh = lane >> 4;

    const int group_size = n_q_heads / n_kv_heads;
    const int h_blocks = (group_size + block_h - 1) / block_h;
    const int h_block = pid % h_blocks;
    const int bh = pid / h_blocks;
    const int batch = bh / n_kv_heads;
    const int kv_head = bh - batch * n_kv_heads;
    const int block_rows = block_m * block_h;
    const int gpt = n_kv_heads * GPH;
    const int wpt = n_kv_heads * WPH;

    const int row = rt * 16 + lr;
    const int row_q = row % block_m;
    const int row_h = h_block * block_h + row / block_m;
    const bool valid_row = row_q < q_len && row_h < group_size;

    const int total_k_len = cache_seqlens[batch] + kv_append_len;
    const int q_abs = total_k_len - q_len + row_q;
    const int n_start = split * split_len;
    const int n_end = min(n_start + split_len, total_k_len);

    // Q fragments, rotated by H32 per 32-group. One row tile at a time through the K tile buffer (16 x HD): stage
    // raw rows, rotate each (row, group) in place, then the tile's two waves pick up their fragments.
    h8 qf[8];               // chunk c in lane half lh = word 16 wv + 8 lh + c of row lr
    #pragma unroll 1
    for (int t = 0; t < RT; ++t)
    {
        for (int i = tid; i < 16 * HD / 8; i += NT)
        {
            const int r = i / (HD / 8), d = (i % (HD / 8)) * 8;
            const int rr = t * 16 + r;
            const int rq = rr % block_m, rh = h_block * block_h + rr / block_m;
            h8 v = h8{};
            if (rq < q_len && rh < group_size)
                v = *(const h8*) (q + ((int64_t) (batch * q_len + rq) * n_q_heads + kv_head * group_size + rh) * HD + d);
            *(h8*) (ks + r * HD + d) = v;
        }
        __syncthreads();
        #pragma unroll 1
        for (int i = tid; i < 16 * GPH; i += NT)
        {
            _Float16* gp = ks + (i / GPH) * HD + (i % GPH) * 32;
            float x[32];
            #pragma unroll
            for (int e = 0; e < 32; ++e) x[e] = (float) gp[e];
            #pragma unroll 1
            for (int j0 = 0; j0 < 32; j0 += 8)
            {
                float y[8] = {};
                #pragma unroll
                for (int e = 0; e < 32; ++e)
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) y[j] += x[e] * (float) h32[e * 32 + j0 + j];
                #pragma unroll
                for (int j = 0; j < 8; ++j) gp[j0 + j] = (_Float16) y[j];
            }
        }
        __syncthreads();
        if (rt == t)
        {
            #pragma unroll
            for (int c = 0; c < 8; ++c) qf[c] = *(const h8*) (ks + lr * HD + (16 * wv + 8 * lh + c) * 8);
        }
        __syncthreads();
    }

    f8 o[8];                // O: lane holds dim 128 wv + 16 dc + lr, rows 8 lh .. +7 (of row tile rt)
    #pragma unroll
    for (int dc = 0; dc < 8; ++dc) o[dc] = f8{};
    float m = -INFINITY;
    float l = 0.0f;

    // Cooperative tile load: threads [0, NT/2) take K words, threads [NT/2, NT) take V items (word w, token quad).
    // Raw words + scales are held in registers from the previous iteration.
    constexpr int KPT = TILE * WPH / (NT / 2);      // K words per K thread
    constexpr int VPT = TILE * WPH / (NT / 2);      // V words per V thread (multiple of 4: whole token quads)
    static_assert(VPT % 4 == 0, "V items are token quads");
    const bool is_k = tid < NT / 2;
    const int lt = is_k ? tid : tid - NT / 2;
    uint32_t rw[KPT];
    float rs[KPT];

#define TILE_BASE(t0) ((int64_t) block_table[batch * num_pages_per_seq + (t0) / page_size] * page_size \
                       - (int64_t) ((t0) / page_size) * page_size)
#define LOAD_TILE(t0)                                                                                         \
    {                                                                                                         \
        const int64_t base = TILE_BASE(t0);                                                                   \
        if (is_k)                                                                                             \
        {                                                                                                     \
            _Pragma("unroll")                                                                                 \
            for (int u = 0; u < KPT; ++u)                                                                     \
            {                                                                                                 \
                const int it = lt * KPT + u, t = it / WPH, w = it % WPH;                                      \
                const int64_t tok = base + min((t0) + t, n_end - 1);                                          \
                rw[u] = k_cache[tok * wpt + kv_head * WPH + w];                                               \
                rs[u] = (float) k_scales[tok * gpt + kv_head * GPH + w / 4];                                  \
            }                                                                                                 \
        }                                                                                                     \
        else                                                                                                  \
        {                                                                                                     \
            _Pragma("unroll")                                                                                 \
            for (int u = 0; u < VPT; ++u)                                                                     \
            {                                                                                                 \
                const int item = lt * (VPT / 4) + u / 4, w = item % WPH, tq = item / WPH;                     \
                const int64_t tok = base + min((t0) + tq * 4 + u % 4, n_end - 1);                             \
                rw[u] = v_cache[tok * wpt + kv_head * WPH + w];                                               \
                rs[u] = (float) v_scales[tok * gpt + kv_head * GPH + w / 4];                                  \
            }                                                                                                 \
        }                                                                                                     \
    }

    if (n_start < n_end) LOAD_TILE(n_start)

    for (int n0 = n_start; n0 < n_end; n0 += TILE)
    {
        // Decode the held words into LDS
        if (is_k)
        {
            #pragma unroll
            for (int u = 0; u < KPT; ++u)
            {
                const int it = lt * KPT + u, t = it / WPH, w = it % WPH;
                const float s8 = rs[u] * 0.125f;
                h8 r;
                #pragma unroll
                for (int j = 0; j < 8; ++j) r[j] = dq(rw[u], j, s8);
                *(h8*) (ks + t * HD + w * 8) = r;
            }
        }
        else
        {
            #pragma unroll
            for (int g = 0; g < VPT / 4; ++g)
            {
                const int item = lt * (VPT / 4) + g, w = item % WPH, tq = item / WPH;
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    h4 col;
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) col[t] = dq(rw[g * 4 + t], j, rs[g * 4 + t] * 0.125f);
                    *(h4*) (vt + (w * 8 + j) * TILE + tq * 4) = col;
                }
            }
        }
        __syncthreads();

        // Next tile's words go out now; they land during this tile's math
        if (n0 + TILE < n_end) LOAD_TILE(n0 + TILE)

        // Partial S^T over this wave's 128 dims
        f8 st = f8{};
        #pragma unroll
        for (int c = 0; c < 8; ++c)
        {
            const h8 ka = *(const h8*) (ks + lr * HD + (16 * wv + 8 * lh + c) * 8);
            st = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ka, qf[c], st);
        }
        sx[rt][wv][lane] = st;
        __syncthreads();
        st += sx[rt][wv ^ 1][lane];

        // Online softmax for row lr (identical in both waves of the row tile)
        float tmax = -INFINITY;
        #pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            const int n = n0 + lh * 8 + i;
            bool ok = valid_row && n < n_end;
            if (causal) ok = ok && n <= q_abs;
            st[i] = ok ? st[i] * scale : -INFINITY;
            tmax = fmaxf(tmax, st[i]);
        }
        tmax = fmaxf(tmax, __shfl_xor(tmax, 16, 32));
        const float m_new = fmaxf(m, tmax);
        const float m_exp = (m_new == -INFINITY) ? 0.0f : m_new;
        const float alpha = (m == -INFINITY) ? 0.0f : __expf(m - m_exp);
        h8 pa;
        float psum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            const float p = (st[i] == -INFINITY) ? 0.0f : __expf(st[i] - m_exp);
            psum += p;
            pa[i] = (_Float16) p;
        }
        psum += __shfl_xor(psum, 16, 32);
        l = l * alpha + psum;
        m = m_new;
        float ar[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) ar[i] = __shfl(alpha, lh * 8 + i, 32);

        // O += P . V over this wave's 8 dim chunks
        #pragma unroll
        for (int dc = 0; dc < 8; ++dc)
        {
            const h8 vb = *(const h8*) (vt + (wv * 128 + dc * 16 + lr) * TILE + lh * 8);
            f8 acc = o[dc];
            #pragma unroll
            for (int i = 0; i < 8; ++i) acc[i] *= ar[i];
            o[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(pa, vb, acc);
        }
        __syncthreads();    // ks / vt / sx are rewritten by the next tile
    }
#undef LOAD_TILE
#undef TILE_BASE

    const int64_t pbase = ((int64_t) pid * num_splits + split) * block_rows;
    #pragma unroll
    for (int dc = 0; dc < 8; ++dc)
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            partial_o[(pbase + rt * 16 + lh * 8 + i) * HD + wv * 128 + dc * 16 + lr] = o[dc][i];
    if (wv == 0 && lh == 0)
    {
        partial_ml[(pbase + row) * 2] = m;
        partial_ml[(pbase + row) * 2 + 1] = l;
    }
#endif
}

extern "C" int attn_q4_launch
(
    const void* q, const void* k_cache, const void* v_cache, const void* block_table, const void* cache_seqlens,
    void* partial_o, void* partial_ml, const void* k_scales, const void* v_scales, const void* h32,
    int split_len, int num_pages_per_seq, int num_splits, int programs,
    int q_len, int kv_append_len, int n_q_heads, int n_kv_heads, int page_size, float scale,
    int block_m, int block_h, int causal, void* stream
)
{
    const int rows = block_m * block_h;
    if (rows % 16 || page_size % TILE) return -1;
    dim3 grid(programs, num_splits);
    #define L(RT_) hipLaunchKernelGGL(attn_q4_split_kernel<RT_>, grid, dim3(RT_ * 64), 0, (hipStream_t) stream, \
        (const half*) q, (const uint32_t*) k_cache, (const uint32_t*) v_cache, (const int*) block_table, \
        (const int*) cache_seqlens, (float*) partial_o, (float*) partial_ml, (const half*) k_scales, \
        (const half*) v_scales, (const half*) h32, split_len, num_pages_per_seq, num_splits, q_len, kv_append_len, \
        n_q_heads, n_kv_heads, page_size, scale, block_m, block_h, causal)
    switch (rows / 16)
    {
        case 1: L(1); break;
        case 2: L(2); break;
        case 4: L(4); break;
        default: return -1;
    }
    #undef L
    return (int) hipGetLastError();
}
