// Split-decode paged attention over a 4-bit quantized KV cache, hand-written for gfx12 (RDNA4) WMMA.
// Drop-in for triton_paged._paged_attn_decode_split_kernel (QCK = QCV = 4, head_dim 256, no window/softcap/sinks):
// same partial_o / partial_ml layout, so the Triton combine kernel finishes the reduction (and the inverse H32
// rotation).
//
// gfx12 v_wmma_f32_16x16x16_f16 (A, B) layouts, probed on gfx1201:
//   A: lane L holds row L%16, k = 8(L/16) .. +7        B: lane L holds column L%16, k = 8(L/16) .. +7
//   D: lane L holds column L%16, rows 8(L/16) .. +7
// A workgroup is 2 waves on one 16-row query tile (grid x = kv-head program x row tile, y = split); wave w owns head
// dims 128 w .. +127. Scores are computed transposed, S^T (tokens x rows) = K . Q^T, each wave over its own 128 dims,
// and the two partial S^T tiles are summed through LDS. A K fragment (8 consecutive head dims of one token) is one
// packed 4-bit word, so K goes straight from global memory into registers; the dim order inside the dot is free as
// long as Q uses the same one, so chunk c of lane half h is word 16 w + 8 h + c (each lane's 8 words contiguous).
// S^T's D layout puts a row's 8 consecutive tokens in one lane, which is the A layout of P for O = P . V. V is the
// only transpose: lane (word 16 w + lr, token half h) decodes 8 tokens of its word and writes one 16-byte row per dim.
// Keeping half the head dims per wave holds the O accumulator at 64 VGPRs, which leaves room for occupancy and for
// the next tile's loads to be in flight during the current tile's softmax and P . V.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

typedef _Float16 h8 __attribute__((ext_vector_type(8)));
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

__global__ __launch_bounds__(64)
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
    int causal,
    int row_tiles                           // block_m * block_h / 16
)
{
#if defined(__gfx1200__) || defined(__gfx1201__)
    // LDS kept small for occupancy: the raw-Q staging only lives before the loop, so it overlays the V tiles
    __shared__ __align__(16) _Float16 vt[2][128 * TILE];    // per wave: V tile, dim-major (its 128 dims)
    _Float16* qs = &vt[0][0];                               // raw Q rows (rotation staging), 16 x HD = both tiles
    __shared__ __align__(16) f8 sx[2][2][32];               // partial S^T exchange [parity][wave][lane]

    const int pid = blockIdx.x / row_tiles;
    const int rt = blockIdx.x % row_tiles;
    const int split = blockIdx.y;
    const int wv = threadIdx.x >> 5;        // dim half
    const int lane = threadIdx.x & 31;
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
    const int q_head = kv_head * group_size + row_h;

    const int total_k_len = cache_seqlens[batch] + kv_append_len;
    const int q_abs = total_k_len - q_len + row_q;
    const int n_start = split * split_len;
    const int n_end = min(n_start + split_len, total_k_len);

    // Stage raw Q rows: thread (wave, lane) copies 64 dims of row lr
    {
        const half* qr = q + ((int64_t) (batch * q_len + row_q) * n_q_heads + q_head) * HD;
        const int d0 = (wv * 2 + lh) * 64;
        #pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            h8 v = h8{};
            if (valid_row) v = *(const h8*) (qr + d0 + i * 8);
            *(h8*) (qs + lr * HD + d0 + i * 8) = v;
        }
    }
    __syncthreads();

    // Rotate Q in place, H32 per 32-group: 16 rows x 8 groups, two (row, group) items per thread
    #pragma unroll 1
    for (int it = threadIdx.x; it < 16 * GPH; it += 64)
    {
        _Float16* gp = qs + (it / GPH) * HD + (it % GPH) * 32;
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
    // Q fragments: chunk c in lane half lh = word 16 wv + 8 lh + c of row lr
    h8 qf[8];
    #pragma unroll
    for (int c = 0; c < 8; ++c) qf[c] = *(const h8*) (qs + lr * HD + (16 * wv + 8 * lh + c) * 8);

    __syncthreads();        // qs (overlaid on vt) fully read before the first V tile is written

    f8 o[8];                // O: lane holds dim 128 wv + 16 dc + lr, rows 8 lh .. +7
    #pragma unroll
    for (int dc = 0; dc < 8; ++dc) o[dc] = f8{};
    float m = -INFINITY;
    float l = 0.0f;

    // Loads for one tile (positions past n_end clamp to the last valid token; masked in the softmax)
    uint4 kw[2];            // K: token lr, words 16 wv + 8 lh .. +7
    float ks8[2];
    uint32_t vw[8];         // V: word 16 wv + lr, tokens 8 lh .. +7
    float vs8[8];
#define TILE_BASE(t0) ((int64_t) block_table[batch * num_pages_per_seq + (t0) / page_size] * page_size \
                       - (int64_t) ((t0) / page_size) * page_size)
#define LOAD_TILE(t0, base)                                                                                   \
    {                                                                                                         \
        const int64_t tk = (base) + min((t0) + lr, n_end - 1);                                                \
        const uint4* kp = (const uint4*) (k_cache + tk * wpt + kv_head * WPH + 16 * wv + 8 * lh);             \
        kw[0] = kp[0];                                                                                        \
        kw[1] = kp[1];                                                                                        \
        const half* ksp = k_scales + tk * gpt + kv_head * GPH + 4 * wv + 2 * lh;                              \
        ks8[0] = (float) ksp[0];                                                                              \
        ks8[1] = (float) ksp[1];                                                                              \
        _Pragma("unroll")                                                                                     \
        for (int t = 0; t < 8; ++t)                                                                           \
        {                                                                                                     \
            const int64_t tv = (base) + min((t0) + 8 * lh + t, n_end - 1);                                    \
            vw[t] = v_cache[tv * wpt + kv_head * WPH + 16 * wv + lr];                                         \
            vs8[t] = (float) v_scales[tv * gpt + kv_head * GPH + 4 * wv + lr / 4];                            \
        }                                                                                                     \
    }
    if (n_start < n_end) LOAD_TILE(n_start, TILE_BASE(n_start))

    int parity = 0;
    for (int n0 = n_start; n0 < n_end; n0 += TILE, parity ^= 1)
    {
        // Partial S^T over this wave's 128 dims
        f8 st = f8{};
        #pragma unroll
        for (int c = 0; c < 8; ++c)
        {
            const uint32_t w = ((const uint32_t*) kw)[c];
            const float s8 = ks8[c / 4] * 0.125f;
            h8 ka;
            #pragma unroll
            for (int j = 0; j < 8; ++j) ka[j] = dq(w, j, s8);
            st = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ka, qf[c], st);
        }
        // V transpose for this wave's dims: dim 8 lr + j (local), tokens 8 lh .. +7
        #pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            h8 col;
            #pragma unroll
            for (int t = 0; t < 8; ++t) col[t] = dq(vw[t], j, vs8[t] * 0.125f);
            *(h8*) (&vt[wv][(lr * 8 + j) * TILE + lh * 8]) = col;
        }
        sx[parity][wv][lane] = st;

        // Next tile's loads go out now and land during the exchange, softmax and P . V
        const int n1 = n0 + TILE;
        if (n1 < n_end) LOAD_TILE(n1, TILE_BASE(n1))

        __syncthreads();
        st += sx[parity][wv ^ 1][lane];

        // Online softmax for row lr (identical in both waves)
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
            const h8 vb = *(const h8*) (&vt[wv][(dc * 16 + lr) * TILE + lh * 8]);
            f8 acc = o[dc];
            #pragma unroll
            for (int i = 0; i < 8; ++i) acc[i] *= ar[i];
            o[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(pa, vb, acc);
        }
        // vt is rewritten next tile: both waves must be past their reads (sx is double-buffered by parity)
        __syncthreads();
    }

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
    const int row_tiles = rows / 16;
    hipLaunchKernelGGL(attn_q4_split_kernel, dim3(programs * row_tiles, num_splits), dim3(64), 0, (hipStream_t) stream,
        (const half*) q, (const uint32_t*) k_cache, (const uint32_t*) v_cache, (const int*) block_table,
        (const int*) cache_seqlens, (float*) partial_o, (float*) partial_ml, (const half*) k_scales,
        (const half*) v_scales, (const half*) h32, split_len, num_pages_per_seq, num_splits, q_len, kv_append_len,
        n_q_heads, n_kv_heads, page_size, scale, block_m, block_h, causal, row_tiles);
    return (int) hipGetLastError();
}
