// Probe the gfx12 v_wmma_f32_16x16x16_f16 register layout on the real card.
// X[16][16] (rows M, K) and W[16][16] (K, cols N) are filled with distinct small integers; each lane loads
// the operands per a hypothesised layout, and the kernel dumps every lane's 8 accumulator floats. The host
// then checks which (row, col) of X*W each slot equals.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>

typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));

// Hypothesis: first operand lane L = row (L % 16) of X, K = 8*(L/16)..+7;
//             second operand lane L = column (L % 16) of W, K = 8*(L/16)..+7
__global__ void probe(const _Float16* X, const _Float16* W, float* out, int swap)
{
    int L = threadIdx.x;
    half8 a, b;
    for (int i = 0; i < 8; i++)
    {
        int k = 8 * (L / 16) + i;
        a[i] = X[(L % 16) * 16 + k];
        b[i] = W[k * 16 + (L % 16)];
    }
    float8 c = {};
    c = swap ? __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b, a, c)
             : __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
    for (int i = 0; i < 8; i++) out[L * 8 + i] = c[i];
}

int main()
{
    _Float16 hX[256], hW[256];
    float ref[16][16];
    for (int i = 0; i < 256; i++) { hX[i] = (_Float16)((i * 7) % 13 - 6); hW[i] = (_Float16)((i * 5) % 11 - 5); }
    for (int r = 0; r < 16; r++)
        for (int n = 0; n < 16; n++)
        {
            float s = 0;
            for (int k = 0; k < 16; k++) s += (float)hX[r * 16 + k] * (float)hW[k * 16 + n];
            ref[r][n] = s;
        }
    _Float16 *dX, *dW; float* dO;
    hipMalloc(&dX, 512); hipMalloc(&dW, 512); hipMalloc(&dO, 32 * 8 * 4);
    hipMemcpy(dX, hX, 512, hipMemcpyHostToDevice); hipMemcpy(dW, hW, 512, hipMemcpyHostToDevice);
    for (int swap = 0; swap < 2; swap++)
    {
        float h[256];
        probe<<<1, 32>>>(dX, dW, dO, swap);
        hipMemcpy(h, dO, sizeof(h), hipMemcpyDeviceToHost);
        // Test layout hypotheses for D: slot (L, i) = D[row][col]
        int ok_a = 0, ok_b = 0;   // a: row = 8*(L/16)+i, col = L%16 ; b: row = L%16, col = 8*(L/16)+i
        for (int L = 0; L < 32; L++)
            for (int i = 0; i < 8; i++)
            {
                float v = h[L * 8 + i];
                if (fabsf(v - ref[8 * (L / 16) + i][L % 16]) < 1e-3f) ok_a++;
                if (fabsf(v - ref[L % 16][8 * (L / 16) + i]) < 1e-3f) ok_b++;
            }
        printf("operands (%s): D[row=8*(L/16)+i][col=L%%16] matches %d/256, D[row=L%%16][col=8*(L/16)+i] matches %d/256\n",
               swap ? "W, X" : "X, W", ok_a, ok_b);
    }
    return 0;
}
