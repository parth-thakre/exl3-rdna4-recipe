// Does hipGraphExecKernelNodeSetParams leak per call on this ROCm? Update one kernel node's argument N times
// (alternating between two pointers, as exllamav3's Graph::launch does every token) and watch device memory.
// usage: graph_setparams_repro [updates] [reinstantiate_every (0 = never)]
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void touch(float* p) { if (threadIdx.x == 0) p[0] += 1.0f; }

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("%s failed: %s at update %ld\n", #x, hipGetErrorString(e), i); return 1; } } while (0)

int main(int argc, char** argv)
{
    long n = argc > 1 ? atol(argv[1]) : 1000000;
    long reinst = argc > 2 ? atol(argv[2]) : 0;
    long i = 0;
    float *a, *b;
    CK(hipMalloc(&a, 4)); CK(hipMalloc(&b, 4));
    hipStream_t s; CK(hipStreamCreate(&s));

    hipGraph_t g; CK(hipGraphCreate(&g, 0));
    void* args[1] = { &a };
    hipKernelNodeParams kp = {};
    kp.func = (void*) touch; kp.gridDim = dim3(1); kp.blockDim = dim3(32); kp.kernelParams = args;
    hipGraphNode_t node; CK(hipGraphAddKernelNode(&node, g, nullptr, 0, &kp));
    hipGraphExec_t ge; CK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));

    size_t free0, total; CK(hipMemGetInfo(&free0, &total));
    for (i = 0; i < n; ++i)
    {
        float* p = (i & 1) ? b : a;
        args[0] = &p;
        CK(hipGraphExecKernelNodeSetParams(ge, node, &kp));
        CK(hipGraphLaunch(ge, s));
        if (reinst && (i + 1) % reinst == 0)
        {
            CK(hipStreamSynchronize(s));
            CK(hipGraphExecDestroy(ge));
            CK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
        }
        if ((i + 1) % (n / 10) == 0)
        {
            CK(hipStreamSynchronize(s));
            size_t f; CK(hipMemGetInfo(&f, &total));
            printf("after %8ld updates: device memory used by the graph so far %8.1f MB\n", i + 1, (double) (free0 - f) / 1e6);
            fflush(stdout);
        }
    }
    CK(hipStreamSynchronize(s));
    printf("done, %ld updates without a crash\n", n);
    return 0;
}
