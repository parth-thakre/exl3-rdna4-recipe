# Kernel experiments

None of this is needed to run the recipe. These are the standalone HIP programs behind some of the patches, and a
hand-written attention kernel that didn't beat Triton. Build them with `kernels/build.sh` (system `hipcc`, gfx1201).

## `wmma_layout_probe.hip`

Probes the gfx12 `v_wmma_f32_16x16x16_f16` register layout on the card. It fills X and W with distinct small integers,
loads the operands per a hypothesised lane layout, dumps each lane's accumulators, and checks which (row, col) of X*W
each slot holds. The WMMA multi-row GEMV patch is built on the layout it confirmed:

- A: lane L holds row L%16, k = 8(L/16) .. +7
- B: lane L holds column L%16, k = 8(L/16) .. +7
- D: lane L holds column L%16, rows 8(L/16) .. +7

## `graph_setparams_repro.hip`

Reproduces the HIP graph kernel-argument exhaustion: it updates one kernel node's argument N times, alternating between
two pointers the way exllamav3's `Graph::launch` does every token. On our ROCm, `hipGraphExecKernelNodeSetParams`
runs out of a fixed kernel-argument pool and segfaults after ~1.6M updates (about 1 h of serving). Re-instantiating
the graph periodically avoids it, which is what the graph re-instantiate patch does.

```
./graph_setparams_repro [updates] [reinstantiate_every (0 = never)]
```

## `attn_q4_*.hip`: hand-written gfx12 attention (not used)

Split-decode paged attention over the 4-bit KV cache with gfx12 WMMA. It computes S^T = K . Q^T, so packed 4-bit
words decode straight into A fragments, and P feeds O = P . V in its native layout; only V goes through LDS to be
transposed. `attn_q4_hip.hip` is the same file as `attn_q4_v3.hip`, the best variant, and is what `build.sh` turns
into `libattn_q4.so`. `test_attn_hip.py` checks it against Triton (max error 1.9e-6) and times both.

60k deep, q_len 8, per layer: Triton packed 600 µs, Triton Q4W 510 µs (what the recipe runs), this kernel 598 µs at
best.

| Variant | µs/layer | Notes |
|---|---|---|
| v1: 4 waves share LDS, 16-token tiles | 1595 | latency-bound: dependent loads, 2 barriers per tile |
| v2: 1 wave per 16 rows | 1530 | 256 VGPRs plus spills (the O accumulator alone is 128 VGPRs) |
| v3: 2 waves per 16 rows, dims split, in-place Q rotation, LDS overlay | 598 | 192 VGPRs, no spills |
| v4: 8 waves decode each tile once into LDS | 711 | barrier-bound; 771 with 32-token tiles, which spill |

Only the v3 and v4 sources are here.

If someone picks this up: double-buffered LDS tiles (one barrier per tile), v3's per-row-tile layout with the K/V
decode shared across row tiles by a single wave pair, and checking whether 32-token tiles can run without spills.
