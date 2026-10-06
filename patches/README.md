# Patches

All against ExLlamaV3 `dev` at 0662fac. TabbyAPI needs no patch at 2fd6cc7, which accepts ROCm GPUs natively.

| File | Applied by `setup/build_exllamav3.sh` | What |
|---|---|---|
| `exllamav3-rdna4.patch` | always | Everything of ours: `git diff 0662fac rdna4-dev`. |
| `optional/pr423-dflash2-rejection-sampling.patch` | by default; `WITH_PR423=0` skips it | Upstream PR #423 by Rafa (@rafatxf), applied after ours. |
| `features/*.patch` | no | Our work split into its six commits, for reading and review. |

The combined patch is the one that's guaranteed: it's what we compile and validate, and it touches only source,
docs and one test (`test_gdn_replay.py`). The files in `features/` are the commits of the `rdna4-dev` branch as
`git format-patch` output. Applied in order with `git apply` on 0662fac they give the same tree, but we only check
that for the combined patch.

1. `0001-ROCm-gfx12-WMMA-multi-row-GEMV-EXL3_GEMV_WMMA.patch`: gfx12 WMMA GEMV for 2-8 rows (draft verification).
2. `0002-ROCm-periodically-re-instantiate-HIP-graphs-...patch`: re-instantiate HIP graphs every 100k node updates
   (`EXL3_GRAPH_REINST`).
3. `0003-Split-decode-attention-GQA-grouping-...patch`: whole-GQA-group split-decode attention, `EXL3_DEC_*` tuning and
   the Q4W kernel.
4. `0004-Gated-DeltaNet-accepted-input-replay-...patch`: DeltaNet accepted-input replay (`EXL3_GDN_REPLAY=1`).
5. `0005-Add-test_gdn_replay.py-...patch` and `0006-test_gdn_replay-...patch`: the replay test.

The fixes from both code reviews (`notes/reviews.md`) are folded into these commits.

## PR #423 (optional)

[turboderp-org/exllamav3#423](https://github.com/turboderp-org/exllamav3/pull/423) by Rafa
([@rafatxf](https://github.com/rafatxf)) adds speculative (rejection) sampling to DFlash2 for stochastic samplers.
Without it, a sampled request verifies DFlash2's greedy draft path by sampling the target and accepting matches. With
it, the draft path is sampled and accepted with probability min(1, p/q), which keeps the target's output distribution.
Greedy requests and other samplers take the old path, and `EXL3_DFLASH_SPEC=0` turns it off at runtime. The patch is
the PR's commit (eef016b upstream) unchanged, rebased onto `rdna4-dev`, with Rafa as its author. The PR was still open
(not merged) on 2026-10-06. It's optional because it hasn't been through our own review yet. The numbers in the main
README were measured without it, and we haven't benchmarked sampled decoding with it.

## Regenerating

`setup/regen_patches.sh` writes all three kinds from a checkout that has the work branches. The exact commands are in
its header. After a port to a newer upstream, regenerate and bump `EXLLAMAV3_COMMIT` in `setup/versions.sh`.

## Licenses

exllamav3 is MIT-licensed, and these patches are distributed under its license. The PR #423 patch is Rafa's code from
a pull request to exllamav3; see the PR for its terms.
