# Patches

All against ExLlamaV3 `dev` at 0662fac. TabbyAPI needs no patch at 2fd6cc7, which accepts ROCm GPUs natively.

| File | Applied by `setup/build_exllamav3.sh` | What |
|---|---|---|
| `exllamav3-rdna4.patch` | always | Everything of ours: `git diff 0662fac rdna4-dev-batch`. |
| `optional/pr423-dflash2-rejection-sampling.patch` | only with `WITH_PR423=1` (off by default) | Open upstream PR #423 by Rafa (@rafatxf), with its review fixes, applied after ours. |
| `features/*.patch` | no | Our work split into its 18 commits, for reading and review. |

`rdna4-dev-batch` and `rdna4-dev-pr423` are our local work branches; they aren't published, and these patch files are
their published form. The combined patch is the one that's guaranteed: it's what we compile and validate, and it touches only source,
docs and three tests (`test_gdn_replay.py`, `test_gemv32.py`, `test_greedy_equal.py`). The files in `features/` are
the commits of the `rdna4-dev-batch` branch as
`git format-patch` output. Applied in order with `git apply` on 0662fac they give the same tree, but we only check
that for the combined patch.

1. `0001-ROCm-gfx12-WMMA-multi-row-GEMV-EXL3_GEMV_WMMA.patch`: gfx12 WMMA GEMV for 2-8 rows (draft verification).
2. `0002-ROCm-periodically-re-instantiate-HIP-graphs-...patch`: re-instantiate HIP graphs every 100k node updates
   (`EXL3_GRAPH_REINST`).
3. `0003-Split-decode-attention-GQA-grouping-...patch`: whole-GQA-group split-decode attention, `EXL3_DEC_*` tuning and
   the Q4W kernel.
4. `0004-Gated-DeltaNet-accepted-input-replay-...patch`: DeltaNet accepted-input replay (`EXL3_GDN_REPLAY=1`).
5. `0005-Add-test_gdn_replay.py-...patch` and `0006-test_gdn_replay-...patch`: the replay test.
6. `0007` to `0017`, the batching series:
   - `0007`, `0008`, `0012`: the WMMA GEMV extended to 16 and then 32 rows (two M tiles), with the 32-row accumulator
     kept in registers.
   - `0009`, `0014`: the GatedMLP as a captured graph up to 32 rows (`EXL3_BC_MLP_MAX_ROWS`), graphs keyed by row count
     and output dtype.
   - `0015`: LoRA adapters and per-projection controls take the unfused path.
   - `0010`, `0016`: the opt-in `EXL3_DRAFT_ROW_BUDGET` (off by default; it was slower) and DFlash2 row pruning for
     short drafts.
   - `0011`, `0017`: `test_gemv32.py` and `test_greedy_equal.py`.
   - `0013`: docs.
7. `0018`: generic wording in the tests' docstrings, comments and one failure message (no logic change).

The commit messages and the intermediate diffs in `features/` are our work history as it happened, so they mention
local work branches and checkout names (`wt-tree`, `wt-dev`, `$W/...`); `0018` replaces the ones that ended up in the
test files' docstrings. They don't affect the combined patch.

The fixes from the code reviews (`notes/reviews.md`) are folded into these commits.

## PR #423 (optional, off by default)

[turboderp-org/exllamav3#423](https://github.com/turboderp-org/exllamav3/pull/423) by Rafa
([@rafatxf](https://github.com/rafatxf)) adds speculative (rejection) sampling to DFlash2 for stochastic samplers.
Without it, a sampled request verifies DFlash2's greedy draft path by sampling the target and accepting matches. With
it, the draft path is sampled and accepted with probability min(1, p/q), which keeps the target's output distribution.
Greedy requests and other samplers take the old path, and `EXL3_DFLASH_SPEC=0` turns it off at runtime. The patch is
the PR's two commits (eef016b and ffd18f5 upstream) rebased onto `rdna4-dev-batch`, with Rafa as their author. The PR
was still open (not merged) on 2026-10-07. The README's numbers were measured without it.

Rebasing the first commit onto the batching series took one change to the PR's code. In
`DFlash2Model.sample_from_state`, the batching series already slices the state and logits to the path rows (and prunes
them to `dflash2_rows` for short drafts), so the PR's sampled branch now passes those (`path_state`, `logits`) to
`walk_sample` instead of slicing `state[:, 1:]` / `logits[:, 1:]` a second time. Two textual conflicts were resolved by
keeping both sides: the docs (our row-budget entry and the PR's two entries, under the one `## Speculative decoding`
heading) and the generator's draft parameters (our `dflash2_rows`, then the PR's sampling setup). The PR's other
changed lines are identical to the original commit. The commit message records the change. The second commit applied
unchanged.

Our review of the first commit found four problems, and Rafa fixed all of them in the second (details in the PR thread
and `notes/reviews.md`):

- A sampled request's speculative verify skipped job-level token masks (`min_new_tokens`, banned-string
  continuations), so a stop token could end generation before `min_new_tokens`.
- `probs()` kept exactly k tokens at a top-k tie, while the fused sampler keeps all tied tokens.
- The sampled DFlash2 path didn't export `draft_conf`, which silently turned off confidence calibration. It now doesn't
  sample speculatively when the calibrator is on.
- Extra device-to-host syncs.

On the GPU (2026-10-07, both commits on the batching series, the same build with `EXL3_DFLASH_SPEC=1` vs `0`): the PR's
GPU tests pass (22); six sampled prose requests capped at 500 tokens (temperature 0.8, top-p 0.95, top-k 20) went from
52.6 to 53.2 tok/s end to end, with draft acceptance 20.3% -> 21.5%; `min_tokens = 64` is honored; greedy output is
identical. Each prompt ran once per mode, so that small difference isn't an established speedup. On the pre-batching
rebase, measured as decode speed in the server log rather than end to end, it was 59.3 -> 62.6 tok/s. It stays off by
default: it changes how sampled requests are verified and gains little on this setup.

`WITH_PR423=1 setup/build_exllamav3.sh` applies it; switching an existing build on or off needs `RESET=1`.

## Regenerating

`setup/regen_patches.sh` writes all three kinds from a checkout that has the work branches. The exact commands are in
its header. After a port to a newer upstream, regenerate and bump `EXLLAMAV3_COMMIT` in `setup/versions.sh`.

## Licenses

exllamav3 is MIT-licensed, and these patches are distributed under its license. The PR #423 patch is Rafa's code from
a pull request to exllamav3; see the PR for its terms.
