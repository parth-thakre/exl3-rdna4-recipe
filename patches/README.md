# Patches

All against ExLlamaV3 v1.6.5 (cdc785d, released 2026-10-25). Before 2026-10-27 the base was `dev` 0f685aa (v1.6.1 plus
two days of commits); that patch is in this repository's history.
TabbyAPI needs no patch: it accepts ROCm GPUs natively since 2fd6cc7 (the recipe uses a3fed43).

| File | Applied by `setup/build_exllamav3.sh` | What |
|---|---|---|
| `exllamav3-rdna4.patch` | always | Everything of ours: `git diff v1.6.5 live-165`. |
| `features/*.patch` | no | The same as its 15 commits, for reading and review. |
| `previous/exllamav3-rdna4-0662fac.patch` | only if you select the previous base (`setup/versions.sh`) | The patch for the previous base, 0662fac, unchanged. |
| `previous/pr423-dflash2-rejection-sampling-0662fac.patch` | only with `WITH_PR423=1`, on the previous base | Open upstream PR #423 by Rafa (@rafatxf), with its review fixes. |

`live-165` is our local work branch; it isn't published, and these patch files are its published form. The combined
patch is the one that's guaranteed: it's what we compile and validate, and it touches only source, docs and three tests
(`test_gemv32.py`, `test_greedy_equal.py`, `tests/kernels/attention/test_paged_decode_layout.py`). The files in
`features/` are the commits of `live-165` as `git format-patch` output. Applied in order with `git am` on v1.6.5 they
give the same tree as the combined patch (checked 2026-10-27).

1. `0001` to `0003`, `0006`: gfx12 WMMA GEMV for 2-8 rows (draft verification), then 16 and 32 rows (two M tiles), with
   the 32-row accumulator kept in registers (`EXL3_GEMV_WMMA`).
2. `0004`, `0008`: the GatedMLP as a captured graph up to 32 rows (`EXL3_BC_MLP_MAX_ROWS`), graphs keyed by row count and
   output dtype.
3. `0009`: LoRA adapters and per-projection controls take the unfused path.
4. `0005`, `0010`, `0011`: `test_gemv32.py` and `test_greedy_equal.py`, and generic wording in the tests.
5. `0007`: docs.
6. `0012`, `0013`: our split-decode attention tuning on upstream's dense (query, head) row layout (v1.6.0): `EXL3_DEC_*`
   overrides, the larger row cap with `EXL3_DEC_GROUP=1`, and the Q4W kernel for the 4-bit cache; tests and docs.
7. `0014`, `0015`: a multi-stage split-decode tile that doesn't fit in shared memory falls back to one stage (it only
   happened with `EXL3_DEC_STAGES=2` at 16 q heads per kv head, not on Qwen3.8), and the launch pick is keyed on the
   single-split variant.

The port from 0f685aa to v1.6.5 (2026-10-27) needed one hand edit, in `0004`. turboderp's f0dd8f6 replaced the per-row
gate/up buffers of `BC_GatedMLP` with views of the shared statics (`slot_pair_view`), the same change `0004` makes; we
kept his version and raised its limit from `MAX_BSZN` (8) to `MLP_MAX_ROWS` (32), with our stricter scratch-size check
in the constructor. `0004`'s commit message still describes the views as its own change. The other 14 commits applied
unchanged.

Upstream since 0f685aa: our DeltaNet scan change (each thread keeps its slice of the recurrent state in registers across
the tokens of a pass, which made the 8-token verify's DeltaNet layers a fifth faster here) and the Mamba2 replay launch
fix. We filed them as issues ([#505](https://github.com/turboderp-org/exllamav3/issues/505),
[#506](https://github.com/turboderp-org/exllamav3/issues/506)) and turboderp merged both the same day; the kernel code
is the same as ours was. On the 9cc3bd1 base they were `0016` to `0018`, which are in this repository's history
(61ed7e2).

What the previous base's patch had and this one doesn't:

- Our DeltaNet accepted-input replay (`EXL3_GDN_REPLAY`) and its test: upstream's v1.6.1 replay does the same job and is
  always on, and since 0f685aa it has our speed. `EXL3_GDN_REPLAY` is now ignored.
- HIP graph re-instantiation (`EXL3_GRAPH_REINST`): upstream turns graphs off on ROCm by default, so it isn't needed.
  `EXL3_GRAPH_REINST` is ignored unless you build the previous base.
- The opt-in `EXL3_DRAFT_ROW_BUDGET` (off by default and slower) and DFlash2 row pruning for short drafts, which clashed
  with upstream's new hybrid draft mode.
- turboderp's 96838c9 backport (graphs off on ROCm), which is now upstream.

The commit messages in `features/` are our work history as it happened, so some mention local checkout names
(`wt-tree`, `$W/...`); `0011` replaces the ones that ended up in the test files' docstrings. They don't affect the
combined patch.

The fixes from the code reviews (`notes/reviews.md`) are folded into these commits.

## Previous base

The previous release of this recipe was on 0662fac, and its patch is kept in `previous/`. Select it with
`EXLLAMAV3_COMMIT=0662fac591992c6d5f11d041b1cf2136d62e536d EXLLAMAV3_PATCH=patches/previous/exllamav3-rdna4-0662fac.patch`
(and `RESET=1` for an existing checkout). Its 19 per-commit patches are in this repository's history before 2026-10-12.

## PR #423 (optional, off by default, previous base only)

[turboderp-org/exllamav3#423](https://github.com/turboderp-org/exllamav3/pull/423) by Rafa
([@rafatxf](https://github.com/rafatxf)) adds speculative (rejection) sampling to DFlash2 for stochastic samplers.
Without it, a sampled request verifies DFlash2's greedy draft path by sampling the target and accepting matches. With
it, the draft path is sampled and accepted with probability min(1, p/q), which keeps the target's output distribution.
Greedy requests and other samplers take the old path, and `EXL3_DFLASH_SPEC=0` turns it off at runtime. The patch is the
PR's two commits (eef016b and ffd18f5 upstream) rebased onto `rdna4-dev-batch-nographs`, with Rafa as their author. The
PR was still open (not merged) on 2026-10-07. The README's numbers were measured without it.

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

`WITH_PR423=1 setup/build_exllamav3.sh` applies it (previous base only); switching an existing build on or off needs
`RESET=1`.

It hasn't been ported to the current base: upstream has since rewritten `DFlash2Model.sample_from_state` (tensor
parallel), and the PR's sampled branch relied on our row pruning, which the current patch no longer has. To use it,
select the previous base as above and add `WITH_PR423=1`.

## Regenerating

`setup/regen_patches.sh` writes the combined patch, the per-commit files and mail patches from a checkout that has the work branches. The exact commands are in
its header. After a port to a newer upstream, regenerate and bump `EXLLAMAV3_COMMIT` in `setup/versions.sh`.

## Licenses

exllamav3 is MIT-licensed, and these patches are distributed under its license. The PR #423 patch is Rafa's code from
a pull request to exllamav3; see the PR for its terms.
