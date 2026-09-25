# Handoff: binocular Nethra (state as of 2026-09-25)

Read in this order before doing anything: `CLAUDE.md`, `nethra/NETHRA_OPERATING_NOTES.md`, this file,
then `nethra/nethra.py`. Work on branch `claude/sharp-thompson-wmszc5`. Scratchpad files from earlier
sessions are gone; everything needed is in the repo.

## Goal (the user's words, condensed)

Extend Nethra, without ML ideas and without assuming what it does: binocular vision input, track
objects in a 500 x 500 x 500 space, an internal state that replicates what it sees, almost real
time. If something does not work, assume the approach or the understanding is wrong, not Nethra.

## User decisions (standing)

- Never-seen positions or situations are not a concern (a real environment always has something to
  look at). Do not design for them or measure them.
- Constant new structure is not ideal. `source_similarity_threshold` (0.999) would ideally vary:
  finer where attention is, much coarser elsewhere. Not implemented, possibly not prudent yet. It
  may be varied freely for tests.

## What changed in the core this session (all on the branch)

| Commit | Change | Behaviour |
|---|---|---|
| c00938b | ML-sounding names renamed (`topology_and_evidence_change`, `_move_evidence_and_construct`, `outgoing_evidence_per_flow`, `incoming_evidence_per_tension`, ...) | identical; old checkpoints load |
| b7005ef | cosine structural recurrence of graded source patterns restored (was frozen on `nethra-smeared-source-frozen` .. `nethra-factorization-frozen`, dropped from the uploaded core) | changes construction for graded input (intended) |
| 890bf19 | frontier path scales with the frontier: lazy decay outside it, hot set carried over, lazy residual traces, pushes registered | bit-identical |
| dfc7522 | faster incidence compilation | bit-identical |

How execution-only changes were verified (redo this for any new one): keep the previous core
(`git show <commit>:nethra/nethra.py > ref.py`), feed identical streams to both, compare
`checkpoint_dict()` and every activation with `==`. Streams: 6 symbol Nethra (cycle plus random
extra pushes of 0.5/0.25), and a binocular retina (R=8, 160 intervals, one and two objects, fast and
slow). Modes: default, `source_support="product"`, frontier 1e-3 and 1e-2 with and without
`frontier_min`, `integrator="rk4"`, evidence change off. Checkpoint round trip mid-stream. One
numeric thread. `np.bincount` adds in array order, so vectorized sums can stay bit-identical.

## Harness

- `nethra/tests/binocular3d.py`: world, eyes, retina, reads. Two pinhole eyes at z = -500,
  x = 220 / 280; R x R receptive Nethra per eye with tent fields one grid spacing wide; each
  object pushes its graded shares (sum 1 per eye). Phases: left eye only, right eye only, both,
  two objects; control binocular from the start. Reads: lead, `ahead()` (P and activation at the
  next image point over the previous one), refound Nethra by phase built, `location()` (3D point
  of a constructed Nethra from the graded pattern the field indexed it by; exact triangulation),
  occlusion on frozen copies.
- `nethra/tests/binocular_path.py`: one object on a repeated circle (radius 150, 60 intervals per
  lap); per lap: built, P next/prev, activation next/prev vs a no-construction field, internal
  state distance, ms per step. Runs in 1-2 minutes. Start from this.

## Measurements (one seed, frontier 1e-2, wall clock, one thread)

| Setting | Result |
|---|---|
| Graded retina, no cosine recurrence, object moving 1 unit per interval, R=8 | one new Nethra every interval; Nethra k has one route S + {N1..N(k-1)} (nested chain); about 700 ms/interval averaged over 300 intervals (process time) |
| Same with cosine recurrence restored, 150 intervals | slow: 38 stored patterns, 75 constructed; about 2 per pattern (entering, staying) |
| `binocular_path.py 16 20` | built 57 in lap 1, 1 in lap 2, then 0; P next/prev (geometric mean) 1.66, 2.07, 2.39 (lap 5), 3.09 (10), 3.69 (20); activation next/prev 0.98 vs 0.91 without construction; internal state 71 units from the object after lap 1; 7.4-8.4 ms/interval |
| `binocular_path.py 32 15` | P next/prev 2.5-3.1 (no steady rise); activation 0.94 vs 0.76; internal state 24 units; 4.2-5.8 ms/interval |
| Monocular first, then binocular (R=8, tiny trace) | each binocular interval refinds the left-eye Nethra touching the current left cells; new binocular Nethra's routes contain them |
| Two objects on small circles (steps 5-6 units, grid spacing 33-67 units) | P 3-ahead/3-behind 0.8-1.1; inconclusive: steps smaller than a cell. Redo with steps >= one grid spacing and images >= 3 spacings apart |
| Warm R=16 field, ~1000 constructed, frontier 1e-2 | 86 ms/interval. Time per interval: `_physical_incidences` 21 ms, `_compile_interval` 20, derivative 16, `_move_evidence_and_construct` 14 + `update_residuals` 13, `_neighbors_from_edges` 7.5. Python loops over ~3,000-4,000 frontier incidences |

Read facts established:
- Live activation right after an interval is dominated by leftover from t-1 (about 37% of its push).
  P shows what structure carries toward the next step; use it for "ahead of the object".
- The internal state (refound constructed Nethra) resolves position to about one retina cell,
  because closure refinds every Nethra whose route cells are all active.

## Next steps (the user picks; do not start one unasked)

1. Two objects, small factors first, paths with steps of at least one grid spacing and images at
   least 3 spacings apart: does P run ahead for both, and do the single-object Nethra carry it?
2. Vary `source_similarity_threshold` in tests (allowed): construction and P on `binocular_path.py`.
3. Array-based per-incidence execution (roadmap item 7), verified bit-identical as above.

## Mistakes made this session (do not repeat)

- Fed both eyes and full 3D motion from the start: skipped small factors (`CLAUDE.md` section 4).
- Launched hour-long runs with no prediction; they could only repeat what 300 intervals showed.
- Edited a script while a run was using it; the run's output could not answer the question.
- `pgrep -f` wait loops matched their own shell and never ended; `pkill`-style kill hit own shell.
- Averaged ratios arithmetically (3.7 reported as 9).
- A scratchpad file named `nethra.py` shadowed the real core on the import path.
