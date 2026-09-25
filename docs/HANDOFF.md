# Handoff: binocular Nethra, focus and consequence (state as of 2026-09-25, fifth session)

## START HERE

Read in this order: `CLAUDE.md`, `nethra/NETHRA_OPERATING_NOTES.md`, this section, then §0g (fifth
session: direction, action loop), §0f and §0e (fourth session), §0-§0d (third session), then
`nethra/nethra.py`. Older sections (§1-§10) are the second session's state and are still valid unless
a §0 section says otherwise.

**Sixth session (§0h):** branch `claude/modest-bohr-avn2ng` = the fifth session plus operator tags and a drain
prototype (not core), and flipped channels (§0i, prototype). Core unchanged.

**Branch:** `claude/pensive-davinci-zla3pj` (Dooces/nethra-cuda-runner). It contains everything from
`claude/tender-cray-bk3jq6` (fourth session) plus the fifth session. No PR opened. Scratchpad files are
gone after a session; every script used is in `nethra/tests/`.

### Fifth session in one paragraph

User: "make progress on direction and timing and develop an action loop; make sure the field and
Nethra do the work, use ablation tests". New core option `direction="split"` (default stays
`"shared"`, bit-identical to before): each incidence gets two conductances, one per flow direction,
each moved by its own term of the existing evidence change, so timing of flow before manifestation
decides which direction strengthens. On moving objects it makes the field's own next step move
forward (ring: 24/24 intervals vs 8/24 shared; shuffled stream: chance). New action loop
(`gaze_loop.py`): an innate lagged reflex drives the eye while the field watches; then the reflex is
switched off and an actuator reads only the motor Nethra's activation, which nothing pushes any more.
With split direction the field-driven eye follows a bouncing object better than its teacher and
roughly halves its teacher's reversal lag (people reverse before the target after 1-2 cycles;
the field does not reach that); lesioned structure, shuffled training or no eye-position input
remove it; shared vs split in the loop is not consistent (0g.3). Split direction costs context capacity
(`cue_capacity` 15 regimes 0.93 -> 0.59, `robust` clean 0.83 -> 0.58): not made default. §0g.

### What changed in the core in the fourth session

- **Latest (§0f): `conduction="top_and_leaves"` is the default.** Covered constructed members earn no
  evidence (as top-only), primitive members always conduct. Chosen over top-only after comparing 8
  rules on the user scripts: it passes the context and cue tests top-only fails, keeps blocking,
  interference and 15-regime results at or above full conduction, at a third of full conduction's
  cost (4-5x top-only's). `"top"` and `"all"` stay available, bit-identical to before;
  `top_only_conduction=True/False` still works; old checkpoints load with `"all"`, the fourth
  session's with what they stored. Earlier in the session (below) top-only was the default.
- `top_only_conduction=True` (default, **provisional**, user decision): the `TopField` prototype moved
  into the core. A route member that lies in a complete route of another member of the same route is
  covered: its incidence earns no evidence (g = 0). Routes stay whole, so closure and construction
  are unchanged. Covered incidences are checkpointed. `False` = previous core, bit-identical; old
  checkpoints load with `False`. `True` = the prototype, bit-identical. **Measured regressions: §0e.2.**
- Execution only, bit-identical (checkpoints, activations and conducting incidences equal on
  symbolic and graded streams; default, frontier, frontier_min, rk4 + 0.9, whole joining, no
  evidence change; top-only on and off; checkpoint round trip; two-object binocular stream; user
  scripts `context_partwise.py`, `human.py` give identical output): per-relation incidence plan,
  cached pattern norms, vectorized interval compile and evidence-change arithmetic, closure
  route-use index, frontier halo cache. Numbers §0e.5.

### State of the user's goals

| goal | state |
|---|---|
| several objects without combination growth | solved for construction (§0) |
| cost must not keep rising | top-only now in core: flat on two objects (§0b). Execution now 12.3 ms/interval on the two-object stream (was 27.0); ETD integration is 45% of that (§0e.5) |
| one object tracked in detail, rest rough | input design tested (§0c); reads not good yet (§0d) |
| expectation of where things go next | **cause found on a tiny exact loop (§0e.1)**: structure carries next exactly; the field carries it weakly (symmetric conduction, hops at g ≈ 0.2 against leak 1), and the P read cannot point at cells that are already active. Delta input makes the structural next exact (§0e.3) |
| CUDA on the Fedora runner | unchanged (§0b) |

### Decisions waiting for the user

0. **Superseded by §0f:** the user asked for a better rule than top-only; `top_and_leaves` is now the
   default. Remaining for the user: accept its cost (30 -> 47 ms/interval on two objects over 400
   intervals vs 7 -> 10 top-only, 55 -> 145 full), and delta input is not adopted (§0f.3).
1. (Earlier, answered) **Top-only stays default?** It fails the context tests: the continuation built second is 2 hops
   further than the first, and context doesn't select it (`context_partwise.py`: C2+X primes Y 0.042
   vs Z 0.0014; `cue_capacity.py`: cue 4 still selects B). Also loses blocking and changes
   interference and spacing (§0e.2). Cost: flat vs rising (§0b). `top_only_conduction=False` restores
   the previous core.
2. Leakage: measured on seven streams (§0e.4). No setting better everywhere; leakage 2 helped
   `cue_capacity` 15 regimes (0.70 → 0.93) and hurt magnitudes elsewhere (P and activation 10-100x
   smaller at 4). Kept at 1.
3. Delta input (§0e.3): makes the structural next exact with one Nethra per position; with coarse
   position cells it helps at one speed and not at three. Whether to use it is an input-design
   decision (CLAUDE.md §4).
4. `_admit_by_parts` direction-blind accounting (§0e.3): a Nethra whose after route is complete on
   the before side and before route on the after side accounts for the reverse transition. Observed,
   not changed.

### Suggested next steps (do not start unasked)

1. Decide 1 above. If top-only stays, the context failure needs a look (open problem, observed
   failure): the second continuation is reached through the first one's Nethra.
2. Where next: the structural read (after routes of Nethra refound by their before route) works on
   exact loops (§0e.1, §0e.3); the fovea stream's structural read was worse than staying put (§0d).
   Find which of the differences (cosine 0.8 substitution, jitter, several objects) breaks it, on a
   stream one step bigger than the ring.
3. Execution: remaining cost is ETD integration (exact); anything faster changes rounding. The
   evidence dicts themselves are still dicts (writes 1-2 ms/interval); moving them to arrays is the
   rest of roadmap item 7.

### How to run things

- Local container: 4 cores, `pip install numpy` if missing. One numeric thread, `timeout` on every
  run, runs under 2-3 minutes.
- Fourth-session scripts (all in `nethra/tests/`, each under a minute):

  | script | what |
  |---|---|
  | `ring_symbolic.py` | env K, LAPS, TOP, LEAK, V=1 prints conductances. One Nethra per ring position; closure, structural next, P toward p(k-2..k+2) |
  | `ring_graded.py` | env L, NC, STEP, LAPS, TOP, LEAK. Tent cells on a ring; structural next and P as a point vs staying put |
  | `ring_split.py` | same env; share of P behind / at / ahead; input-free continuation |
  | `ring_trace.py` | same env + X: routes, integrals and every conducting incidence's flow at position X |
  | `delta_input.py` | env L, NC, DELTA, ND, VMAX, SPEEDS, PASSES, TOP, LEAK, TH, TOL, DUMP. Bouncing object with displacement Nethra |
  | `delta_runlength.py` | env as delta_input (ND=3, VMAX=1 default): d+ activation along a run; Nethra refound across a run |
  | `context_trace.py` | env TOP: `context_partwise.py` stream; conducting incidences of X, Y, Z, C1, C2 and every constructed Nethra |
  | `with_params.py` | `LEAK=2 TOP=0 python3 with_params.py script.py args`: runs a user script with other defaults |
  | `bitcheck_cores.py` | `bitcheck_cores.py OLD.py NEW.py`: bit-identity of two core files (see §0e.5) |
  | `exec_stream.py`, `exec_prof.py`, `exec_parts.py` | two-object checkpoint (`exec_stream.py 120 80 out.json`, env TOP), then ms/interval + hashes (`exec_prof.py CORE.py out.json [profile]`), ms by part (`exec_parts.py CORE.py out.json`) |

- Third-session scripts and runner: see §0-§0d and the old "How to run things" notes in §4 and §8.
  Example runtimes there were for the core before top-only; most are faster now.
- Bit-identity check pattern: keep the old core (`git show <commit>:nethra/nethra.py > old.py`),
  `bitcheck_cores.py old.py nethra/nethra.py`, plus `exec_prof.py` on both for the two-object stream.
- Known: `source_support="product"` and `"min"` give a different checkpoint on every run (pre-existing);
  the adaptive `_tol` (frontier_min) is not checkpointed, so a round trip in that mode diverges.

### Third session's start-here (kept for reference; its top-only and runner notes still apply)

#### What changed in the core this session

- `join_on_recurrence=True` (default) in `nethra/nethra.py`: a transition of pushed Nethra seen for
  the first time is subtracted per part (`_admit_by_parts`); only the unaccounted remainder is
  joined; a transition that recurs is joined whole (`_admit_whole_support`). `False` = previous
  core, bit-identical; old checkpoints load with `False`; `witnessed_transitions` is checkpointed.
  Details and measurements: §0.
- Nothing else in the core. `_admit_graded` gained a `before_description` argument (refactor only).

#### Prototypes and harness added (not core)

| file | what |
|---|---|
| `nethra/tests/top_conduction_prototype.py` | `TopField`: routes stay whole for closure; at construction only the top members of a route earn incidence evidence (covered leaves stay at g = 0); frontier halo through conducting incidences. `GpuTopField` = same + GPU integration. §0b |
| `nethra/tests/gpu_field.py` | `GpuField` / `GpuMixin`: ETD integration on the GPU (cupy); `NETHRA_GPU=0` = numpy, bit-identical to CPU ETD. §0b |
| `nethra/tests/gpu_bench.py` + `.github/workflows/nethra-gpu-bench.yml` | runner benchmark: CPU vs GPU, core vs top-only. Triggered by a push to this branch touching those files, or manually. §0b |
| `nethra/tests/focus_fovea.py` | focus test: coarse periphery, fine gaze-centred fovea, gaze Nethra, pursuit device; env options for fovea mode, gaze grid, leakage; reads: P centroid, share of P toward previous/current/next cells, structural next. §0c, §0d |
| `nethra/tests/binocular_multi.py` | now also `WHOLE=1` (join_on_recurrence False) and `PER2` (second loop period) |
| `nethra/tests/partwise_prototype.py` | pinned to `join_on_recurrence=False` so its earlier numbers reproduce |

#### State of the user's goals (§1)

| goal | state |
|---|---|
| several objects without combination growth | solved for construction (§0): 24 built per 120 together instead of 120; eyes and contexts still joined |
| cost must not keep rising | not solved in the core. Top-only conduction (prototype) makes it flat on the two-object stream (13 → 19 ms over 400 intervals, §0b); with the fovea setup it still rises (§0c, §0d) |
| one object tracked in detail, rest rough | input design built and tested (§0c); cost and expectation reads not good yet (§0d) |
| expectation of where things go next | **open, main problem**: neither P nor a structural read gives a next position better than "staying put" (§0d) |
| CUDA on the Fedora runner | works (RTX 5070, cupy); only useful for the current core's exact integration; with top-only, CPU is faster (§0b). Bottleneck is per-incidence Python bookkeeping |

#### Decisions waiting for the user

1. Top-only conduction into the core? (changes which incidences conduct; CLAUDE.md §2.1). Gains:
   flat cost, sharper context. (Fourth session: context is not sharper, it selects the first-built continuation, §0e.2.) Losses: consequence reach through shared leaf cells, focus tilt (§0b).
2. Leakage (drain rate) other than 1? Leakage 4 halves the frontier in the fovea stream (§0d). The
   user asked about charge vs drain; it is a parameter (notes §4, CLAUDE.md §2.3) and also shifts the
   admission-seed balance and the M/P scale. Only frontier/cost were measured, not construction
   quality or the other streams.
3. Input design for focus (change-signalling cells, grids): no option tried so far helped (§0d).

#### Suggested next steps (in order; do not start unasked)

1. **Why the field carries no usable "where next"** (§0d). Measured: P goes ~equally to previous and
   next cells and 70-80% to cells of other loop positions (the object's whole structure is lit and
   conduction is symmetric). The direction exists only in before/after routes, but reading after
   routes of Nethra refound by their before route is also worse than staying put. Look first at what
   the before routes contain (they are the whole closure of the interval, including refound
   constructed Nethra), and at which Nethra are refound by their before route at each interval, on a
   tiny stream (one object, one exact loop, no jitter) where the right answer is known.
2. Array storage of incidences (roadmap item 7), bit-identical: the remaining cost is per-incidence
   Python work (evidence change, incidence compilation, closure, pattern cosine), not integration.
3. **User's idea to consider: encode deltas.** If Nethra received displacement (the change of
   position between consecutive intervals) as its own graded input, construction could come to
   carry "how long in a direction": a steady motion pushes the same displacement Nethra interval
   after interval, and the chain of intervals with the same displacement is what closure refinds.
   The numbers already exist in every stream (positions at t-1 and t). Notes for whoever picks it up:
   - Feed it like any graded value (notes §3): receptive Nethra over displacement per axis (or
     direction × speed), each pushed with its share. Negative displacement is a region of the
     receptive range, not a negative push (notes §2 rejects signed presence-change events and -1
     pushes; this is a quantity, not a presence change).
   - Different from what §0d tried: those change-signalling cells were pushed with the increase of
     one cell's input, not with displacement.
   - With lag-one pursuit the fovea offset already is the image displacement of the focused object;
     §0c/§0d show its pattern is nearly constant, which lit that object's whole structure. So
     test first how a constantly pushed displacement pattern behaves (holding-like presence,
     notes §7) and what closure refinds over a run of equal displacements, on a tiny stream.
   - Input design is the user's call (CLAUDE.md §4); predict first (CLAUDE.md §0).
4. Only after 1: fovea/gaze design again; multi-scale reach; Δt (§7.4, which is the same kind of
   graded input).

#### How to run things

- Local container: 4 cores, `pip install numpy` if missing. One numeric thread, `timeout` on every
  run, runs under 2-3 minutes. Examples: `binocular_multi.py 16 0.8 10 3 160 40` (~25 s);
  `focus_fovea.py top 0.01 2 80 40` (~40 s); `context_partwise.py`, `consequence_reach.py 8 0.95 10 40 10 0`,
  `focus_symbolic.py` (their numbers are in §0-§0b).
- To run a prototype field in a script written for the core: replace `core.NethraField` before the
  script runs (see how `gpu_bench.py` picks classes); capture the base class first to avoid
  recursion.
- Fedora runner (`runs-on: self-hosted`, runner name `fedora`, Python 3.14 at `/usr/bin/python3`,
  numpy, cupy, numba, torch, RTX 5070): add a workflow whose `on: push: paths:` matches the files you
  push, or `workflow_dispatch`; read results from the job log (GitHub MCP `get_job_logs`) or the
  uploaded artifact. Pattern: `.github/workflows/nethra-gpu-bench.yml`.
- Bit-identity check pattern (used for `join_on_recurrence`): keep the old core
  (`git show <commit>:nethra/nethra.py`), import it from a separate directory, feed identical
  streams, compare a hash of `checkpoint_dict()` and of all activations; include frontier, rk4 and a
  checkpoint round trip mid-stream.
- Known: `source_support="product"` gives a different checkpoint on every run (pre-existing).

## 0i. Sixth session, part 2: flipped channels (prototype, not core)

User's idea: each directed channel between Nethra can be individually reversed: instead of N
sourcing m, N drains m ("a makes b unlikely", "something happens when something does not").
User's answers: drained charge goes into the draining Nethra (yes); where flips come from and
whether closure uses them: test.

### 0i.1 What was built (`tests/flip_prototype.py`, `tests/flip_streams.py`)

- `direction="split"` (two channels per incidence). A flipped channel N -| m pulls `h max(0, a_N)`
  out of m into N while m has charge (at most m's charge per substep plus its inflow; m never below
  zero, lowest seen -4e-5 from that limiter). h = conductance(e), h(0) = 0. Not a route member.
- Evidence cannot cross zero on an existing channel: its change is proportional to its own flow,
  which vanishes with g (the admission-seed fixed point, notes §4). So the sign is set at
  construction: each handle of the transition gets as flipped members what was expected and did
  not come. `expect="structure"`: after-route members of Nethra refound by their first route in
  the before side; `"residual"`: Nethra with M - P < 0. New flips start at admission_seed; inert
  ones found again are re-seeded (as the core re-seeds inert routes). `flip_to`: "all" or "leaves".
- Residual at m is net of the pulls (M + Q - P) for the core's evidence change too;
  `delta e(N -| m) = -outgoing_evidence_per_flow * q * r_m`.
- `closure_mode="field"`: flips are field only. `"absent"`: a Nethra whose flipped member is in the
  positive closure is blocked, closure recomputed once.
- Off: bit-identical to the core with direction="split", integrator="rk4" (both closure modes).

### 0i.2 Streams (300 trials, seed 0; baseline is split without flips; outcomes never silent)

| stream, read (last block) | off (split) | structure field all / leaves | structure absent all / leaves | residual field all / leaves | residual absent all / leaves |
|---|---|---|---|---|---|
| A->X, AB->Y, C->X: X after AB/A; CB/C | 1.10; 1.05 | 1.16 / 1.13; 1.07 / 1.06 | 1.22 / 1.09; 1.10 / 1.04 | 1.15 / 1.13; 1.06 / 1.05 | 1.19 / 1.09; 1.16 / 1.04 |
| A->X, B->X, AB->Y: X after AB/A | 1.90 | 1.84 / 1.90 | 1.58 / 1.90 | 1.85 / 1.88 | 1.68 / 1.88 |
| operator, trained share c(x+1) O+ / O- | 0.562 / 0.572 | 0.473 / 0.444; 0.557 / 0.537 | 0.564 / 0.507; 0.561 / 0.533 | 0.512 / 0.537; 0.520 / 0.537 | 0.595 / 0.575; 0.529 / 0.527 |
| operator, held-out c3 with O+ (off 0.369) | | 0.133 / 0.352 | 0.125 / 0.376 | 0.208 / 0.341 | 0.259 / 0.366 |
| X->B 75%: share B | 0.532 | 0.530 / 0.535 | 0.512 / 0.535 | 0.506 / 0.504 | 0.585 / 0.504 |
| bounce: share ahead, right / left | 0.398 / 0.422 | 0.400 / 0.424; 0.401 / 0.422 | same as field | 0.392 / 0.439; 0.394 / 0.430 | same as field |
| extinction: X after A end of A->X / end of A->Y / 20 into A->X again | 0.0197 / 0.0158 / 0.0162 | 0.0170 / 0.0086 / 0.0166; 0.0191 / 0.0083 / 0.0163 | 0.0112 / 0.0081 / 0.0086; leaves = field | 0.0190 / 0.0122 / 0.0167; 0.0189 / 0.0121 / 0.0163 | 0.0138 / 0.0071 / 0.0103; leaves = field |

Flips at the end: 11-5,077 (operator structure all 4,650); max h 0.08-0.2 except extinction (0.83).
Cost up to 2.5x off (field) and 6.5x (absent, all: operator 13 -> 84 ms/interval).

- Only extinction changed clearly: during A -> Y the A -> Y handle drains X (0.0158 -> 0.0083-0.0086,
  structure). Back on A -> X, 20 trials reach 0.0163-0.0166 (86-89% of the level before, off 97%
  but off barely went down); first learning at 20 trials was 0.0102 (leaves), 0.0040 (all) vs off
  0.0151: early filler flips slow first learning.
- AB-only suppression did not form. Traced (`negfeature`, structure, leaves): evidence on the AB->Y
  Nethra's flip onto X moved +0.61 in total at the Y intervals and -14.77 at intervals where X came
  (A->X and C->X trials); ends at 0 and is re-seeded at every AB->Y. Cause: A alone conducts into the
  AB Nethra (primitive members conduct without closure), so the flip pulls during A trials, and when
  X comes its residual (~0.2) is about 10x the over-carry the flip cancels in AB trials (~0.03).
- Early on (30 trials, structure, all) flips from random filler transitions landed on A and on the
  A -> X Nethra: X after A 0.0006 (off ~0.003); they wore off by 150-200 trials (X comes, they weaken).
- `absent` with `all` removes structure broadly: held-out O+ 0.125, relearning after extinction
  stays at 0.0086, AB/B in negpattern 2.66. With `leaves`, `absent` equals `field` on most streams.
- Bounce: flips barely formed (h ~0); no change.
- Stability: nothing negative beyond the limiter's -4e-5; no oscillation seen.

### 0i.3 Decisions waiting for the user

1. The flip is erased because the draining Nethra is partly active without its route complete.
   Options: pull only while the draining Nethra is refound (a closure gate on the flip: the
   rejected "recognition gates conduction" class, but on a new channel); or evidence change on flips
   weighted differently when the target comes (asymmetric rates: a parameter). Not chosen.
2. Structure-expectation flips from unpredictable transitions (random fillers) are numerous
   (hundreds to thousands) and temporarily harm what they touch. `leaves` bounds the count 3-6x.

## 0h. Sixth session: operator tags, and drain incidences (prototype, not core)

User: "is there a straightforward way to use the field to abstract operators"; then "what if Nethra
of Nethra or their leaves could have a negation instead of excitation, or drain instead of source";
then "check on many tiny streams, a few hundred cycles each, to see what settles".

### 0h.1 Operator tags (no core change)

Stream: 6 symbols cycled forward 8 laps, then trials [c_i, O+] -> c(i+1), [c_i, O-] -> c(i-1),
O+ held out on c3. Share of P toward c(x+1) against c(x-1), frozen copy:

| operand | tag | shared, 6 / 20 per (operand, op) | split, 6 |
|---|---|---|---|
| trained c1 | O+ / O- / none | 0.49-0.51 / 0.51-0.52 / 0.49-0.52; 20: 0.495 / 0.494 / 0.488 | 0.67-0.68 / 0.70-0.71 / 0.78 |
| held-out c3 | O+ / none | 0.42-0.43 / 0.38-0.39; 20: 0.34 / 0.33 | 0.58-0.61 / 0.68-0.71 |

- The tag does not select even on trained operands. The Nethra {c1, O+, ...} | {c2, ...} is built
  but stays at seed-level g (~0.2); tag-free sequence Nethra carry the flow (the §0f.2 weak spot).
- A tag is a hub (as d+/d- in §0f.3). Closure is membership only; nothing is shared across
  operands but the tag, so no transfer to a held-out operand.
- Per trial, M - P at the neighbour that did not come is -0.03 (negative in 72/72 trials); at the
  result +0.16 to +0.24. Evidence change clamps at 0, so this is discarded today.

### 0h.2 Drain incidences (`tests/drain_prototype.py`, `tests/drain_streams.py`)

- Negative g in the conduction term is anti-diffusion (a pair's difference grows once g < -leak/2):
  not used. Law used: s -| m adds `- h_sm max(0, a_s) a_m` to leaf m (shunt). It never pushes, never
  makes m negative, adds charge nowhere; bilinear, so RK4. Targets are leaves only; closure and
  construction are unchanged (drains are field only, not route members).
- Drain evidence: `delta d_sm = rate A_s A_m ((P_m - D_m) - M_m)`, clamped at 0, s != m, with
  D_m the charge the drains took from m (product of interval integrals: declared approximation).
  The drain's own effect is counted in what the field carried, so it stops growing once it cancels
  the over-carry. Rate = outgoing_evidence_per_flow. h = the conductance map (h(0) = 0), ceiling
  g_max. Sources: pushed Nethra of the prior interval (`leaves`), constructed Nethra refound in it
  (`built`), or `both`.
- Drain off is bit-identical to the core with integrator="rk4" (checkpoint hash, 200 intervals).
- Streams (300 trials each, probed every 50; predictions in the `drain_streams.py` docstring):
  negfeature (A -> X, AB -> silent, C -> X), negpattern (A -> X, B -> X, AB -> silent), operator
  (0h.1), prob (X -> B 75% / C 25%), ring (6 positions, 50 laps), extinction (A -> X 100, A -> silent
  100, A -> X 100). Seed 0. Last block:

| stream, read | off | leaves / built / both, stated rate | x100 rate (h at g_max) | x100 rate, ceiling 10 g_max (diagnostic) |
|---|---|---|---|---|
| negfeature X after AB / A | 1.16 | 1.14 / 1.16 / 1.13 | 1.05 / 1.12 / 1.03 | **0.62** / 1.06 / **0.63** |
| negfeature X after CB / C | 1.09 | 1.07 / 1.09 / 1.07 | 0.93 / 1.06 / 0.91 | **0.48** / 0.98 / **0.50** |
| negpattern X after AB / A | 1.85 | 1.83 / 1.85 / 1.84 | 1.76 / 1.80 / 1.75 | 1.61 / 1.70 / 1.64 |
| operator trained share O+ / O- | 0.493 / 0.481 | same to 3 decimals | 0.493 / 0.482 | 0.496 / 0.487 (leaves) |
| prob share B | 0.587 | 0.587 | 0.60-0.61 | 0.63-0.64 (0.64-0.68 at 250) |
| ring share ahead | 0.500 | 0.501 | 0.502-0.504 | 0.507-0.513 |
| extinction X after A, end of extinction (start 0.019) | 0.016 | 0.016 | 0.011-0.014 | **0.002-0.004** |
| extinction, 20 trials into reacquisition / end | 0.018 / 0.020 | same | 0.016 / 0.019 | 0.011-0.013 / 0.016 |

- At the stated rate nothing changes (every read within 2% of off): drain evidence grows linearly
  and is far from settling (negfeature, B -| X: d 0 -> 21 over 100 AB trials; D 0.002 vs P 0.045).
- It cannot settle either: to cancel P = 0.045 needs h A_B A_X = 0.045, h ~ 7.5 against a ceiling
  of 1.5. A leaf's activation away from pushes is 0.01-0.02 and it loses charge through leak plus
  all its conductances, so a shunt capped at g_max removes at most ~10-15% of it.
- Only with the ceiling raised (diagnostic, not a proposal) does something emerge, and only from
  leaf sources: B inhibits X after A and after C (summation transfer, as in conditioned inhibition;
  Rescorla 1969, animals); extinction goes near zero and relearns at about the first-learning speed
  (no savings; people and animals show savings and spontaneous recovery, Pavlov 1927, Bouton 2004);
  probability shares sharpen a little, no winner.
- Constructed-Nethra sources stay weak at every setting: a constructed Nethra's activation comes only
  by conduction (§0e.1) and is small, so both its drain learning (A_s) and its drain (a_s) are small.
  So configural inhibition (negpattern, operator selection) did not appear.
- Stability: lowest activation 0 in every run; settled blocks equal to 4 digits (ring); no
  oscillation. Shunting only adds dissipation. Cost up to 2.5x off (operator 10 -> 27 ms, x100 rate, ceiling 10).
- Side effect: with strong drains construction changed (extinction 31 -> 39 Nethra): drains change
  M, and M - P feeds construction.
- Closure-level negation (routes with a negated member) was not built. Negation on leaves is
  evaluated against the pushed pattern; negation on constructed Nethra would be evaluated once
  against the finished positive closure, so it cannot feed back into itself (same settled rule as
  "a description containing itself is skipped", old ledger 2026-09-23).

### 0h.3 Decisions waiting for the user

1. Drain ceiling: a shunt at g_max is too weak to matter. Options: its own ceiling (a parameter),
   a drain sized relative to the target's own outflow (leak + sum g), or a sink form (charge removed
   at a rate set by s, stopping at zero). Not chosen.
2. Configural drains (from constructed Nethra) are weak because constructed Nethra carry little
   activation; same root as the conjunction weak spot (§0f.2, §0h.1).

## 0g. Fifth session: direction from timing, and an action loop

### 0g.1 Core option `direction="split"`

- Field law with `"split"`: incidence (relation R, member m) has g(R->m) and g(m->R).
  Flow j -> i = g_ji max(0, a_j - a_i). With g_ji = g_ij this is exactly the shared law.
  Both start at the admission seed (route registration, reseeding, graded side support).
- Evidence: prior flow from the completed interval runs R -> m when A_R > A_m (through g(R->m)) and
  m -> R otherwise (through g(m->R)). The outgoing term (flow into m, then m manifests or not)
  moves g(R->m); the incoming term (tension shared among members that fed R) moves g(m->R). Same
  formulas and rates as before; nothing reads before/after routes.
- Covered incidences (conduction rule) zero both. Route summary = max over both directions.
- Integration is RK4 in split mode (ETD needs a linear passive operator). About 2x slower on
  `consequence_reach` (52.6 vs 26.2 ms/interval), 4x on `cue_capacity`.
- Checkpoints store `incidence_evidence_in` and the parameter only in split mode; shared checkpoints
  are byte-identical to before.

Checks:
- `bitcheck_cores.py` old core vs new core (default `"shared"`): ALL IDENTICAL (symbolic and graded,
  2 seeds, default / frontier / frontier_min / rk4+0.9 / whole joining / evidence change off, top
  on/off, round trip).
- Split with evidence change off = shared with RK4, activations identical on a 100-interval stream.
- Split checkpoint round trip mid-stream: activations and checkpoint identical.

### 0g.2 Direction on moving objects (field reads only)

Symbolic ring K=6, one Nethra per position (`direction_ring.py`). Predicted from the code: forward
incidences (p_k -> N_(k+1), N_(k+1) -> p_(k+1)) strengthen, their reverse stays at the seed. Seen:

| incidence of N_(k+1) = {N_k, p_k} \| {p_(k+1)} | g relation->member / member->relation |
|---|---|
| p_k (before member) | 0.22 / 1.03 |
| p_(k+1) (after member) | 0.75 / 0.21 |
| shared, same incidences | 0.99 / 0.99 and 0.68 / 0.68 |

Free step on a copy (nothing pushed), gain over leakage-only decay, last lap:

| | gain at p(k+1) | gain at p(k-1) |
|---|---|---|
| shared | +0.016 | +0.008 (spills backward) |
| split | +0.019 | -0.002 |

Graded ring, 12 positions, 8 tent cells, step 1, 20 laps (`direction_motion.py`); the live
activation centre after a free step, relative to the centre now, along the true motion:

| | centre shift (true step 1.00) | intervals shifting forward | gain ahead / behind |
|---|---|---|---|
| shared | -0.086 | 8/24 | 0.59 / 0.41 |
| split | **+0.207** | **24/24** | 0.81 / 0.19 |
| shared, positions shuffled in time | +0.101 | 13/24 | 0.48 / 0.52 |
| split, positions shuffled in time | +0.092 | 13/24 | 0.49 / 0.51 |

- The positive gain peaks 2 cell spacings ahead: the next cell is already active, so it gains
  nothing (as in §0e.1). The centre shift is the read that shows the step.
- Bouncing object (`STREAM=BOUNCE`), graded: split does not help (16/16 cells: forward in 4/30
  intervals, shared 5/30). On a line visited both ways the pushed position is the same for both
  directions; direction has to come from what came before. Symbolic bounce
  (`direction_bounce.py`, L=5): gain(next) > gain(behind) in 6/6 intervals in both modes; split
  learns directions per incidence from use, so Nethra built for rightward steps are also used for
  leftward ones (e.g. N2 = {N1,p1} | {p2}: g(p2->N2) 0.82).

### 0g.3 Action loop (`gaze_loop.py`)

World: object bounces on 0..L-1 at speed 1. Eye at e (clamped). Nethra: retina (one per offset
o = x - e, |o| <= R), motor command mL / mR, eye position (one per e). Each interval: push retina
o, eye position e and, while the reflex drives, the motor command m; step; then e += m.

- Learning (300 intervals): innate reflex with latency LAT, m_t = sign(o_(t-LAT)); motor Nethra
  pushed with the reflex's command.
- Test (80 intervals, same world state, evidence change off): reflex off; nothing pushes the motor
  Nethra; actuator m = sign(a_R - a_L) if |a_R - a_L| > 0.002. The actuator is identical in every
  condition.
- Why this design (measured, `gaze_retina_device.py`, `gaze_motor_probe.py`):
  1. An actuator that reads the retina already tracks with no structure at all (mean |o| 1.00 in
     every condition): the device did the work. Discarded.
  2. A motor Nethra that is pushed with the movement holds ~0.3 leftover; structure adds ~0.02-0.04
     to the other one. Reading it is momentum. So at test the motor Nethra is not pushed.

Results, LAT 2, R 4, split (share |o| <= 1; "with object" = moving intervals that move the object's
way; reversal lag = intervals from object reversal to eye reversal):

| L | teacher (reflex) | best fixed eye | **trained, reflex off** | lesion | shuffled training |
|---|---|---|---|---|---|
| 8 | 3.33 | 1.82 | 2.02, with object 0.70, lags 0-4 | 2.79, moves 0.06 | 1.79, moves 0.15 |
| 10 | 3.70, lag 4 | 2.25 | **2.15**, near 0.39, with object 0.76, lags 3,0,4,2,1,0,3,2 | 4.64, moves 0.06 | 3.76, moves 0.01 |
| 12 | 3.88 | 2.67 | 2.62, with object 0.81 | 5.79 | 3.51, moves 0.01 |
| 14 (R 5) | 4.03 | 3.30 | **2.16**, with object 0.83, lags 3,2,1,0,0 | 6.35 | 4.14 |

Ablations at L=10 (mean |o|, trained):

| | result |
|---|---|
| shared direction | 4.62 (moves 0.79, with object 0.48: drives into a wall); but 1.94 after 288 intervals (see exposure table) |
| no eye-position Nethra (`PROP=0`), split | 3.79 (reversal lags 2-15) |
| no construction during learning | 4.64, moves 0.09 |
| LAT 1 (teacher 2.58) | split 2.86, shared 4.67 |
| LAT 3 (teacher 4.35) | split 1.88, near 0.54, with object 0.88, lags 0,3,0,2,0,2,0,2,0 |
| 600 learning intervals | split 1.98 |
| evidence change on at test | split 1.96, shared 3.01 |
| actuator threshold 0.0005 / 0.01 | split 2.04 / 2.39 (moves 0.06 at 0.01) |

- What the field does here: learns from the reflex which motor command follows which retina and
  eye-position pattern, and drives it with the reflex gone. It acts on the latest pattern, not the
  one the reflex used, so it is earlier than its teacher; reversal lag 0 means the eye turned in the
  same interval as the object (it uses the eye-position Nethra: without them the lags are 2-15).
- What it does not do: it is not better than the best fixed eye at L=8. Shuffled training
  sometimes moves (L=8: 1.79 with 15% moving intervals, parked near the middle).
- One run per configuration (deterministic world); the configurations are the replications.

**Human reference (user rule, CLAUDE.md §2.10: judge against what people do, not an ideal).**
People: smooth pursuit starts ~100-130 ms after target motion, saccades ~200-250 ms; on a periodic
target the eye often begins to reverse before the target after one or two cycles (anticipation of
the reversal), with residual position error and catch-up saccades. (Sources from search:
Barnes & Asselman 1991, J Physiol, "The mechanism of prediction in human smooth pursuit"; J Neurosci
29(42):13302, 2009; Frontiers Syst Neurosci 2013, 7:4. Full texts were not reachable from the
container; numbers are from their abstracts/snippets.) The lagged reflex (LAT 2, reversal lag 4)
stands for the unpracticed latency.

Exposure needed (L=10, LAT 2, split; test 72 intervals, reflex off; one cycle = 18 intervals):

| learning cycles | 1 | 2 | 3 | 5 | 8 | 16 |
|---|---|---|---|---|---|---|
| split: eye reversal lags | 3,4,1,3,3,4,1,3 | 2,4,1,2,6,6,3,2 | 2,5,4,2,2,5,4,2 | 2,4,2,2,3,4,2,2 | 2,4,2,4,2,4,2,4 | 2,2,3,2,1,2,3,2 |
| split: mean abs(o) | 2.74 | 2.86 | 2.11 | 2.03 | 1.99 | 2.10 |
| shared: mean abs(o) | 2.71 | 2.67 | 3.18 | 3.14 | 3.58 | **1.94** |

- Against people: the field roughly halves its teacher's reversal lag (4 -> ~2) from the first
  cycles on, but it does not reverse before the object as people do after 1-2 cycles; lag 0 shows
  up only in some runs (300 intervals: 2 of 8 reversals). Below the human reference.
- **Correction:** shared direction is not a consistent failure in this loop: 4.62 at 300 learning
  intervals but 1.94 at 288. The split vs shared difference in the action loop depends on where
  learning stops; not established.

### 0g.4 Split direction on the user scripts (`DIR=split python3 with_params.py ...`)

| script | shared (default) | split |
|---|---|---|
| `cue_capacity` 8 regimes 11/39/61%; 15 regimes | 1.00/1.00/0.94; 0.93 | 1.00/0.99/0.86; **0.59** |
| same, cue2 C (right) / B; no cue B / C | 0.103/0.063; 0.051/0.070 | 0.104/0.059; 0.035/0.069 |
| `robust` clean / partial / noisy / noisy2 | 0.83/0.40/0.50/0.04 | **0.58**/0.38/0.44/0.04 |
| `human 14` blocking ratio | 0.18 | 0.25 |
| same, interference after A: B / C | 0.017 / 0.041 | 0.007 / 0.042 |
| same, spacing massed / spaced | 0.040 / 0.026 | 0.022 / **0.035** (spaced ahead) |
| same, XOR per 150 blocks | 0.51-0.75 | 0.38-0.53 |
| `context_partwise` small first, act right/wrong C1; C2 | .042/.039; .042/.040 | .044/.041; .045/.040 |
| same, P right/wrong C1; C2 | .047/.048; .047/.049 | .050/.053; .053/.052 |
| `focus_symbolic` share with / without F (P) | 0.484 / 0.480 | 0.409 / 0.487 |
| `consequence_reach` P toward F first k; at k=8; peak; act F from k | 8; 8.9e-4; 5.0e-2; 3 | 8; 5.6e-3; 6.7e-2; 5 |
| ms/interval (`consequence_reach`) | 26.2 | 52.6 |

Split carries sequences further ahead (consequence reach, ring, action loop) and loses on
co-present context (regimes, robust clean, XOR, focus). Not investigated why.

Against what people do (CLAUDE.md §2.10), qualitative direction only:
- Spacing: people retain spaced exposure better than massed at a delayed test. Split: after 200
  unrelated intervals spaced 0.0345 vs massed 0.0213 (people's direction); shared: massed 0.0392 vs
  spaced 0.0269 (opposite).
- Blocking, retroactive interference (recent answer dominates, old kept), combination-only (XOR)
  learned slower than simple: both modes show the direction people show.
- 15 overlapping regimes from 6 features, `robust` probes: no human data for these exact tasks
  known here; no human bar stated.

### 0g.5 Decisions waiting for the user

1. `direction="split"` stays an option, default `"shared"`. Adopt, keep as option, or look at the
   context loss first (what co-present context needs from the reverse direction)?
2. Action loop design: reflex-taught, motor Nethra not pushed at test. The alternative (motor
   pushed with the movement) is momentum by measurement (0g.3).

### 0g.6 Suggested next steps (do not start unasked)

1. Trace why split loses 15-regime selection (`cue_trace.py`-style on one regime pair): which
   member -> relation incidences of the context Nethra lose evidence.
2. Action loop: the field only repeats the reflex's mapping, earlier. For focus that goes to what is
   consequential, the loop needs something that makes some states matter (user's "want", §1.3).
3. Timing without place (how long a run lasts): not tested this session; §0e.3 showed structure
   cannot count a run of equal patterns.

### 0g.7 Scripts (`nethra/tests/`)

| script | what |
|---|---|
| `direction_ring.py` | env K, LAPS, ORDER: symbolic ring, both modes: directed g, P, free-step gain |
| `direction_motion.py` | env STREAM=RING/BOUNCE, L, NC, STEP, LAPS, SPEEDS, PASSES, DIR, SHUF: centre shift, gain ahead/behind, profile |
| `direction_bounce.py` | env L, PASSES, DIR, COND: symbolic bounce topology and free-step gain |
| `gaze_loop.py` | env L, R, DIR, LAT, TRAIN, TEST, DEAD, LEARN, PROP, CONDS: action loop (0g.3), ~10-20 s |
| `gaze_retina_device.py` | first loop design (actuator reads retina): tracks without structure, discarded |
| `gaze_motor_probe.py` | motor Nethra activation trained vs lesioned during learning (why the motor is not pushed at test) |
| `with_params.py` | now also env DIR=shared/split |

## 0f. Fourth session, part 2: a conduction rule better than top-only; delta input by field reads

User: "test top-only as default, and try delta input, though make sure you aren't just migrating
towards another framework and the field itself is doing the work"; then "you can try something other
than top-only as default, run tests and explain why it's clearly better, make it work".

### 0f.1 Why top-only fails (traced)

`cue_trace.py` (the `cue_capacity.py` cue stream: A -> B with cue1 40 times, then A -> C with cue2):
- N6 = {A, c1} | {B, c1} is built first, with nothing before it, so A, B, c1 conduct to it directly;
  evidence change saturates them (g 1.50).
- N10 = {A, N9, c2} | {C, c2}: its before route contains N9's after route {A, c2} (N9 = the previous
  transition, Z -> A+c2), so A and c2 are covered; N10 is driven only through N9 and its g stays at
  the seed (0.20). A + c2 reaches C via 1.5 -> 0.2 -> 0.2, B via 1.5 -> 1.5: B wins.
- In a continuous stream this always happens: a Nethra's before route is the previous interval's
  closure, which contains the previous transition Nethra's after route. So under top-only the
  present drives "what comes next" only through "how it got here", and whichever Nethra was built
  first on a member gets the direct incidences, saturates, and dominates (rich get richer).
  `context_partwise.py` (§0e.2) is the same mechanism.

### 0f.2 Rules compared (all decide once, at route registration, which members earn evidence;
routes stay whole, so closure and construction are identical in all of them)

| rule | members earning evidence |
|---|---|
| top | members not covered (previous default) |
| all | every member (the core before 2026-09-25) |
| **topleaves (now `top_and_leaves`, default)** | top members + every primitive member (no routes: pushed Nethra) |
| leaves | primitive members only (top rule when a route has none) |
| beforeall / afterall | every member on the before / after side, top rule on the other |
| leavesbefore / leavesafter | primitive members added on the before / after side only |

User scripts, leakage 1 (`with_conduction_variant.py`, prototype `conduction_variants.py`):

| test | top | all | afterall | beforeall | **topleaves** | leaves | leavesafter |
|---|---|---|---|---|---|---|---|
| `context_partwise` small first, activation right / wrong, C1; C2 | .043/.001; **.001/.042** | .041/.037; .041/.037 | .043/.021; .042/.022 | .041/.033; .049/.039 | .042/.039; .042/.040 | .037/.039; .041/.036 | .043/.021; .046/.022 |
| same, P right / wrong, C1; C2 | .043/.001; .001/.042 | .046/.042; .046/.042 | .044/.017; .044/.018 | .043/.043; .061/.042 | **.047/.048; .047/.049** | .042/.049; .048/.043 | .050/.020; .048/.021 |
| `cue_capacity` cue2: C (right) / B | .003/.043 | .101/.065 | .006/.045 | .088/.068 | **.103/.063** | .102/.059 | .005/.043 |
| same, no cue (recent C / old B) | .001/.003 | .069/.052 | .028/.004 | .009/.014 | **.070/.051** | .062/.007 | .030/.003 |
| 8 regimes 61% overlap; 15 regimes | .88; .70 | .94; .83 | .85; .12 | .91; 1.00 | **.94; .93** | .93; .80 | .84; .15 |
| `human.py 14` blocking ratio (low = blocking) | .89 | .38 | 1.00 | .98 | **.18** | 9.71 | .86 |
| interference after 150 unrelated, B / C | .001/.000 | .015/.036 | .001/.004 | .001/.004 | **.014/.037** | .001/.050 | .001/.004 |
| spacing, massed / spaced | .002/.046 | .037/.025 | .002/.032 | .003/.040 | .040/.026 | .038/.028 | .003/.033 |
| `robust.py` clean / partial / noisy / noisy2 | 1.00/.40/.58/.06 | .83/.38/.50/.04 | .92/.42/.54/.06 | .92/.35/.52/.06 | .83/.40/.50/.04 | .83/.40/.48/.04 | 1.00/.40/.58/.06 |
| `focus_symbolic` share with / without F | .489/.488 | .495/.482 | .456/.490 | .437/.473 | .484/.480 | .485/.479 | .459/.485 |
| `consequence_reach` P toward F: first k; peak | 6; 1.3e-2 | 8; 4.3e-2 | 6; 8.6e-3 | 8; 6.0e-2 | 8; 5.0e-2 | 8; 5.1e-2 | 6; 5.7e-3 |

Cost, two objects (`conduction_cost.py 0 400 x`: gpu_bench stream, 3 alone laps each, then 400
together, tolerance 0.01, local, one thread), ms/interval per 80 together; frontier; conducting
incidences at the end:

| rule | 0-79 ... 320-399 | frontier | conducting of 8,871 |
|---|---|---|---|
| top | 7, 8, 9, 10, 10 | 114 -> 136 | 968 |
| leavesafter | 16, 17, 21, 23, 24 | 176 -> 231 | 2,104 |
| leavesbefore | 24, 27, 28, 31, 38 | 198 -> 256 | 3,180 |
| **topleaves** | 30, 33, 40, 43, 47 | 216 -> 277 | 4,316 |
| beforeall | 39, 54, 66, 76, 87 | 216 -> 323 | 6,182 |
| all | 55, 74, 94, 112, 145 | 239 -> 358 | 8,871 |

Why topleaves (`top_and_leaves`):
- Every pushed Nethra conducts directly to every constructed Nethra whose route holds it, whatever
  was built before. The present drives the Nethra that expects the next step directly (fixes 0f.1).
  Constructed members inside another member's route stay covered (the duplication top-only removed
  among constructed Nethra).
- It is the only rule that passes every test full conduction passes: cue selection, recency,
  regimes (0.93 vs 0.83 full), blocking (0.18, stronger than full), interference, consequence reach.
- Cost: a third of full conduction and it rises at about the same relative rate as top-only (+57% vs
  +43% over 320 intervals; full +164%). It is 4-5x top-only.
- Rules that keep the before side covered (top, afterall, leavesafter) fail cue selection and
  15 regimes; rules that open the before side fully (beforeall) lose blocking and cost 2x.
- Weak spot: `context_partwise.py` small factors first. Activation right in both rows but by 7%;
  P slightly wrong in both (0.047 vs 0.048). The conjunction Nethra (C1+X -> Y, C2+X -> Z) conduct
  directly with C, X and Y/Z but stay at seed-level g (0.15-0.29) while the phase-1 X -> Y, X -> Z
  Nethra are at 1.5 (`context_trace.py`); full conduction has the same small margin (P 0.046 vs
  0.042). The context tilt is small because evidence does not grow on the conjunction Nethra; that is
  evidence change, not conduction. Open.

Core change: parameter `conduction` ("top_and_leaves" default, "top", "all"); `_covered_members`
holds the rule. Checks: "top" and "all" equal the previous commit's `top_only_conduction` True/False
(checkpoints with the parameter key normalized, activations and conducting incidences; all
bitcheck modes); the default equals the prototype rule (`conduction_variants.py` topleaves) on the
bitcheck streams with a round trip, and gives the same output on `cue_capacity`, `human`, `robust`,
`focus_symbolic`, `consequence_reach`.

### 0f.3 Delta input, judged by field reads only

The structural read of §0e.3 (cells of after routes of Nethra refound by their before route) is a
lookup done by the harness on the topology, a transition table. The question here is whether the
field (P, frontier 0) carries the next position better with displacement Nethra. `delta_field.py`:
object bouncing on a line; MODE none / delta / shuffle (same displacement values permuted in time,
so they carry no information about the motion); ablation = frozen copy, same interval pushed
without the displacement Nethra. Share of P over position cells landing on the true next position:

| setting | none | delta | shuffle | delta, d not pushed at read |
|---|---|---|---|---|
| L=8, one per position, top | 0.33 | 0.22 | 0.20 | 0.24 |
| L=16, one per position, top | 0.64 (P total 0.001) | 0.075 | 0.089 | 0.082 |
| L=16, all | 0.40 | 0.14 | 0.16 | 0.18 |
| L=16, topleaves | 0.40 | 0.14 | 0.16 | 0.18 |
| L=16, speeds 1-3, topleaves | 0.19 | 0.13 | 0.11 | 0.13 |
| L=16, 6 coarse cells, top | 0.06 | 0.03 | 0.03 | 0.03 |

- Delta lowers the share at the next position in every setting, and real displacement is no better
  than shuffled: the field does not use the displacement's information about the motion.
- Mechanism (`delta_field.py` trace at x=8 moving left, L=16, top): d- (A 0.205) conducts to every
  leftward Nethra (A 0.045-0.10 all along the line), which pour P into every cell x1..x14. The
  displacement Nethra is a hub for "moving left", present all run long, not a pointer to the next
  cell. Without delta, the leftward Nethra conduct to no cells at all (covered), so the 0.64 share
  at L=16 top is a share of 0.001.
- So delta input makes the harness's structural lookup exact (§0e.3) and leaves the field's
  expectation worse. Not adopted.
- Also measured (§0e.1, rules above): no conduction rule makes positive P point forward on the
  graded ring (behind >= ahead at step 1 for every rule); the input-free continuation beats staying
  put only in some settings (afterall 0.77 on 12/8, topleaves 0.52 on 12/12, all 0.52 on 12/12).

### 0f.4 Scripts (`nethra/tests/`)

| script | what |
|---|---|
| `conduction_variants.py` | prototype rules (env COND), installed on `nethra.NethraField` |
| `with_conduction_variant.py` | `COND=afterall python3 with_conduction_variant.py cue_capacity.py` |
| `conduction_cost.py` | `COND=... conduction_cost.py 0 400 x`: cost table above |
| `cue_trace.py` | env COND: cue stream topology, conductances, activations |
| `delta_field.py` | env COND (default top), L, NC, MODE, ND, VMAX, SPEEDS, PASSES, ABL |
| `with_params.py` | now env LEAK and COND=top/all/top_and_leaves (core values) |
| `context_trace.py` | env COND (core values, default top_and_leaves) |

## 0e. Fourth session (2026-09-25, night): top-only in core, where next, delta input, leakage, execution

User's requests: put top-only into the core, provisionally; test leakage; find on a tiny exact loop
why the field doesn't carry where next; array storage; test the delta input idea.

### 0e.1 Why the field doesn't carry "where next" (tiny exact loops)

**Symbolic ring** (`ring_symbolic.py`, K=6, one Nethra per position, pushed 1.0, top-only). Predicted
from the code and matched: lap 1 builds N1..N5, lap 2 builds N6 (the wrap), then 0. N_k routes
{p_(k-1), N_(k-1)} | {p_k}; N1 = {p0} | {p1}; N6 = {N5, p5} | {N1, p0}. Closure at p_k = {p_k, N_k,
N_(k+1)}: N_(k+1) is refound by its before route and its after route is {p_(k+1)}, so **the structural
next is exact in every interval**. Conducting incidences (top-only): p_k - N_k, N_k - N_(k+1), N1 - p0,
N1 - p1. Conductances stay near the admission seed (g(14) = 0.196; 0.2-0.35, N1 0.56-0.78).

P toward the cells, last two laps (P = sum of max(0, g (A_relation - A_member)), completed interval):

| position k | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| P toward p(k-1) | 0 | 0 | 0 | 0 | 0 | 0 |
| P toward p(k+1) | 0.011-0.017 | 0.002-0.003 | 5e-4 | 1e-4 | 0 | 1e-3 |
| P toward p(k+2) | 0 | 0 | 0 | 0 | 0.003 | 1e-4 |

- Previous cell: 0, because its own leftover makes A_member > A_relation.
- Next cell: reached p_k -> N_k -> N_(k+1) -> p_(k+1), 3 hops at g ~ 0.2 against leak 1 (1e-4);
  at k=0 only 2 hops (N1 holds both p0 and p1): 1e-2.
- Being refound gives a Nethra no activation. N_(k+1) is structurally present but its activation
  comes only by conduction from the pushed cell.

**Graded ring** (`ring_graded.py`, 12 positions, 8 tent cells, step 1): built 8, 3, 1, 0, 0 per lap.
Structural next (cells of after routes of Nethra refound by their first route) 0.50 from the true
next, staying put 1.00, P as a point 3.6. Split of positive P (`ring_split.py`, last lap):

| setting | P behind | P at the object (within one cell spacing) | P ahead | input-free continuation as a point / staying put |
|---|---|---|---|---|
| 12 positions, 8 cells | 0.49 | **0.00** | 0.51 | 2.56 / 1.00 |
| 12 positions, 12 cells | 0.41 | 0.00 | 0.59 | 0.90 / 1.00 |
| 24 positions, 8 cells | 0.53 | 0.00 | 0.47 | 4.65 / 1.00 |
| 24 positions, 8 cells, step 5 (> spacing 3) | 0.32 | 0.00 | 0.68 | 3.69 / 5.00 |
| 12, 8, full conduction | 0.53 | 0.00 | 0.47 | 1.32 / 1.00 |
| 12, 8, leakage 2 / 4 | 0.57 / 0.52 | 0.00 | 0.43 / 0.48 | 2.42 / 1.70 |

(Input-free continuation: a copy of the field stepped once with nothing pushed; cells' activation
minus leakage-only decay, positive part, as a point.)

- **P toward the cells at the object is 0.00 in every setting.** When the step is smaller than a
  receptive field, the next position's cells are the ones active now, and no positive flow can go
  into a Nethra more active than its relation. So P as a point cannot show a small step; it only
  reaches cells that are off.
- The rest splits about half behind, half ahead. Traced at x=5 (`ring_trace.py`): leftover in the
  previous cell c2 (A 0.19) feeds N2 (the Nethra of an earlier transition; A 0.05), which flows into
  c1 behind (+0.0066); ahead, N6 is two hops from the pushed cells (A 0.001, +0.0001 into c5).
- Leakage 1, 2, 4 does not change the split.
- Direction exists in the structure (before vs after route), not in conduction: an incidence
  conducts both ways, and the same incidence carries the forward expectation before a cell is pushed
  and the backward spill after it, so evidence change pushes its g up and down.

### 0e.2 Top-only in the core, and what it breaks

Implementation: parameter `top_only_conduction` (default True). Equal to `TopField` bit-identically
(`bitcheck_cores.py`-style check against the prototype on symbolic and graded streams, default,
frontier, rk4 + 0.9, whole joining, round trip). Covered incidences are zeroed after construction
re-seeds a retained route (as the prototype did); the route summary Counter keeps the re-seeded max
(only its > 0 is read).

User scripts, leakage 1, full conduction (False) vs top-only (True):

| script | full conduction | top-only |
|---|---|---|
| `context_partwise.py`, small factors first: C1+X P toward Y / Z | 0.0458 / 0.0424 | 0.0432 / 0.0014 |
| same, C2+X P toward Z (right) / Y | 0.0456 / 0.0419 | **0.0014 / 0.0418 (wrong)** |
| `cue_capacity.py` A->B then A->C, no cue: B / C | 0.052 / 0.069 (recent) | 0.0034 / 0.0011 (first) |
| same with differentiating cues: cue1 B / C; cue2 B / C | 0.102 / 0.065; 0.065 / **0.101** | 0.086 / 0.0008; **0.043 / 0.003 (wrong)** |
| `cue_capacity.py` 8 regimes at 11/39/61% overlap; 15 regimes | 1.00 / 1.00 / 0.94; 0.83 (234 s) | 1.00 / 0.94 / 0.88; 0.70 |
| `human.py 14` blocking ratio | 0.38 | 0.89 (no blocking) |
| same, interference after A: B / C; after 150 unrelated | 0.018 / 0.040; 0.015 / 0.036 | 0.0016 / 0.0024; 0.0011 / 0.0001 |
| same, spacing: massed / spaced, immediately | 0.037 / 0.025 | 0.0017 / 0.046 |
| same, XOR learning curve | 0.59-0.78 | 0.55-0.69 |
| `robust.py` Nethra clean / partial / noisy / noisy2 | 0.83 / 0.38 / 0.50 / 0.04 | 1.00 / 0.40 / 0.58 / 0.06 |
| `focus_symbolic.py` share with / without F | 0.495 / 0.482 | 0.489 / 0.488 |
| `consequence_reach.py 8 0.95 10 40 10 0` P toward F on contact, first k / peak; ms | k=8 / 4.3e-2; 51 ms | k=6 / 1.3e-2; 10 ms |

**Correction to §0b:** the third session reported top-only `context_partwise.py` as "P right/wrong
0.0432/0.0014"; that is only the C1 row. In the C2 row top-only primes the wrong continuation 30:1.

Mechanism (`context_trace.py`): X conducts only to N10, the first Nethra built with X (X -> Y was
learned first). Y hangs on N11 = {N10, X} | {Y}, next to N10: X reaches Y in 3 hops. Z hangs only on
N14 = {N10, N11, N13, X} | {Z}, whose members X, N10, N11 are covered by N13: X reaches Z via N10 ->
N11 -> N13 -> N14, 5 hops (g N14-Z 0.28 vs N11-Y 1.48). The phase-2 conjunction Nethra (C2 + X -> Z)
conduct to neither X nor Z directly either. Whatever is built second on a member is reached through
what was built first on it.

### 0e.3 Delta input (`delta_input.py`, `delta_runlength.py`)

Object bouncing on a line of L positions; speed per pass from SPEEDS. Displacement Nethra: ND tent
cells over [-VMAX, VMAX], pushed with their shares of x_t - x_(t-1) in the same interval as the
position. Last 2 passes; errors in positions.

| position cells | speeds | structural next without delta | with delta | staying put |
|---|---|---|---|---|
| one per position, L=8 | 1 | 0.86 (1.9 cells in the set) | **0.00** (1.0) | 1.00 |
| one per position, L=16 | 1 | 0.93 | **0.00** | 1.00 |
| one per position, L=16 | 1, 2, 3 | 1.77 (2.7 cells) | **0.02** (ND 7) | 1.70 |
| 6 tent cells over 16 | 1 | 1.10 | 0.67 (ND 3) | 1.00 |
| 6 tent cells over 16 | 1, 2, 3 | 1.72 | 1.75 (ND 7), 0.62 (ND 4; 26/30 intervals have a read) | 1.70 |

- Predicted and seen: without delta, at x every Nethra whose after route is cells(x) is refound in
  both directions, so both x+1 and x-1 are read. With delta the after routes carry d+ or d-, and only
  the current direction's Nethra is refound.
- P: share toward displacement Nethra 0.15-0.33; P ahead / behind along the motion 0.60 / 0.40
  (without delta 0.37-0.79 ahead, varying). P at the object's own cells 0.00-0.01 as in §0e.1.
- Refound constructed Nethra per interval fall with delta (3.9 -> 2.1; coarse 7.3 -> 4.5).
- **How long in a direction:** no constructed Nethra is refound across a run of equal
  displacements; each is refound in at most 2 intervals (its before and after). Every interval of a
  run pushes the same displacement pattern, so structure cannot count a run. Duration shows only as
  charge on d+: 0.219, 0.230, 0.234, 0.235 ... at leakage 1 (saturates in 2 intervals); 0.318,
  0.375, 0.412, ... 0.539 at step 8, still rising, at leakage 0.25. d+ itself stays at ~0.22 of its
  push because it conducts into every Nethra of that direction.
- Side finding, `_admit_by_parts`: without delta, the first leftward pass built nothing. A rightward
  Nethra N(x -> x+1) has route {x+1} complete on the before side of the leftward transition x+1 -> x
  and route {x, ...} complete on the after side, so it counts as accounting for both parts. The
  accounting does not ask which route is the before route. Leftward structure was built on the second
  leftward pass (whole joining on recurrence). Not changed.

### 0e.4 Leakage (user: "you can test")

All top-only; user scripts via `with_params.py` (sets the default leakage of every field).

| script | leakage 1 | 2 | 4 |
|---|---|---|---|
| `cue_capacity.py` 8 regimes 11/39/61%; 15 regimes | 1.00/0.94/0.88; 0.70 | 1.00/1.00/0.96; **0.93** | 1.00/1.00/0.94; 0.90 |
| `cue_capacity.py` cue2 B / C (C right) | 0.043 / 0.003 | 0.022 / 0.009 | 0.0075 / 0.0040 |
| `human.py 14` blocking ratio | 0.89 | 0.78 (both ~0) | 1.02 (both ~0) |
| same, interference after 150: B / C | 0.0011 / 0.0001 | 0.0285 / 0.0000 | 0.0079 / 0.0000 |
| same, spacing massed / spaced | 0.0017 / 0.046 | 0.0001 / 0.025 | 0.0000 / 0.0019 |
| `robust.py` clean / partial / noisy / noisy2 | 1.00/0.40/0.58/0.06 | 1.00/0.38/0.65/0.06 | 0.67/0.38/0.46/0.06 |
| `context_partwise.py` small first, C1: Y / Z; C2: Z / Y | 0.043/0.0014; 0.0014/0.042 | 0.0045/0.00006; 0.00006/0.0042 | 1e-4/0; 0/1e-4 |
| `focus_symbolic.py` share with / without F | 0.489 / 0.488 | 0.505 / 0.507 | 0.502 / 0.502 |
| `consequence_reach.py` P toward F on contact, first k / peak; ms | 6 / 1.3e-2; 10 | 6 / 1.9e-3; 7.6 | 4 / 1.5e-4; 5.1 |
| ring (§0e.1) P behind / ahead | 0.49 / 0.51 | 0.57 / 0.43 | 0.52 / 0.48 |

- Higher leakage shrinks every P and activation (10x at 2, 100-1000x at 4 in `context_partwise.py`,
  `consequence_reach.py`) and cost (10 -> 5 ms). Construction counts equal in these scripts (the
  counts printed are the same at every leakage).
- `cue_capacity.py` 15 regimes: 0.70 -> 0.93 -> 0.90. The others: no consistent change.
- Leakage stays 1 (parameter; CLAUDE.md §2.3).

### 0e.5 Execution (roadmap item 7), bit-identical

Two-object binocular stream (gpu_bench stream: 3 alone laps each, 120 intervals together, 80
measured; 857 Nethra, frontier 0.01, one thread, local container), ms/interval:

| | start of session | end |
|---|---|---|
| top-only (default) | 27.0 | **12.3** |
| full conduction | 135.6 | 119.0 |

End state by part, top-only (`exec_parts.py`): ETD integration 5.6, evidence change + construction
2.3, compile 0.8, closure 0.7, pair statistics 0.7, incidence compile 0.7, pattern refinding 0.3,
frontier 0.25.

What changed (all execution indexes, rebuilt from routes; nothing persistent):
- `_physical_incidences`: per relation, its non-covered incidences grouped by member in
  first-appearance order (rows and dict order unchanged; edges with only covered incidences are left
  out, they have g = 0). With top-only 6,060 of 6,907 incidences are covered.
- `_canonical_source_event`: stored patterns' sorted values and norms cached.
- `_compile_interval`: directed incidences and co-supplier pairs built with numpy in the same
  per-receiver order; pair codes handed to `update_residuals`.
- `closure`: route uses per member with route-use ids; the event projection is skipped for routes
  that hold only unqualified evidence (a non-empty projection cannot match).
- frontier halo neighbours cached per Nethra.
- evidence change: prior flow, tension and both evidence terms computed with numpy over the
  conducting incidences in `physical` order (np.bincount adds each bin in array order from 0.0).
- Not done: the evidence Counters are still dicts (the writes remain a Python loop). Remaining time
  is mostly exact integration; faster integration would change rounding.

Checks: `bitcheck_cores.py` against the previous commit after each change (checkpoint, activations
and conducting incidences equal; symbolic and graded streams, 2 seeds, default / frontier 1e-2 /
frontier_min / rk4 + 0.9 / whole joining / evidence change off, top-only on and off, checkpoint round
trip except frontier_min whose `_tol` is not checkpointed); `exec_prof.py` hashes on the two-object
stream (both conduction modes); `context_partwise.py` and `human.py 14` outputs identical.

## 0. Third session (2026-09-25): per-part subtraction decided and moved into the core

The user's answer to §9.1: test (b) structural per-part subtraction for vision, or an alternative;
use whichever is better. Result: an alternative, now the core default.

**Rule (`join_on_recurrence=True`, default):** a transition of pushed Nethra (the Nethra pushed in
interval t-1, the Nethra pushed in t) seen for the first time is subtracted per part
(`_admit_by_parts`, the `PartwiseField` logic of §6.2): constructed Nethra with a route complete on
each side account for the pushed Nethra in those routes; nothing is built for accounted parts, the
unaccounted remainder is joined on its own. When the same transition of pushed Nethra recurs, the
whole interval is joined as before (`_admit_whole_support`). A co-presence that recurs becomes
structure (one occurrence later than before); one that happens once does not.
`join_on_recurrence=False` is the previous core, bit-identical. Old checkpoints load with False.
Bookkeeping: `witnessed_transitions` (set of pushed-member pairs, checkpointed; grows with the
number of distinct transitions).

Checks: `False` = previous core, identical checkpoints and activations on a symbolic and a graded
stream, each with default, frontier 1e-2, rk4 + threshold 0.9; `True` = the scratch prototype
bit-identical on the same; checkpoint round trip mid-stream bit-identical.
`source_support="product"` could not be compared: **the previous core itself gives a different
checkpoint on every run in that mode** (3 runs, 3 hashes; graded stream). Not investigated.

Recurrence criteria tried (scratch prototypes; two objects, 0.8, jitter ±10, R=16, `binocular_multi.py`):

| gate | eyes joined (mono first) | context conjunctions | 40/37 loops, 400 together: whole-joined intervals, built |
|---|---|---|---|
| canonical pattern recurs (cosine) | yes | yes | 126, 151; cost 76 → 624 ms |
| pushed member set recurs | yes | yes | 52, 97 |
| canonical pattern pair recurs | yes | yes | 52, 90 |
| **pushed member-set pair recurs (adopted)** | yes | yes | **18, 68**; 73 → 169 ms |
| none (per-part always, option b) | **no** | **no** | 0, 51; 15 → 29 ms |

At cosine 0.8 joint patterns of independent objects "recur" approximately, so a cosine gate still
joins them. The adopted gate uses the exact set of pushed Nethra, independent of the threshold.

Measurements, adopted rule vs previous core vs (b), `16 0.8 10 3 160 40` (loops 40 and 30):

| | built together 0-119 | ms/interval together | 3D located obj0 / obj1 per 40 |
|---|---|---|---|
| previous core, monocular first | 120 | 107 → 305 (0-79; 400-800 later, §6.1) | 35/27, 36/31 |
| (b) per-part, monocular first | 23 | 16-20 | 4/6, 8/5, 6/3 |
| **adopted, monocular first** | **24** (4 in 120-159) | **63-81** | **34/27, 35/31, 34/30, 34/27** |
| previous core, both eyes first | 120 | 47 → 165 | 31/31, 36/30, 27/34, 32/30 |
| adopted, both eyes first | 59 | 32-56 (parallel runs) | same as core |

Longer, joint never repeating (loops 40/37, 400 together, adopted): built 22/12/9/14/11 per 80
(47 of 68 from unaccounted single-object remainders, i.e. jittered positions not seen alone yet;
18 on recurrence); frontier 239 → 358, cost 73 → 169 ms, still rising. The rise is frontier size
(integrated Nethra), not construction; that is the execution item (§8).

Other streams, adopted vs previous core:
- `context_partwise.py`: the 8 conjunction Nethra are built (one block later); P right/wrong
  0.0458/0.0424 vs 0.0467/0.0428. (b) builds nothing there.
- `consequence_reach.py 8 0.95 10 40 10 0`: 263 vs 281 constructed, 31 vs 35 ms; P toward F on
  contact 2.4e-3 ... 4.3e-2 vs 2.3e-3 ... 4.2e-2; misses 0 in both.
- `focus_symbolic.py`: share 0.495 / 0.482 vs 0.495 / 0.481.

Open from this: rising frontier cost with two objects (§8); `product` mode nondeterminism; the
witnessed-transition set grows without bound on a never-repeating stream.

## 0b. Third session, part 2: why cost rises, participation threshold, top-only conduction, GPU

User: cost must not keep rising; expected ~10 Nethra participating with two objects; "the top Nethra
is computed, not every leaf"; maybe a dynamic participation threshold that rises with confidence;
look at what humans do; use the Fedora runner for CUDA.

**Who participates** (two objects together, interval 240, current core): pushed 16; |a| >= 0.1:
0-2, >= 0.03: 10-19, >= 0.01: ~130 (100+ constructed), >= 0.001: ~300-390 of 864. Frontier (one-hop
halo) 256-355, 3.7k-5.7k incidences. So ~10-19 carry real activation, but the ~110 thin ones
(0.01-0.03) together hold more activation than the top ones. Cause: a route has ~23 members, 93% of
them leaves already inside another member's route (§6.1), and **each member is a conducting
incidence**. 6,909 of 7,799 incidences are such leaf duplicates. Every constructed Nethra touching an
active cell draws flow from the leaves directly.

**Participation threshold** (frontier tolerance; from interval 240, 80 intervals, error vs exact for
the 13.5 Nethra/interval with exact |a| >= 0.03). Structure reads (built, refound, 3D located, state
distance) are identical at every tolerance, including exact.

| current core | ms | relative error | top-10 same as exact |
|---|---|---|---|
| exact | 484 | 0 | 1 |
| 0.01 | 137 (rising) | 3.0% | 0.99 |
| 0.03 | 57 | 13.5% | 0.96 |
| 0.05 | 53 | 19.5% | 0.95 |
| dynamic 0.01 / 0.05 | 65 | 18% | 0.95 |

Dynamic = next interval's tolerance high when every pushed Nethra was accounted for (by parts or
whole), low otherwise; 66/80 intervals were accounted. Dynamic gives the same trade as a fixed
tolerance at its high value. No gain found.

**Top-only conduction (`nethra/tests/top_conduction_prototype.py`, prototype, not core):** routes
stay whole (closure unchanged); at construction a member that lies in a complete route of another
member of the same route (covered) earns no incidence evidence, so g = 0 there. Only the top members
conduct; leaves reach the relation through the top member's own routes. Frontier halo through
conducting incidences only. Construction and every structure read are identical to the core.

| two objects, 40/37 loops | current core (join_on_recurrence) | top-only |
|---|---|---|
| ms/interval together, 0-79 ... 320-399 (tol .01) | 73, 97, 113, 130, 169 | **13, 17, 18, 21, 19** |
| frontier | 239 → 358 | 114 → 136 (levels off) |
| exact integration, whole field (866 Nethra), local | 484 ms | 79 ms |
| error of important Nethra at tol .01 / .03 | 3.0% / 13.5% | 1.7% / 4.2% |

Behaviour on the other streams, current core vs top-only:
- `context_partwise.py` small factors first, P right/wrong: 0.0458/0.0424 vs **0.0432/0.0014**. **(Fourth session: this is only the C1 row; in the C2 row top-only primes the wrong continuation, §0e.2.)**
  Context only: C1 0.104/0.054 vs 0.069/0.060, C2 0.103/0.055 vs 0.109/0.019 (depends on build order:
  a path through more hops is weaker).
- `consequence_reach.py 8 0.95 10 40 10 0`: P toward F starts at k=8 and peaks 4.3e-2 vs starts at
  k=6 and peaks 1.3e-2. The earlier reach was receptive overlap through leaf cells (§7.1); with leaves
  not conducting, anticipation needs the top Nethra itself to activate. 7.9 vs 31.7 ms.
- `focus_symbolic.py`: F tilt of the gaze share +0.013 vs ~0.

Not moved into the core: which incidences conduct is a field-law question (CLAUDE.md §2.1:
"all earned incidences conduct"; here the leaves are never earned). **User's decision.**

**What humans do** (for the threshold question): only ~1-4% of cortical neurons are strongly active
at once and the energy cost is at active synapses (Lennie 2003, Curr Biol 13:493); expected input
gives lower overall V1 activity with a sharper representation (Kok, Jehee, de Lange 2012, Neuron
75:265); gain shifts between a focused mode when the task is predictable and a broad mode when it is
not (Aston-Jones & Cohen 2005, Annu Rev Neurosci 28:403). Sparse activity with a hierarchy of
convergent connections matches top-only conduction better than a moving threshold.

**GPU (Fedora runner, RTX 5070, cupy; `nethra/tests/gpu_field.py`, `gpu_bench.py`, workflow
`nethra-gpu-bench.yml`, run 36103291516).** ETD integration on the device; equals CPU ETD to 3e-16
(numpy fallback bit-identical). Two objects, 866 Nethra, 40 intervals after 240 together, one CPU
thread:

| | CPU exact | GPU exact | CPU tol .01 | GPU tol .01 |
|---|---|---|---|---|
| current core | 258 (integration 199) | 130 (66) | 65 (25) | 65 (23) |
| top-only | **36 (2.6)** | 101 (67) | **15.7 (2.3)** | 64 (52) |

The GPU halves exact integration for the current core. With top-only conduction integration is
2.6 ms for the whole field; the rest is per-incidence Python bookkeeping (evidence change, incidence
compilation, closure, pattern cosine), which a GPU integrator does not touch. Next for speed: array
storage of incidences (roadmap item 7), bit-identical.

## 0c. Third session, part 3: focus = fine fovea at the gaze, top-only background (first test)

User: top-only for the background, more detail around focus. Understood as: detail comes from what
is pushed at the gaze (a fine fovea per eye + gaze Nethra over the 3D fixation point), background
from coarse periphery cells; top-only conduction everywhere. A gaze-dependent switch of which
incidences conduct would be a gate on g (CLAUDE.md §2.1), so it was not tried.

Script `nethra/tests/focus_fovea.py`. Per eye: periphery 8x8 tent cells over the whole image, fovea
8x8 tent cells over +-0.2 image units around the image of the fixation point; gaze 6^3 tent cells
over the box (8 pushed). Pursuit device outside the core: fixation(t) = focused object at t-1, so
the fovea sees the object's step. Stream: object 0 alone tracked (3 laps of 40), object 1 alone
tracked (3 laps of 37), then both, gaze on object 0. 0.8, jitter ±10, tolerance 0.01. Reads: P
toward fovea cells as a centroid = expected next offset of the focused object; P toward the
periphery cells within 2 spacings of the background object = its expected next image point; both
against "staying put" (image units; fovea spacing 0.057, periphery 0.286).

| both objects, 160 intervals | ms/interval | frontier (210 constructed) | focused, expected next offset | background, expected next point |
|---|---|---|---|---|
| current core | 232 → 627 | 404 → 474 | 0.060-0.071 (staying put 0.036-0.043) | 0.19-0.21 (0.048-0.055) |
| top-only | 105 → 132 | 376 → 442 | 0.060-0.068 | 0.17-0.20 |
| top-only, no fovea | 25 → 43 | 180 → 239 (117 constructed) | | |
| top-only, no gaze | 55 → 78 | 234 → 288 (193 constructed) | | |

Built during both: 17-28 per 40, not settling within 160 intervals.

Measured cause of the frontier (1 lap alone + 40 both, top-only): 65 of 76 constructed Nethra above
tolerance, 53 of 216 gaze cells (8 pushed per interval), 31 fovea, 32 periphery. With pursuit the
focused object's fovea pattern is nearly the same every interval (its step), so the same few fovea
cells are pushed every interval and are members of nearly every Nethra built for that object; its
whole structure stays lit and conducts hop to hop along the chain, top-only or not.

Both expectation reads are worse than staying put in every variant (as §5.3 found for P as a 3D point).

Human reference: an image held still on the retina fades (adaptation), and much of the retinal
output signals change (transient cells). In Nethra a constant push is constant presence (like
holding). Options, not started (input design, CLAUDE.md §4 says input goes in as the notes say; user's call):
1. Fovea through change-signalling receptive Nethra (pushed with the positive part of the change of
   a cell's input), beside or instead of sustained ones.
2. Coarser gaze grid / fewer gaze cells.
3. First find out why P as an expected next point is worse than staying put (open since §5.3)
   before building more input channels on it.

## 0d. Third session, part 4: the three options and charge vs drain (`focus_fovea.py`)

All top-only, tolerance 0.01, 2 laps alone per object, both objects together with gaze on object 0.
Reads added: coarse periphery expectation for the focused object as well (same resolution as the
background); share of P toward previous-only / current / next-only cells; a structural read (cells in
the after routes of constructed Nethra whose first (before) route is complete in this interval's
closure). Image units; staying put ≈ 0.04-0.07.

| setting, both objects 0-79 | frontier | ms (parallel runs) | focused fovea offset / staying | focused coarse point / staying | background / staying |
|---|---|---|---|---|---|
| sustained fovea (base) | 345-379 | 87-120 | 0.073 / 0.048 | 0.19-0.22 / 0.065 | 0.20 / 0.05 |
| 1. change-signalling fovea only | 355-398 | 99-119 | - | 0.17-0.20 / 0.064 | 0.21-0.22 / 0.05 |
| 1. sustained + change-signalling | 452-494 | 137-202 | 0.068-0.071 | 0.19-0.21 | 0.19-0.21 |
| 2. gaze grid 4 (instead of 6) | 267-295 | 76-93 | 0.075 | 0.20-0.22 | 0.20-0.21 |
| leakage 2 | 291-322 | 74-95 | 0.072 | 0.18-0.21 | 0.19-0.20 |
| leakage 4 | 192-207 | 47-59 | 0.078-0.080 | 0.20-0.22 | 0.20 |
| leakage 4 + gaze 4, 160 intervals | 174 → 208 | 40 → 68 (alone) | 0.078-0.082 | 0.19-0.22 | 0.20-0.22 |

Change-signalling cells are pushed with max(0, input now - input last interval) per fovea cell.

3. Why expectation reads are worse than staying put:
- Share of P (base): toward previous-only / current / next-only cells: fovea 0.05-0.06 / 0.06-0.09 /
  0.07-0.09; background periphery 0.07 / 0.12-0.14 / 0.06-0.07. Previous ≈ next (conduction is
  symmetric; the only forward bias is leftover in the previous cells), and the three together get
  only 20-30% of P; the rest goes to cells of other positions along the loop.
- Structural read: focused 0.113-0.152 (staying put 0.059-0.072), background 0.18-0.34 (0.043-0.046);
  available in only 26-72 of 80 intervals.

Charge vs drain (the user's question): leakage sets the leftover (e^-leak per interval at C=1,
dt=1) and the per-hop spread (about g/(g+leak) per hop). Leakage 4 halves the frontier; it does not
change the reads. Construction counts at leakage 2/4 equal those at 1 in these runs (27/24/...), but
admission-seed balance, M/P and the other streams were not measured.

## 1. Goal (the user's words, condensed)

1. Binocular vision input; track objects in a 500 x 500 x 500 space; an internal state that
   replicates what it sees; almost real time. If something doesn't work, assume the approach or
   the understanding is wrong, not Nethra.
2. Once structure has settled is what matters; building it may take long.
3. Like people: one thing is tracked accurately at a time; the rest goes by roughly in the
   background. Focus moves to what is consequential. A "want" (explicitly **not** a reward, and not
   surprise minimisation as the whole story) could come from associating something like feeding,
   or a signal tied to the field carrying things well, with states. The extension is carrying
   expectations further ahead: look at an object, expect when it becomes consequential, move focus
   then. The user expects this needs delayed sequences / linking across time to be solved.
4. Time: intervals are fixed steps now. The user asked whether a time-difference input can
   decouple the field from fixed intervals, and whether the field can come to carry Δt (§7.4).
5. Execution: use CUDA on the Fedora self-hosted runner (20 CPU cores there, GPU). Not started (§8).

## 2. Standing user decisions and working rules learned this session

- Never-seen positions/situations are not a concern.
- `source_similarity_threshold` may be varied freely in tests.
- Runs: **anything over 2-3 minutes means something is wrong; stop it.** Short runs, often;
  go bigger only when a divergence from the short run is expected and stated.
- Do not start a next step unasked. The user picks (§9).
- The user describes ideas loosely and asks for them to be understood in the code first
  ("investigate and make sure you understand"). Do that, report, then build.
- Mistakes made this session (do not repeat):
  - Predicted "high share" for a context test without computing magnitudes; the current core
    gives 0.52 there (§6.3). Predict numbers, not adjectives.
  - Put a consequence zone at the image centre, where every far object also projects (§7.1).
  - Read weak expectations (P toward F) with the frontier on: the frontier never integrated them
    (F's activation 1e-9 while the read P said 1e-3). **Weak expectations need frontier 0.**
  - One run went ~5 minutes (exact integration, 1,170 intervals at 280 ms).
  - Reported a positive-sounding "reach of 8 intervals" that is receptive overlap (§7.1). Say the
    mechanism with the number.

## 3. Core facts checked in code this session (use them in predictions)

- Construction: `_admit_whole_support` joins the complete closure of consecutive intervals.
  (Since §0: only when the transition of pushed Nethra recurs; otherwise `_admit_by_parts`.)
  Reuse happens only through the source-pair index (`_existing_temporal_support_relations`) or
  `_accounted`, and `_accounted` **skips** handles already indexed to a different source pair.
  So subtraction before construction is per whole interval, not per part.
- Construction gate: `unresolved = sum over pushed Nethra of max(0, M - P) > admission_threshold`
  (1e-12). P never comes near M (§5.4), so this gate is effectively always open.
- Structural source event: `_canonical_source_event` replaces the pushed pattern by the stored
  pattern with the highest cosine above the threshold, and closure starts from **that pattern's
  members**, which can differ from the Nethra actually pushed (§6.2).
- Closure refinds a Nethra when one of its routes is complete, by membership only; graded values
  do not enter. A route that is a subset of the present cells is refound too.
- `step(dt)` takes any dt: integration is over dt, leakage decay `exp(-leak dt / C)`, activation
  integrals `A` scale with dt. Construction joins consecutive intervals whatever dt is; evidence
  changes once per interval with rates not scaled by dt.
- Frontier (`frontier_tolerance`): Nethra below tolerance and outside the one-hop halo only decay.
  Anything whose activation stays below tolerance never receives flow.

## 4. Harness scripts (all in `nethra/tests/`, all run in under 2 minutes with the arguments shown)

| Script | What | Example (runtime) |
|---|---|---|
| `binocular3d.py` | world, eyes, retina, reads (previous session). Eyes at z=-500, x=220/280; R x R tent receptive Nethra per eye | imported by the others |
| `binocular_path.py` | one object, exactly repeated circle (previous session) | `16 5` |
| `binocular_settled.py` | one object on a closed Lissajous loop with per-lap jitter; phases left eye, right eye, both. Per lap: built, cost, refound, state distance, P lead, P as a 3D point, unresolved/M and P/M at the pushed Nethra | `16 0.8 40 8 10` (~7 s) |
| `binocular_patterns.py` | distinct structural patterns over a position grid of the box (no field) | `16 25 0.8 0.95` (~8 s) |
| `binocular_multi.py` | objects on jittered loops, each alone first, then together; env `PARTWISE=1` uses the prototype, `EXACTSRC=1` its exact-source variant, `NOMONO=1` skips the monocular phases | `NOMONO=1 PARTWISE=1 EXACTSRC=1 ... 16 0.8 10 3 160 40` (~10 s; current core without the env vars takes minutes: two objects cost 400-800 ms/interval) |
| `partwise_prototype.py` | **prototype subclasses, not core**: per-part structural subtraction (`PartwiseField`), and with the pushed pattern as structural event (`ExactSourcePartwiseField`) | imported |
| `context_partwise.py` | context after small factors, current core vs prototype | no args (~5 s) |
| `consequence_reach.py` | objects approach a contact zone (contact) or pass (miss); F pushed after food contacts; optional whole-field colour Nethra; reads P toward F and live F by intervals before the end. `TOL` env = frontier tolerance | `8 0.95 10 40 10 0` (~25 s); `TOL=0 ... 8 0.95 10 16 10 1` (~15 s; 80 trials exact took ~5 min) |
| `focus_symbolic.py` | symbolic focus test: periphery x colour, gaze, fovea, F; does F bias the expected gaze toward food | no args (~40 s) |

## 5. Measurements: one object

All: frontier 1e-2 unless noted, one numeric thread, wall clock.

### 5.1 Free motion (bouncing, 8 units/interval), R=16

| threshold | built over 680 intervals | patterns | ms/interval at the end |
|---|---|---|---|
| 0.999 | 679 (one per interval) | 676 | 26.6 (from 6.7; frontier 47 → 111) |
| 0.95 | 575 | 289 | 23 |
| 0.8 | 297 | 143 | 11 |

Cost follows the number of constructed Nethra touching the active cells. Reads that are trivial
here: "state distance" 4-8 units (the Nethra built this interval is indexed by this very pattern)
and cos(P, next image) ~0.01 (P toward pushed Nethra is 0).

### 5.2 Patterns needed to cover the box (grid step 25, single object)

| R | 0.8 | 0.95 | 0.99 |
|---|---|---|---|
| 16 | 562 (curve flattening: 302/400/485/562 at 1/8,1/4,1/2,all points) | 2,269 (rising) | 5,891 (rising) |
| 32 | 2,056 | 5,364 | |

### 5.3 Settled loop (40 positions/lap across the box, steps 31-56 units), R=16

| setting | settles | settled ms | refound centre to object |
|---|---|---|---|
| exact path, 0.999 | after 2 laps | 10 | median 24 |
| jitter ±10, 0.999 | never (40 new/lap) | 97 at lap 12, rising | 20-26 |
| jitter ±10, 0.95 | slowly (12/lap at lap 12) | 50, rising | ~23 |
| jitter ±10, 0.8 | from lap 7 (0-4/lap) | 14-16 | ~28 |

P lead along the motion +0.15 grid spacings (a step is ~1 spacing). P turned into a 3D point:
98 units from the next position, 111 from the previous, while the current position is 48 from the
next. P is not a usable position forecast here.

### 5.4 P against M at the pushed Nethra (exact loop)

P/M: 0.05 (first binocular lap), 0.17 (lap 24), 0.25 (lap 104), 0.27 (lap 204), flattening. Only
~8% of pushed Nethra ever have P >= M. Settled conductances: median g 0.235 (g_max 1.5), 90th
percentile 0.61. Activation integrals: constructed Nethra median 0.0097, pushed cells up to 0.12.
This is roadmap item 6 (residual scale) measured on a settled stream.

## 6. Measurements: several objects

### 6.1 Current core

- Two objects on exactly repeated loops of equal length (8 positions): built 26 exactly as
  predicted (8 alone A, 9 alone B, 9 together). With both present, each object's own Nethra keep
  being refound (2.4 and 2.9 per interval); the Nethra built together add 2.6. Share of P toward
  each object's next point from Nethra built alone: 0.92/0.86 in the first joint lap, ~0.67 after.
- Two jittered loops (40 and 30 positions; together they repeat every 120 intervals), 0.8, R=16,
  monocular phases first: together built 40 per 40 intervals until 120, then 5-12. Cost **400-800
  ms/interval**.
  - The frontier is ~500 Nethra (core 154, halo 357), ~10,000 incidences, ~350,000 F61 supplier
    pairs, largest degrees ~100. Above 400 Nethra `auto` uses RK4 with 17 substeps
    (136 derivative evaluations per interval). ETD instead: 250-800 ms (max activation difference
    1e-5 after 5 intervals).
  - Exact integration (948 Nethra) ~1 s/interval. Tolerance 0.03 leaves only the pushed Nethra
    in the core (max activation anywhere is 0.083); activations then differ from exact by up to 0.03.
  - Routes: mean 23.4 members; 93% of members lie inside another member's complete route in the
    same route; ~1.6 "top" members per route.
- Independent objects: joint patterns are new at almost every interval, so construction settles
  only when joint states recur (lcm of periods; never for independent motion). Coarser thresholds
  cannot separate objects: below ~0.7 cosine the joint pattern refinds one object's stored pattern
  and the other drops out of the structural description.

### 6.2 "Only the top Nethra" (user's idea) and per-part subtraction

- Routes reduced to their top members (scratch experiment): closure runs away along the chain
  (refound per interval mean 15, max 28, against 2.6 with full routes). The cells in a route say
  *when* a Nethra applies: N(k+1) is present at k (via its before route) and at k+1 (after route);
  a route {N(k+1)} cannot tell which, so N(k+2) = {N(k+1)} | ... is refound at k, and so on.
- Per-part structural subtraction (prototype `PartwiseField`): every constructed Nethra with one
  route complete on the before side and one on the after side accounts for the pushed Nethra in
  those routes; if nothing remains nothing is built; otherwise build from the remainder only.

| two objects, 0.8, jitter ±10, R=16 | built in first 120 together | together ms | 3D located |
|---|---|---|---|
| current core, monocular first | 120 | 400-800 | yes |
| prototype, monocular first | 23 | 15-22 | **12-21 of 120** (eyes never joined) |
| current core, both eyes from start | 120 | 45-184 rising | yes |
| prototype, both eyes from start | 58 | 19-38 | 119/120 |
| exact-source prototype, both eyes, 3 alone laps | 20 | 28-33 | 38-40 of 40 |
| same, 10 alone laps | 11 (5 intervals with one object unaccounted) | ~36 | 38-40 of 40 |

- Why the substituting prototype still built: at 0.8 the structural event has exactly the pushed
  Nethra in only 160/240 and 105/180 intervals of the alone phases (62/80 together); at 0.95:
  196/240, 152/180, 78/80. Object Nethra built on substituted cells are not refound by the actual
  cells.

### 6.3 What per-part structural subtraction breaks

`context_partwise.py`: C1, C2 alone and X->Y, X->Z alone first, then C1+X->Y and C2+X->Z blocks.
Read on a frozen copy after C+X.

| | current core | prototype |
|---|---|---|
| small factors first: P toward right / wrong | 0.0467 / 0.0428 (8 conjunction Nethra, routes of 9-11) | 0.0572 / 0.0568, **nothing built in phase 2** |
| context blocks only | 0.1042 / 0.0536 | identical |

Structure alone cannot tell "independent object" from "context that matters" (or "the two eyes
of one object"). Per-part structural subtraction contradicts CLAUDE.md section 4 (larger structure
builds from small factors). Not moved into the core.

The numeric residual could tell them apart (a known object's next cells would have P ~ M; under a
context, P splits between Y and Z while M lands on one), but P reaches only 0.27 of M (§5.4).

## 7. Measurements: consequence, focus, time

### 7.1 How far ahead a consequence is carried (`consequence_reach.py`)

R=8, 0.95, 10-interval straight approaches from far (z 380-480) to a contact zone near the eyes or
to miss points; F pushed in the interval after food contacts; two silent intervals between trials.

- Zone at the image centre: P toward F ~5.8e-3 at every distance for contacts **and** misses.
  Cause: far objects project near the image centre too, onto the same central cells as the zone.
- Zone low in the view (250, 460, 30), P toward F on contact trials by intervals before the end
  (k = 8 ... 1): 2.3e-3, 5.1e-3, 6.9e-3, 1.0e-2, 1.1e-2, 1.9e-2, 2.8e-2, 4.2e-2. Misses 0.
- **Mechanism: receptive overlap, not reach through time.** Only one Nethra carries F; its route
  holds the 8 cells of the contact point. At k=10 the object's cells already share 2-4 of them;
  inflow grows with shared cells (4 -> 8). A path sharing no cells yet gives exactly 0 inflow.
  Chains of transitions carry nothing further ahead within one interval (leakage 1, dt 1). How far
  ahead the field carries a consequence = receptive field width relative to motion per interval.
  Coarse cells warn early, fine cells locate.

### 7.2 Which object is consequential (colour)

Two colours; whole-field colour Nethra pushed while an object of that colour is seen; F only after
food contacts. 80 trials.

| frontier 1e-2 | k=10 | k=5 | k=1 |
|---|---|---|---|
| food contact, P toward F | 1.6e-3 | 8.6e-3 | 2.9e-2 |
| food miss | 3.7e-3 | 5.5e-3 | 4.8e-3 |
| other-colour contact | 0 | 8.4e-3 | 3.0e-2 |
| other-colour miss | 0 | 0 | 0 |

Exact integration (live activation of F): food miss 2.5e-3 at every distance, other miss 1.1e-3;
contacts ramp to 2.4e-2 for both colours. The food colour Nethra is a member of the F Nethra's
route, so food anywhere carries F; but P is a sum over members and 8 zone cells outweigh one colour
Nethra, so a non-food object at the zone carries F as much. Only closure is crisp about "food and
at the zone". Exact integration here: 281 ms/interval (82 with frontier 1e-2), ~490 Nethra.

### 7.3 Focus: does a consequence pull the expected gaze (`focus_symbolic.py`)

Symbolic: 4 locations x 2 colours of periphery Nethra, 4 gaze Nethra, 2 fovea colour Nethra, F.
Trial: object appears (gaze elsewhere), scripted saccade, held 3 intervals, F after food; 2 silent.
Test on frozen copies: food and other object together, gaze elsewhere; share of P toward the gaze
Nethra at the food location.

- Share 0.495 with F, 0.481 without (both vary 0.15-0.85 by location pair).
- Paired on identical streams (F the only difference): +0.022 mean, positive 49/72. 40 trials
  +0.030 (12/16), 160 trials +0.025 (14/16): steady, not growing.
- Mechanism seen: the Nethra joining "food held at l" to F has route {periphery food at l, gaze l,
  fovea food, ...} | {F, ...}; food in the periphery at l drives it directly and it conducts into
  gaze l (largest single P toward gaze-at-food, 0.031). But the hold Nethra of both objects prime
  their own gaze locations about as much (0.022-0.031), so the tilt is small.

### 7.4 Δt (answer to the user's question, not yet tested)

- `step(dt)` already integrates any dt; field decay follows real time if the real elapsed time is
  passed. Construction and evidence change are per interval and do not scale with dt; P (from
  activation integrals) scales with dt while M does not, so varying dt changes M - P.
- The field can carry Δt only as a co-present fact: push Δt as a graded input through overlapping
  receptive Nethra (like any other value). Transitions then carry the Δt that elapsed.
- Test design (not run): constant-speed object sampled at random Δt in {0.5, 1, 2}; compare
  (a) step(1) always, (b) step(Δt), (c) step(Δt) plus Δt receptive Nethra; read construction count
  and P toward the next image point split by Δt. Predict first (CLAUDE.md section 0).

## 8. Execution / CUDA (second session's plan; see §0b for what was done)

- Fedora runner: `runs-on: self-hosted`. Earlier sessions triggered it with a workflow whose
  `on: push: paths:` matches the pushed files (see `.github/workflows/nethra-online-hotfield-cuda.yml`)
  or `workflow_dispatch`; results come back as uploaded artifacts / job logs.
  `.github/workflows/python-cuda-smoke.yml` reports whether cupy / numba / torch exist there.
  The user says: 20 CPU cores and CUDA.
- What costs time (two objects, §6.1): per-incidence Python loops in `_physical_incidences`,
  `_compile_interval`, `_move_evidence_and_construct`, `update_residuals`; the integrator is 25%
  (ETD) to 60% (RK4). Array execution checked bit-identical (previous session's procedure below),
  then a GPU path, is the plan the user agreed to in principle.
- Bit-identity procedure (previous session): keep the previous core
  (`git show <commit>:nethra/nethra.py > ref.py`), feed identical streams to both, compare
  `checkpoint_dict()` and every activation with `==`; modes: default, `source_support="product"`,
  frontier 1e-3/1e-2 with and without `frontier_min`, `integrator="rk4"`, evidence change off,
  checkpoint round trip mid-stream; one numeric thread. `np.bincount` adds in array order.

## 9. Open decisions for the user and proposed next steps (do not start unasked)

1. **Per part vs per interval.** Decided in §0 (recurrence-gated per-part subtraction, core default). Former options: (a) numeric residual per pushed Nethra (Nethra-native,
   separates objects from contexts in principle, but needs P comparable to M: residual scale or
   parameters, the user's call); (b) structural per-part subtraction for vision only (works for
   objects, loses eye joining unless each object is seen binocularly first, loses context
   conjunctions); (c) keep whole-interval construction and accept combination growth.
2. **Fovea and gaze** (the user's "one thing at a time"): design not built yet.
   - Per eye, gaze-centred retina: a small fine fovea and a coarse periphery, per colour; gaze
     proprioception as graded receptive Nethra over the fixation point (3D grid, so the binding of
     x, y, depth stays in one pattern); vergence puts the fixated object at zero disparity.
   - A tracked object's fovea pattern stays nearly constant, so its position lives in the gaze
     Nethra and it is continuously present (tracking acts as holding, see notes §7 on holding).
   - Devices outside the core: a pursuit reflex keeping the fixated object centred; saccades read
     the field's expected gaze (P toward gaze Nethra). A single eye must point one way; the device
     may pick a location, the core never does.
   - Predict first: structure count with several objects (fovea fine + periphery coarse), then the
     §7.3 bias in this setting.
3. **Reach further ahead**: multi-scale receptive Nethra (a coarse layer beside the fine one) to
   test whether anticipation starts earlier while position stays fine (§7.1 says reach = overlap).
   Beyond overlap, only holding joins distant intervals (notes §7).
4. **Δt** test (§7.4).
5. **Array / CUDA execution** on the Fedora runner (§8).

## 10. Previous session (condensed)

- Core commits on this branch: c00938b renames (old checkpoints load), b7005ef cosine structural
  recurrence restored, 890bf19 frontier path scales with the frontier (bit-identical), dfc7522
  faster incidence compilation (bit-identical).
- Established reads: live activation right after an interval is dominated by leftover (~37% of its
  push); P shows what structure carries forward. Internal state (refound constructed Nethra)
  resolves position to about one retina cell.
- Mistakes then: fed both eyes and full 3D motion from the start; hour-long runs with no
  prediction; edited a running script; `pgrep -f` loops; arithmetic mean of ratios; a scratchpad
  file named `nethra.py` shadowed the core.
