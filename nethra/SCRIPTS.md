# Scripts index

One line per script. `tests/` = user tests and traces used by the handoff; `experiments/` = measurement
harnesses added later. Run from the script's own directory, one numeric thread, with `timeout`.

Health check: `nethra/tests/smoke.sh` (~40 s, exits non-zero on any change in the default core's reference numbers).


## `nethra/tests/`

| script | what it does |
|---|---|
| `baselines.py` | Is Nethra doing anything a trivial learner cannot? Two textbook baselines on the same worlds, |
| `binocular.py` | Binocular 1D retina with moving objects. Each eye has W receptive Nethra (positions). An object |
| `binocular3d.py` | Binocular vision of point objects moving in a 500 x 500 x 500 box, small factors first. |
| `binocular_multi.py` | Objects on jittered closed loops. Each object alone first (left eye, right eye, both eyes), then all |
| `binocular_path.py` | One object going round the same closed path, seen by both eyes (binocular3d.py's world and |
| `binocular_patterns.py` | How many distinct structural source patterns (cosine matching at the given thresholds) a dense |
| `binocular_settled.py` | One object on a closed 3D loop across the box, each lap jittered by +-J units per axis. |
| `bitcheck_cores.py` | bitcheck.py REF.py NEW.py : identical checkpoints and activations on symbolic and graded streams, |
| `conduction_cost.py` | Cost by conduction variant (env COND, see conduction_variants.py): two-object binocular stream |
| `conduction_variants.py` | Conduction rule variants (PROTOTYPE, 2026-09-25 comparison, docs/HANDOFF_LOG.md section 0f). |
| `consequence_reach.py` | Consequence reach. Objects approach a contact zone near the eyes (contact trials) or pass it |
| `context_partwise.py` | Context after small factors: C1, C2 alone and X->Y, X->Z alone first, then C1 with X->Y and |
| `context_trace.py` | Trace why context selects the continuation built second (C1/C2 + X -> Y/Z); prints conductances per hop (log §0e.2). |
| `cue_capacity.py` | 1 A->B then A->C, with and without a differentiating cue present (cue1 during A->B, cue2 during A->C). |
| `cue_trace.py` | Trace cue selection in cue_capacity (A->B then A->C with cues) under a conduction variant. |
| `cycle_stream.py` | One stream of 500 numbers built from different cycles, one seed. What does the field carry |
| `cycles.py` | Cycles at different periods, one long stream fed several times. Watch structure build. |
| `delta_field.py` | Delta input, field reads only (env COND as in conduction_variants.py, default top; topleaves = core default). Object bounces on positions 0..L-1 (spee |
| `delta_input.py` | Delta input test. One object bouncing on a line of positions 0..L-1 with speed from SPEEDS |
| `delta_runlength.py` | How long in a direction: object bounces on 0..L-1 at speed 1 with displacement Nethra (d-, d0, d+). |
| `direction_bounce.py` | Symbolic bounce: positions 0..L-1, one Nethra each, pushed 1.0; object bounces 0->L-1->0... |
| `direction_motion.py` | 1D object, graded tent cells. Stream: RING (step STEP around a ring) or BOUNCE (bounces between |
| `direction_ring.py` | Symbolic ring, one Nethra per position. Reads last lap: directed conductances of N_(k+1) |
| `direction_variants.py` | PROTOTYPE, NOT PART OF THE CORE. Variants of direction="split" evidence change (docs/HANDOFF_LOG.md 0h). |
| `drain_prototype.py` | Drain incidences (PROTOTYPE, not core; user request 2026-09-25). docs/HANDOFF.md section 0h. |
| `drain_streams.py` | Tiny streams for the drain prototype (drain_prototype.py), a few hundred trials each, probed |
| `exec_parts.py` | parts.py CORE.py stream.json : ms/interval by part (wall clock wrappers, exclusive of nested wrapped parts). |
| `exec_prof.py` | usage: prof.py CORE.py stream.json [profile] -> ms/interval over the measure intervals, hash of result |
| `exec_stream.py` | Two-object binocular stream of gpu_bench.py (current core, top-only default, tol 0.01, 0.8). |
| `flip_prototype.py` | Flipped channels (PROTOTYPE, not core; user request 2026-09-25). docs/HANDOFF.md section 0i. |
| `flip_streams.py` | Tiny streams for flipped channels (flip_prototype.py), 300 trials each (bounce: 300 intervals), |
| `focus_fovea.py` | Focus test: coarse periphery (background) + fine gaze-centred fovea + gaze Nethra (focus). |
| `focus_symbolic.py` | Symbolic focus test. Nethra: periphery Pc[l][c] (4 locations x colour food/other), gaze G[l], |
| `gaze_loop.py` | Closed sensorimotor loop, reflex-taught, then field-driven. |
| `gaze_motor_probe.py` | Expose closed-loop with the retina reflex (as gaze_retina_device.py). Then, on frozen copies at each of the |
| `gaze_retina_device.py` | Closed sensorimotor loop. World: object bounces on 0..L-1 at speed 1. Eye at e. Each interval: |
| `gpu_bench.py` | Cost of one interval with two objects in view, CPU vs GPU integration, current core vs top-only |
| `gpu_field.py` | Execution only: integrate each interval on the GPU (cupy), same equation as nethra.py. |
| `human.py` | Tests with known human/animal signatures. Contract core unchanged: g_min 0, one push per present |
| `nethra_etd.py` | Import shim: re-exports nethra.py (old scripts import this name). |
| `nethra_presence.py` | Import shim: re-exports nethra.py (human.py and older scripts import this name). |
| `partwise_prototype.py` | PROTOTYPE, NOT PART OF THE CORE (subclasses of NethraField used by binocular_multi.py and |
| `regime_flow.py` | At a read (after ctx+B), decompose prior flow P (completed interval integrals) into each outcome O by |
| `regime_probe.py` | Split(-neg) stream; at each read (ctx+B of last quarter) re-run that interval on copies: |
| `regime_trace.py` | Nethra leading to each outcome: relations having O_j in a route |
| `regimes.py` | cue_capacity part 2 stream (contexts = pairs of P features, pushed in all 3 intervals of a block: |
| `ring_graded.py` | Tiny exact loop, graded: one object on a ring of L integer positions, NC tent cells (spacing |
| `ring_split.py` | Per interval (last lap): share of positive P toward cells behind / at / ahead of the object |
| `ring_symbolic.py` | Tiny exact loop: one object on a ring of K positions, one Nethra per position (symbolic) or |
| `ring_trace.py` | Trace ring_graded.py: which constructed Nethra carry the next position. |
| `robust.py` | Robustness and confidence. World: k regimes; context = 2 features from a pool of P; block = |
| `scale.py` | Cost and depth of the contract core as experience grows (g_min 0, admission seed 14, one push per |
| `smear2.py` | Graded smear counting 0..4 (triangular receptive Nethra, width 1.5, so at value v the neighbours |
| `timers.py` | Timers (square waves, periods PER, default 2,4,8) + process E pushed D intervals (D may be a list: drawn per episode), then finish O, then a gap of ti |
| `timers_trace.py` | Topology and per-relation flow into O at each k of the last episode (timers.py stream). Arg: COND. |
| `top_conduction_prototype.py` | PROTOTYPE, NOT PART OF THE CORE. Routes stay whole (closure unchanged); at construction only the top members of a |
| `transfer.py` | Sub-pattern transfer. Is a new next interval after an already-experienced sequence |
| `with_conduction_variant.py` | Run any test script with a conduction variant installed: with_conduction_variant.py SCRIPT ARGS (COND=...). |
| `with_direction_variant.py` | usage: VAR=op [DIR=split] python3 with_direction_variant.py script.py args (a nethra/tests script with |
| `with_params.py` | usage: [LEAK=2] [COND=top/all/top_and_leaves] [DIR=shared/split] python3 with_params.py script.py args... (runs a |

## `nethra/experiments/`

| script | what it does |
|---|---|
| `alpha.py` | Bigger alphabet, context selection with a common factor. P features; every pair is a context (k regimes), |
| `etd_operator_free_prototype.py` | Prototype (not in core): ETD operators applied as Q (f * (Q^T x)) instead of 7 formed N x N matrices. |
| `ext.py` | Nethra-only extension tests. One stream, one pass, no labels. Read: P toward candidates at the named interval |
| `par.py` | Run independent Nethra jobs in parallel processes, one thread each (GIL-free; one field per process). |

## `nethra/experiments/blocking/`

| script | what it does |
|---|---|
| `bl.py` | Blocking factor scan. Base = human.py test 1 (Kamin, separate fields). One factor changed at a time. |
| `prospect.py` | Kamin stream (human.py test 1). At each O interval: complete prospective state at O vs the observation's own |
| `scan.py` | Kamin blocking at 30/120/400 compound trials, blocked vs control, live and P read; KW='{...}' sets field parameters. |
