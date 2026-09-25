# Handoff: binocular Nethra, focus and consequence (state as of 2026-09-25, third session)

Read in this order before doing anything: `CLAUDE.md`, `nethra/NETHRA_OPERATING_NOTES.md`, this file,
then `nethra/nethra.py`. Branch `claude/sharp-thompson-wmszc5` (PR Dooces/nethra-cuda-runner#1 into
`main`; pushes to the branch update it). Scratchpad files are gone after a session; every script
below is in `nethra/tests/`.

**`nethra.py` was not changed in this session.** Everything new is harness scripts, one prototype
subclass file (`nethra/tests/partwise_prototype.py`, not core), and measurements.

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
- `context_partwise.py` small factors first, P right/wrong: 0.0458/0.0424 vs **0.0432/0.0014**.
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

## 8. Execution / CUDA (not started)

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
