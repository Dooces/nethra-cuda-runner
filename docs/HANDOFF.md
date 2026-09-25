# Nethra handoff

State after five working sessions on 2026-09-25. This file is the current state and the instructions.
Every measurement behind it is in `docs/HANDOFF_LOG.md` (verbatim session log, newest first). Section
numbers written "log §x" point there.

## 1. Instructions for the next context window

1. Read in this order: `CLAUDE.md` (all of it), `nethra/NETHRA_OPERATING_NOTES.md`, this file. Open a
   log section only when this file points to it. Read the parts of `nethra/nethra.py` you will make
   predictions about or change (CLAUDE.md §0, §2.7).
2. Branch: `claude/pensive-davinci-zla3pj` (Dooces/nethra-cuda-runner) holds all five sessions; `main`
   and the older `claude/*` branches are its ancestors. Open PR: https://github.com/Dooces/nethra-cuda-runner/pull/3
   (pushing to the branch updates it). If the session names another branch, start that branch from
   this one.
3. Setup: `pip install numpy` if missing; one numeric thread
   (`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`); `timeout` on every run; a run over 2-3
   minutes means something is wrong. Scratch scripts go in the scratchpad.
4. Replies to the user: very short and plain, no filler. Say what was pushed, what was read and the
   numbers; say plainly when something failed or was not done.
5. The bar is what people can do, as a sanity check (§2). Never judge against an optimum or a
   dedicated program.
6. Open work, in order (details §6):
   1. the user's open request: **fix the context loss of `direction="split"`** (traced, not fixed);
   2. the user's standing request: progress on **direction, timing and the action loop**.
   Anything else: propose it and ask first.
7. For every experiment:
   - write the prediction from the code first, run tiny, compare (CLAUDE.md §0);
   - for every claim that the field does the work, run the ablations: lesion (every constructed
     Nethra's incidence evidence 0), shuffled exposure, no construction, and without the input the
     claim depends on;
   - replicate before calling a difference (vary start, speed, seed, exposure length): single runs
     reversed twice today (§5, method);
   - after patching anything (wrapper, variant, parameter), print the parameter from inside the run
     to confirm the patch took effect.
8. At the end of a session: update §3-§6 of this file, and put the session's measurements at the top
   of the log as a new section.

## 2. The bar: people as a sanity check (user, 2026-09-25; CLAUDE.md §2.10)

- What people can do on the same task, with a source, says what is good enough. If people can do it
  and the field cannot (or only far below them), that is a failure.
- Not a cutoff: doing better than people is fine.
- Not a template: reproducing people's particular behaviour or limits (their lag, the spacing
  effect, a 2:1 forward bias in recall) is not a goal and not an argument for a design.
- Not an optimum: a dedicated program or machine doing better (a perfect actuator, a filter, a lookup
  table, an optimal controller) says nothing about what the whole system should organically do.
- No human data known for a task: say so; don't invent one.

Applied to what exists (the log's §0g.3, §0g.4 and §0h.3 were written before this was clarified):

| capability | people (sources §8) | field | check |
|---|---|---|---|
| anticipate a periodic reversal | start turning before the object after 1-2 cycles | turns 2-4 intervals after the object (the reflex: 4) | not good enough |
| expect the next step of a motion | yes | the field's own next step moves forward in 24/24 intervals with split, 8/24 with shared (graded ring) | split ok; shared not good enough |
| recall a sequence backward as well | yes, about half as often as forward | not measured directly; with split the backward conductances stay near the seed (0.22 vs 1.03) | unknown; measure |
| use what was present together (context selects the outcome) | pairs studied together are recalled both ways about equally; no data for these exact tests | shared: 15 regimes 0.93, 4 features 1.00; split: 0.59, 0.73 | split far below shared; no human number |
| retain after a delay | yes (spacing helps people) | both modes retain after 200 unrelated intervals | ok; spacing itself is not a criterion |
| attention pulled toward what was consequential | yes | shared +0.004 share, split -0.078 | not good enough |

## 3. The core now (`nethra/nethra.py`)

| parameter | default | other values | log |
|---|---|---|---|
| `join_on_recurrence` | `True`: first sight of a transition of pushed Nethra is subtracted per part; a recurring transition is joined whole | `False` = old whole-interval joining | §0 |
| `conduction` | `"top_and_leaves"`: covered constructed members earn no evidence, primitive members always do | `"top"`, `"all"` | §0f |
| `direction` | `"shared"`: one conductance per incidence, both ways | `"split"`: one conductance per flow direction, each moved by its own term of the evidence change; RK4 only, 2-4x slower | §0g, §0h |

Execution speedups are bit-identical (log §0e.5). With `direction="shared"` the core is bit-identical
to the one before the fifth session (`bitcheck_cores.py`: ALL IDENTICAL).

Known quirks (observed, not fixed):
- `_admit_by_parts` does not check which side a route is on (log §0e.3): a Nethra built for x -> x+1
  accounts for x+1 -> x, so the first pass in the other direction builds nothing.
- `_registering_before` is set in `_admit_sides` and never read.
- `source_support="product"` / `"min"` give a different checkpoint on every run; the adaptive
  frontier tolerance `_tol` is not checkpointed.
- Weak expectations need `frontier_tolerance=0` (CLAUDE.md §3).

## 4. The user's goals and where they stand (goals in the user's words: log §1)

| goal | state | log |
|---|---|---|
| binocular vision, several objects, near real time | construction no longer grows with combinations (24 built per 120 intervals together instead of 120) | §0 |
| cost must not keep rising | two objects over 400 intervals: `top_and_leaves` 30 -> 47 ms/interval (`top` 7 -> 10, `all` 55 -> 145); still rising slowly | §0f.2 |
| one thing in detail, the rest rough; focus to what is consequential | fovea input design tested; its expectation reads are worse than staying put; pull toward the consequential location +0.004 | §0c, §0d, §7.3 |
| expect where things go next | structure holds the next step exactly on exact loops; the field carries it only with `split`, which breaks context; on a bounce (both directions over the same places) neither mode helps | §0e.1, §0g.2, §0h |
| a "want" (explicitly not a reward) | not started; the user's design decision | §1 |
| Δt | not tested | §7.4 |
| action | reflex-taught gaze loop: the field takes the reflex's job and acts earlier; no reliable anticipation | §0g.3 |
| CUDA on the Fedora runner | works for integration; with `top` conduction the CPU is faster (not measured for `top_and_leaves`); what remains is per-incidence Python bookkeeping | §0b, §0e.5 |

## 5. What is established

Reads:
- Live activation right after an interval is mostly leftover (about 37% of the push). A pushed
  Nethra's activation swamps what structure adds: in the gaze loop the pushed motor Nethra holds ~0.3,
  structure adds 0.02-0.04 to the other one (log §0g.3). A read of a pushed Nethra is mostly its push.
- P cannot point at a Nethra already more active than its relation; small steps land on active cells
  (§0e.1). The free-step read (copy, evidence change off, step once with nothing pushed, compare the
  centre of activation) shows motion; P as a point does not.
- On a settled loop P reaches ~0.27 of M, so construction's gate is always open (§5.4).
- Lookups on topology done by the harness ("structural next") are not field reads (§0f.3).

Construction and conduction:
- Per-part subtraction on first sight with whole joining on recurrence keeps eye joining and context
  conjunctions and stops combination growth (§0).
- Displacement (delta) input makes the harness lookup exact and the field's expectation worse; real
  displacement does no better than shuffled; not adopted (§0f.3).
- `top_and_leaves` chosen from 8 rules: passes the cue and context tests `top` fails, at a third of
  `all`'s cost (§0f.2). Leakage kept at 1: 2 and 4 shrink every magnitude 10-1000x with no consistent
  gain (§0e.4).

Direction (`split`, log §0g, §0h):
- Evidence splits by timing without reading route sides: on a ring, p_k -> N_(k+1) (the member that
  comes before) 1.03, N_(k+1) -> p_k 0.22. The field's own next step moves forward in 24/24
  intervals (shared 8/24); positions shuffled in time 13/24.
- It loses co-present context: 15 regimes 0.93 -> 0.59, `robust` clean 0.83 -> 0.58.
- Mechanism (traced, 4 features / 6 contexts): in shared one incidence number lets a failed flow in
  one direction remove the other direction, and relations that hold an outcome with its context fill
  the outcome back in. In split the member -> relation evidence comes from the tension share, which
  measures whether the relation's own expectations came true, not whether the member came first. So
  members that don't discriminate (the cue present in every context, the next block's random
  features) keep driving relations. Reading the same split evidence symmetrically with the relation
  -> member evidence restores selection (0.95 vs 0.78).
- Eleven variants of which terms also move the other direction (log §0h.2): each trades direction for
  context; none keeps both. Closest: `op` (15 regimes 0.84, ring 21/24). Member -> relation evidence from the
  relation's own manifestation minus its inflow collapses (conduction dies).

Action (log §0g.3):
- Ablations caught two devices doing the work: an actuator reading the retina tracks with no
  structure at all; an actuator reading a motor Nethra that is pushed with the movement reads momentum.
- Reflex-taught loop (`gaze_loop.py`): during exposure a lagged reflex drives the eye and pushes the
  motor Nethra; at test the reflex is off, nothing pushes the motor Nethra, the actuator reads their
  activation. L=10 (mean distance eye-object): reflex 3.70, best fixed eye 2.25, split field 2.15;
  lesion 4.64, shuffled exposure 3.76, without eye-position Nethra 3.79. The field turns at reversals
  2-4 intervals after the object (reflex 4).
- Split vs shared in this loop is not settled: shared 4.62 after 300 intervals, 1.94 after 288.
- Caveat: exposure and test differ (motor Nethra pushed vs not), so at test closure never refinds
  Nethra holding motor Nethra. It is a habit replacing a reflex, not action with its own reason.

Method (mistakes of this session, not to repeat; older ones log §2, §10):
- judged a result against an optimum (a perfect actuator) instead of people;
- let a device do the work; caught only by the no-structure ablation;
- a wrapper edit that silently failed made a whole batch invalid (caught because outputs matched);
- conclusions from one run or one row were reversed later (log §0e.2 correction; shared "failing" in
  the loop);
- banned words in docs (training, teacher, capacity); fixed.

## 6. Next work

### 6.1 Open request: fix split's context loss
- Next measurement: per block, the evidence of B -> N_j (B = the member present in every context) and
  of each context feature -> N_j, shared vs split, on `regimes.py` with `P=4 K=6 BLK=40` (shared 1.00,
  split 0.73; 5-15 s). Question: why B's member -> relation conductance stays ~1.3 in split. People
  give a cue present on every trial little strength of its own (a sanity reference, not a template).
- What a fix needs: a member -> relation evidence term that measures the member coming before the
  relation's manifestation. The tension share does not measure that; the relation's manifestation
  minus its inflow collapses.
- Acceptance: context selection at shared's level (`regimes.py` P=4 K=6 and P=5 K=10, then
  `cue_capacity` 15 regimes) with split's forward direction on the ring (`direction_motion.py`);
  then the other capability scripts.
- Tools: `regimes.py`, `regime_trace.py`, `regime_flow.py`, `regime_probe.py`,
  `direction_variants.py` (env VAR), `with_direction_variant.py`.

### 6.2 Direction-aware `_admit_by_parts` (observed failure; construction only)
- `before_routes` / `after_routes` accept any route complete on either side. Routes carry no side;
  `relation_source_events[relation]` holds the witnessed (before, after) source pairs.
- Prediction to check first: symbolic bounce L=5 (`direction_bounce.py`) builds per interval
  `0111100000000111100...` today: the first leftward pass builds nothing, the second builds 4.
  A direction-aware accounting should build on the first leftward pass; streams that only go one way
  (ring) must stay bit-identical.
- It changes default construction: show before/after numbers and let the user decide.

### 6.3 Replicate the action loop
`gaze_loop.py` over start positions, L, LAT and exposure lengths, at least 5 runs each; report the
spread for split and shared before comparing them.

### 6.4 Action beyond imitation (user decision)
The loop needs something that makes some states matter (a "want", explicitly not a reward; log §1).
Ask the user before designing it.

### 6.5 Smaller items
- Remove `_registering_before` (code change: run `bitcheck_cores.py`).
- A human-reference line with a source for each capability script.
- Older open items: Δt (log §7.4), coarser receptive Nethra for earlier anticipation (§9.3),
  fovea/gaze design (§9.2, §0c, §0d), array storage of evidence (roadmap 7, §0e.5).

## 7. Tools

Scripts in `nethra/tests/` (per-script details in the log's script tables):

| group | scripts |
|---|---|
| capability tests (the user's; rerun only when asked, CLAUDE.md §2.4) | `cue_capacity.py`, `context_partwise.py`, `human.py 14`, `robust.py`, `focus_symbolic.py`, `consequence_reach.py 8 0.95 10 40 10 0` |
| wrappers | `with_params.py` (env LEAK, COND, DIR), `with_direction_variant.py` (env VAR plus those), `with_conduction_variant.py` |
| direction | `direction_ring.py`, `direction_motion.py`, `direction_bounce.py`, `direction_variants.py` |
| context trace | `regimes.py`, `regime_trace.py`, `regime_flow.py`, `regime_probe.py`, `context_trace.py`, `cue_trace.py` |
| action | `gaze_loop.py`, `gaze_motor_probe.py`, `gaze_retina_device.py` (discarded design) |
| loops, where next | `ring_symbolic.py`, `ring_graded.py`, `ring_split.py`, `ring_trace.py`, `delta_input.py`, `delta_runlength.py`, `delta_field.py` |
| binocular, focus | `binocular*.py`, `focus_fovea.py`, `partwise_prototype.py` |
| cost, execution | `conduction_cost.py`, `exec_stream.py`, `exec_prof.py`, `exec_parts.py`, `bitcheck_cores.py`, `gpu_field.py`, `gpu_bench.py` |

- Runtimes: shared capability scripts 5-140 s; split 2-4x slower (`cue_capacity` split ~510 s, over
  the limit: start with small `regimes.py` cases); `gaze_loop.py` 10-20 s.
- Bit-identity: `git show <commit>:nethra/nethra.py > old.py`, then
  `python3 nethra/tests/bitcheck_cores.py old.py nethra/nethra.py` (~2 min, prints ALL IDENTICAL).
- Fedora runner (self-hosted: RTX 5070, cupy, numba, torch, Python 3.14 at `/usr/bin/python3`):
  a workflow whose `on: push: paths:` matches the pushed files, or `workflow_dispatch`; read job logs
  or artifacts (log §0b, §8).

## 8. Human references used (with sources)

Pursuit and recall numbers come from abstracts and search snippets; the full texts were not
reachable from the container.
- Pursuit starts ~100-130 ms after the object moves, saccades ~200-250 ms: Frontiers in Systems
  Neuroscience 2013, 7:4 (cognitive processes in smooth pursuit).
- On periodic motion the eye starts reversing before the object after 1-2 cycles: Barnes & Asselman
  1991, J Physiol ("The mechanism of prediction in human smooth pursuit eye movements"); J Neurosci
  2009, 29(42):13302.
- Sequences are recalled forward about twice as often as backward: Kahana 1996, Memory & Cognition
  (lag-CRP); Howard & Kahana 2002.
- Pairs studied together are recalled both ways about equally: Kahana 2002, Memory & Cognition 30:823.
- Spaced exposure is retained better than massed: Cepeda et al. 2006, Psychological Bulletin 132:354.
- Stimuli once tied to reward capture attention: Anderson, Laurent & Yantis 2011, PNAS 108:10367.
- Only ~1-4% of cortical neurons strongly active at once: Lennie 2003, Curr Biol 13:493; expected
  input gives lower, sharper activity: Kok, Jehee & de Lange 2012, Neuron 75:265 (log §0b).
- No source recorded in this repo for: blocking, retroactive interference and slower combination-only
  (XOR) learning in people (`human.py`); retinal adaptation (log §0c).
