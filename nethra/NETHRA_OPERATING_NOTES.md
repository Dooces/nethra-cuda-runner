# Nethra operating notes

Read this before feeding, reading, testing, or changing `nethra.py`. Every section below exists
because the opposite was assumed at some point and cost days of work.

## 1. What Nethra is

- One population of ordinary **Nethra**. Sensory inputs, actuator bindings and every constructed
  relation are the same type and obey the same equations. There is no input layer, output layer,
  hidden state, label, target, or decoder.
- A **field**: every Nethra has an activation `a`. Every earned incidence conducts in both
  directions (with `conduction`, covered constructed route members never earn evidence, section 4) with conductance `g(e) = g_max (1 - exp(-e / tau))`, where `e` is that incidence's
  evidence. Contributions add. Leakage is the only sink.

      C da_i/dt = J_i - leak * a_i + sum_j g_ij (a_j - a_i) + Gamma_i

  `J` is whatever current was pushed this interval. `Gamma` is the F61 convergence redistribution:
  it moves charge toward a receiver when several independent suppliers feed it at once. It sums to
  zero.
- **Structure**: a Nethra has support routes, each an unordered set of other Nethra.
  - **Closure** refinds every Nethra one of whose routes is completely present in the current
    description, recursively.
  - **Construction** happens only from unresolved residual. It first reuses every compatible
    existing Nethra, and admits one weak new Nethra only when existing structure does not account
    for the experience. With `join_on_recurrence` (default) a transition of pushed Nethra seen for
    the first time is accounted per part and only the unaccounted remainder is joined; a
    transition that recurs is joined whole. The new Nethra's before side (the previous interval's closure) and after
    side (this interval's closure) are separate routes.
- **Evidence change**: after each interval the field is integrated twice from the same start, with the
  real input and with zero input. The difference is what the input actually caused (`M`).
  - Prior incidence flows `P` are what the field was already carrying toward each Nethra.
  - Each incidence's evidence moves by the mismatch `epsilon = M - P`: an outgoing term plus a
    shared incoming term.
  - Residual traces feed F61's independence weights.

## 2. What Nethra is not (each of these was tried and was wrong)

| Wrong assumption | Why it is wrong |
|---|---|
| It predicts a next token; score it with accuracy, argmax, or C-vs-D | There is no output. The live field is the expectation. Equal priming of two continuations that were equally experienced is the correct state. |
| Context is a hidden state; the field should keep a symbol whose support is gone | Context lives in Nethra whose support is present. If the support left, 50/50 is right. |
| Leftover activation is noise; subtract a never-learned field | Leftover activity is the current regime and part of the expectation. A field with no earned structure is only a reference for how much construction and evidence added. |
| Recognition may gate conduction (only refound Nethra conduct) | Changes the field law. All earned incidences conduct. |
| A multiplier on `g` from co-activation or "resonance" | Changes the field law and adds an overlap detector. |
| Signed {-1, 0, +1} presence-change events in the construction description | Destroys magnitude; the contract keeps exact graded evidence. |
| Push -1 on a symbol when it leaves (delta feed) | That is a real negative current. It drags the leaving Nethra and everything it touches negative. Departure is simply not pushing; leakage handles it. |
| Evidence is stuck, so add a conductance floor (`g_min > 0`) | Adds a second conduction law. The real cause was the admission seed (section 4). |
| Other models' claims about the code | Several were wrong (for example, "refound only through X"). Check every claim in code before using it. |

## 3. How to feed it

- Push a number onto each Nethra that is present in this interval, then `step(dt)`. Repeat.
  One interval per observation. Nothing else is fed: no labels, no history window, no targets.
- Graded inputs use overlapping receptive Nethra, each pushed with its share of the value.
- Co-present facts (a feature and the object it belongs to, or a context that persists through a
  block) are simply pushed together. That is the legal way for context to exist.
- Construction joins consecutive intervals. A silent interval (nothing pushed) leaves the before
  side empty, so nothing is built across a gap.

## 4. Parameters that matter

| Parameter | Tested value | Effect |
|---|---|---|
| `admission_seed` | 14 (default) | Starting evidence of a new Nethra. At 0.01 the equations sit at a fixed point: a new Nethra's conductance is about 1.5e-4, its activation never exceeds its members, all flow runs member-to-relation, and both evidence-change terms stay zero forever (measured over 15,000 intervals). It must be large enough that `g(seed)` is comparable to the leak. Lower values (about 5) follow recent regimes and switch fast; higher values (14 to 50) keep accumulated structure and resist switching. |
| `g_min` | 0 (default) | Keep 0. `g(0) = 0`. |
| `source_similarity_threshold` | 0.999 (default) | Structural recurrence of graded source patterns: a new pattern refinds the stored pattern with the highest cosine above this (earliest among equals); otherwise it is stored. Physical current stays exact. Cosine ignores overall size, so one Nethra pushed 1.0 and 1.25 is the same structural event. Without this, graded input never recurs exactly and construction adds a Nethra every interval (a nested chain while the pushed members stay the same). |
| `leakage`, `capacitance` | 1, 1 | Field timescale. One interval of `dt = 1` per observation. |
| `join_on_recurrence` | True (default) | Construction per part on first sight of a transition of pushed Nethra, whole-interval joining once the same transition recurs. Independent objects then no longer build a Nethra per joint state; eyes of one object and contexts that recur are still joined (one occurrence later). False = whole-interval joining every interval (previous behaviour, bit-identical; old checkpoints load with False). Keeps a set of witnessed transitions (bookkeeping, grows with distinct transitions). See docs/HANDOFF.md section 0. |
| `conduction` | `"top_and_leaves"` (default since 2026-09-25) | Which members of a newly registered route earn incidence evidence. A constructed member lying in a complete route of another member of the same route (covered) earns none (g = 0); primitive members always earn. `"top"`: primitive members can be covered too (flat cost, but in a continuous stream the before side of every transition is covered by the previous transition's Nethra, so the present drives what comes next only through how it got there; context and cue tests fail). `"all"`: every member (old checkpoints load with it). Routes stay whole, so closure and construction are the same in all three. Measurements: docs/HANDOFF.md §0f. The keyword `top_only_conduction=True/False` still maps to `"top"`/`"all"`. |
| `direction` | `"shared"` (default) | `"split"` (added 2026-09-25 on user request: direction and timing): each incidence has two conductances, relation -> member and member -> relation, each with its own evidence; flow j -> i = g_ji max(0, a_j - a_i), so with equal conductances it is the shared law. Both start at the admission seed. The outgoing evidence term moves relation -> member evidence, the incoming term member -> relation evidence, so which direction strengthens is decided by whether flow came before a manifestation (timing), not by before/after routes. Integration is RK4 (piecewise linear). With evidence change off it equals `"shared"` exactly. Measurements: docs/HANDOFF.md §0g. |
| `integrator` | `auto` | Runtime only. ETD (exact passive part, eigendecomposition) up to 400 Nethra, RK4 with stiffness subdivision above that. Both integrate the same equation. |

## 5. How to read it

- **Live activation** of any Nethra right after an interval, before the next input. That is the
  expectation. Report activations, or shares among the Nethra of interest. Do not pick a winner and
  call the rest wrong unless the world is deterministic.
- **Closure** (`previous_closure`) shows which constructed Nethra are refound, and therefore which
  regimes are structurally present.
- **Topology**: every constructed Nethra and its routes can be listed directly (see `strengths.py`).

## 6. Tested capabilities (numbers from the test scripts)

| Capability | Result | Script |
|---|---|---|
| Probability matching | P(B) 0.5 / 0.75 / 0.9 gives B share 0.50 / 0.63 / 0.84 | `capability.py` |
| Regime switching | Adopts a new regime within about one block (seed 5) | `capability.py`, `explore.py` |
| Persistent-context regime selection | Correct continuation top 100% for 2, 4, 8 and 16 regimes | `explore2.py`, `scale2.py` |
| Overlapping contexts | 8 regimes from shared features: 1.00 / 1.00 / 0.94 at 11% / 39% / 61% overlap; 15 regimes at 57% overlap: 0.83 | `cue_capacity.py` |
| Same cue, new answer, **with** a differentiating cue | Both answers kept; each cue selects its own (about 1.6 to 1) | `cue_capacity.py` |
| Same cue, new answer, **no** cue | Recent answer dominates (retroactive interference, as in people) | `human.py` |
| Old regimes kept | Old regimes 100% after 400 blocks of only new ones | `strengths.py` |
| Few-shot | New regime top after 2 exposures, old regimes unaffected | `strengths.py` |
| Pattern completion | A partial group primes its missing member first | `explore.py` |
| Blocking / spacing | Blocking at seed 14; spacing effect at seed 5 | `human.py` |
| Simple vs combination-only context | Simple built fast; XOR slow and unstable | `human.py` |
| Bounded construction | One Nethra per distinct experienced transition, then reuse | `scale.py` |
| Exact checkpoint continuation | Bit-identical after reload | fix tests |

## 6b. Robustness and confidence (`tests/robust.py`, `tests/familiarity.py`)

- Setup: 12 regimes, each context a pair from 10 features. Probes use the full context, one
  feature only, the context plus a distractor, or one feature plus two distractors.

  | Method | Clean | Partial | Noisy | Noisy2 |
  |---|---|---|---|---|
  | Nethra | 0.83 | 0.38 | 0.50 | 0.04 |
  | Count table | 0.75 | 0.38 | 0.58 | 0.08 |
  | Rescorla-Wagner | 1.00 | 0.38 | 0.58 | 0.08 |
  | Configural lookup | 1.00 | 0.00 | 0.00 | 0.00 |

  Nethra degrades gracefully, like the linear learners, where configural lookup collapses. It is not
  more robust than them.

- **Structural recognition** comes from closure: the constructed Nethra refound at the moment of reading.

  | Probe | Constructed Nethra refound |
  |---|---|
  | Complete known context | 2 |
  | Partial context | 0 |
  | Never-seen combination | 0 |
  | Context plus distractor | 3.4 |

  So the same field that gives graded expectations also gives crisp recognition of whether a known
  configuration is fully present. A count table has no such signal; a lookup table has no graded
  one.

- Candidate confidence readouts:
  - Structural recognition (above).
  - Concentration of priming among candidates. It tracked the odds in the two-way test
    (0.50 / 0.63 / 0.84), but is flat across 12 candidates.
  - Total priming (familiarity). It drops with partial input and rises with distractors.
  - Post-outcome residual. It is dominated by the raw manifestation scale and is not usable yet.

## 7. Known limits and open points

- **Relation to textbook learners (`tests/baselines.py`):**

  | Method | 15 overlapping regimes | Old set kept after 400 new-only blocks | Blocking ratio | XOR |
  |---|---|---|---|---|
  | Count table | 0.83 | 0.00 | 1.00 | 0.51 |
  | Rescorla-Wagner | 0.49 | 1.00 | 0.00 | 0.50 |
  | Configural lookup | 1.00 | 1.00 | n/a | 1.00 |
  | Nethra | 0.83 | 1.00 | 0.40 | 0.43-0.65 |

  A blocking ratio of 1.00 means no blocking; 0.00 means complete blocking. No single simple learner
  shows Nethra's combination. A configural lookup table beats it on raw accuracy everywhere tested
  here. Its value has not yet been shown on raw capability. It is a distinctive, human-like profile
  in one field.

- **Frontier (optional, declared approximation):** `frontier_tolerance > 0` integrates only Nethra
  with |a| >= tolerance or source current, plus their one-hop incidence halo. Everything else decays
  by leakage alone for that interval and takes no part in evidence change or construction. Binocular world, 66 inputs,
  277 Nethra (`tests/frontier_test.py`):

  | Setting | ms/interval | Nethra integrated | mean abs(a - exact) | Next position hit | 8 regimes correct |
  |---|---|---|---|---|---|
  | exact | 41.5 | 275 | 0 | 0.41 | 1.00 |
  | 1e-3 | 17.2 | 136 | 5.8e-4 | 0.39 | 1.00 |
  | 1e-2 | 10.8 | 92 | 8.3e-4 | 0.40 | 1.00 |

  The adaptive setting (`frontier_min`) moves the tolerance down when the interval's source was
  unexplained. It currently stays near its low end, because raw manifestation is almost always far
  larger than prior flow, so "unexplained" is near 1. A usable effort signal needs a better
  surprise measure first.
- **Without the frontier, cost is not bounded by the active set.** Every interval integrates and moves evidence over the
  whole Nethra population.
  - Activation never becomes exactly zero, so the active set cannot be cut exactly.
  - In the binocular world (66 inputs, `binocular.py`) about 125 Nethra are active above 1e-3,
    while cost follows the total (286 Nethra: 55 ms per interval).
  - Bounding cost at the frontier requires a numerical tolerance (treat activation below some
    epsilon as zero). That is an approximation of the field and must be declared as one.
- ETD's eigendecomposition is O(N^3) per interval. `auto` switches to RK4 above 400 Nethra.
- Construction does not bridge silent intervals.
- Combination-only (XOR) contexts are weak, which is plausibly human-like.
- **Holding an expectation across a gap (`tests/waiting.py`).**
  - Nothing in the field law sustains activity, and construction only joins consecutive intervals.
  - A cue therefore gets linked to an outcome several intervals later only if its support is still
    present when the outcome arrives: rehearsal.
  - Result, `topology_and_evidence_change` off, 4 random fillers between cue and outcome:

    | Condition | Cue's own outcome primed more | Nethra joining cue and outcome | ms/interval |
    |---|---|---|---|
    | No rehearsal | 0.35 | 1 | 11 |
    | Cue re-pushed at a steady 0.5 | 1.00 | 7 | 24 |
    | Same, gap varying 2 to 7 intervals | 1.00 (no timer needed) | 7 | 27 |
    | Re-push scaled by the cue's own activation (loop back) | fails | — | — |

    The loop-back fails because its amplitude changes every interval, so the source never recurs
    exactly. With exact reuse that creates a new Nethra every interval; with graded support it gives
    0.45, no better than chance.
  - The harness currently decides what to rehearse; a native way to do that is open.
