# Nethra roadmap

Built from `nethra/nethra.py` (one-file core), `nethra/NETHRA_OPERATING_NOTES.md` and the tests in
`nethra/tests/`, as of 2026-09-25. Read the operating notes and the mistake ledger before starting
any task. Every task respects the contract in the notes: one Nethra type, no winner/selector, all
learned incidences conduct, no gating or multiplier on `g`, `g(0) = 0`, departure is not a
negative push. Per the ledger (2026-09-23), validation is one-off and disposable unless the user
explicitly asks for a standing test file or workflow.

## Why this order

1. **Trust the numbers first.** Done (Phase 0 below): the core reproduces, is deterministic,
   resumes exactly, and obeys its own equation.
2. **Measure before adding mechanism.** Adaptive frontier, native rehearsal and confidence all
   stall on the same missing piece: a usable surprise signal.
3. **Keep the population bounded.** Phase 0 showed that 10% amplitude noise makes `"exact"` source
   support build one new Nethra every interval. Every later cost and real-data task depends on
   fixing that, so graded recurrence moved up from Phase 3 to 1.4.
4. **Make it cheap before making it big.** Cost follows total population, not active set.
5. **Fix known capability gaps next**: gap-holding, then combination contexts.
6. **Then scale to real streams and closed-loop action.**

| # | Task | Depends on | Why now |
|---|---|---|---|
| 0 | Reproducibility check | — | Done, results below |
| 1.1 | Normalized surprise readout | 0 | Unblocks 1.2, 2.2, 3.1 |
| 1.2 | Confidence calibration study | 1.1 | Decide which readout is usable |
| 1.3 | Admission-seed map | 0 | Later tasks need a chosen seed |
| 1.4 | Graded source recurrence evaluation | 0 | Noisy input grows N by one per interval |
| 2.1 | Profile + remove O(N) Python loops | 1.4 | Cheapest speedup, zero semantics |
| 2.2 | Adaptive frontier on new surprise | 1.1, 2.1 | Bound cost by active set |
| 2.3 | Array backend (CuPy) for RK4 path | 2.1 | Large N on the GPU runner |
| 2.4 | Krylov exact passive step for N > 400 | 2.3 | Keep ETD accuracy past 400 |
| 3.1 | Native gap-holding (loop-back via receptive Nethra) | 1.1, 1.4 | Biggest structural limit |
| 3.2 | Combination (XOR) contexts | 1.3 | Configural lookup beats Nethra |
| 4.1 | Stochastic sequence world, log-loss scoring | 1.4, 2.x | First real-stream capability test |
| 4.2 | Market stream replay (AAPL branches) | 4.1 | Existing data, honest baselines |
| 4.3 | Closed-loop feeding / hand-ball revisit | 2.x, 3.1 | Action, memory-bounded |
| 4.4 | Long-horizon persistence + checkpoint size | 4.x | Run for days, resume exactly |

---

## Phase 0 — Reproducibility (done 2026-09-25, one-off checks, no standing files)

Layout: `nethra/nethra.py`, notes, ledger (repo copy, last entry 2026-09-24), `nethra/tests/` with
the `nethra_presence` / `nethra_etd` shims, `robust.py`, `smear2.py`. The imports work through the
shims.

| Check | Result |
|---|---|
| `robust.py`, Nethra rows (baselines stubbed: `baselines.py` not in repo) | clean 0.83, partial 0.38, noisy 0.50, noisy2 0.04: matches notes 6b exactly; 86 s |
| `smear2.py`, a(v+1)/a(v-1) for v = 1, 2, 3 | seed 5: trained 1.006 / 0.960 / 0.776; seed 14: 0.941 / 0.979 / 0.952; never-learned 0.525 / 0.493 / 0.471. Learning roughly doubles forward priming but v+1 does not exceed v-1 |
| Determinism: same inputs twice, N = 191 | bit-identical activations and checkpoint |
| Checkpoint save at 120 intervals, reload, 120 more, N = 251 | bit-identical to uninterrupted run |
| Charge balance `d(sum a)/dt = sum J - leak sum a` (so Gamma sums to zero), learned field, 30 intervals | ETD rel err 7.8e-15; RK4 9.2e-9 |
| ETD vs RK4, one interval, N = 191 | max abs diff 1.1e-5 (max activation 0.097) |

Findings:
- **Unbounded construction under noise.** 180 intervals, 12 inputs: constant amplitudes build 24
  Nethra; the same world with 10% amplitude noise builds 179 (one per interval) under `"exact"`,
  44 under `"min"`, 46 under `"product"`. This is the contract working as stated (every noisy
  transition is distinct), but it makes `"exact"` unusable for real inputs. Hence 1.4.
- **Default ETD is not the most accurate integrator.** Against ETD with 64 pieces: ETD 2 pieces
  (default) 3.1e-5, ETD 8 pieces 4.3e-7, RK4 default (8 substeps) 1.2e-5, RK4 32 substeps
  6.8e-8. ETD's passive part is exact, but the stepped F61 convergence term dominates its error at
  `etd_pieces=2`. Small in absolute terms; matters only if two integrators are compared to 1e-5.
- **Checkpoint omits the adaptive frontier tolerance** (`_tol`). Exact resume holds only when
  `frontier_min` is unset.
- Not rerunnable yet: the other scripts the notes cite (`baselines.py`, `capability.py`,
  `explore*.py`, `scale*.py`, `cue_capacity.py`, `human.py`, `strengths.py`, `familiarity.py`,
  `waiting.py`, `frontier_test.py`, `binocular.py`) are not in the repo.

Not done, and only on explicit request (ledger 2026-09-23): a standing test runner, invariant test
files, CI workflows. Multi-seed reporting (10+ seeds, mean and 95% interval) is a per-task rule
instead of a harness.

## Phase 1 — Readouts before mechanism

### 1.1 Normalized surprise readout
- **Problem (notes §7):** "unexplained" is almost always near 1, because raw manifestation `M` on
  source Nethra is dominated by the injected current itself, which no prior flow `P` can match.
- **Options (read-only, no field change):**
  1. **Expectation match (recommended first):** notes §5 say the live field before input *is* the
     expectation. Surprise = `1 - cos(a_before, J)` over Nethra that can receive source, or the
     share-weighted version `1 - sum_i J_i * a_i / (|J| * sum a)`. Uses only quantities that
     already exist.
  2. **Internal-only residual:** `sum max(0, eps_m) / sum M_m` over Nethra *without* source
     current, so injected current doesn't swamp the ratio.
  3. **Shape not magnitude:** `1 - cos(P, M)` over all Nethra.
- **Why first:** frontier adaptation (2.2), rehearsal (3.1) and confidence (1.2) all need it.
- **Test:**
  - Deterministic world after learning: surprise near 0.
  - Probability-matching world (P(B) = 0.5 / 0.75 / 0.9): mean surprise on outcome X correlates
    with `-log p(X)`; compare to a count-table surprisal.
  - Regime switch: spike on the first block, decay within about one block (seed 5).
  - Pick the option with highest correlation to count-table surprisal and cleanest switch spike.

### 1.2 Confidence calibration study
- **What:** score each candidate readout from notes 6b (structural recognition, share, familiarity,
  1.1 surprise) by AUROC for "correct continuation is top" on the `robust.py` probes, plus
  reliability curves.
- **Why:** 6b lists readouts but not whether they predict being right. A calibrated readout is what
  a caller (or later a native policy) needs to know when to trust the field.
- **Test:** report AUROC per readout per condition, 10 seeds. A readout is "usable" if AUROC > 0.75
  in clean and partial conditions. Record result in the notes either way.

### 1.3 Admission-seed map
- **What:** sweep `admission_seed` in {3, 5, 8, 14, 25, 50} over the full harness.
- **Why:** seed trades switching speed (5) for retention (14–50); blocking appears at 14, spacing at
  5. `g(5) = 0.07`, `g(14) = 0.20`, `g(50) = 0.59` vs leak 1. Later tasks need a declared default
  per world type, and the tradeoff should be one chart, not folklore.
- **Test:** Pareto table: regime-switch latency vs old-regime retention vs few-shot. Pick defaults.
  No mechanism change.

### 1.4 Graded source recurrence evaluation
- **What:** `source_support="min"` and `"product"` exist but are marked experimental. Run the full
  harness on both vs `"exact"`, plus `robust.py` partial/noisy probes and a real-valued world
  (receptive-field encoded values with noise).
- **Why:** real inputs never recur bit-exactly; with `"exact"` construction grows by one Nethra
  per interval under 10% amplitude noise (Phase 0). Also the only path to better partial/noisy robustness
  (currently 0.38 / 0.50, same as linear learners).
- **Test:** Nethra count per 10k intervals (bounded?), accuracy on robust probes, noisy real-valued
  world. Promote one mode to default only if it matches `"exact"` on the clean harness.

## Phase 2 — Bounded cost

### 2.1 Profile, then remove O(N) Python loops
- **Where:** per interval, `step` scans all Nethra for source current; `_native_learn` builds
  `epsilon` over all Nethra; `update_residuals` updates `rho` for all Nethra; `step` zeros
  `external` for all Nethra.
- **How:** keep a set of Nethra with pending external current (`push` registers); compute
  `epsilon` only where `target` or `predicted` is nonzero (absence = 0 already); keep `rho` as a
  numpy array indexed by creation order and decay it lazily (store last-touched step, apply
  `lam ** k` on read). All exact, no semantic change.
- **Why:** cheapest speedup, zero risk to the law; makes frontier gains real rather than eaten by
  bookkeeping.
- **Test:** Phase 0 checks (determinism, checkpoint bit-identity, charge balance) still pass; `scale.py` ms/interval vs N
  before and after.

### 2.2 Adaptive frontier on the new surprise
- **What:** feed 1.1's surprise into the existing `frontier_min` rule instead of the near-1
  "unexplained" ratio.
- **Why:** frontier already gives 2.4x–3.8x at 1e-3/1e-2 with no accuracy loss; the adaptive part
  is stuck because its input signal is flat.
- **Test:** rerun `frontier_test.py` table; add surprise-driven row. Pass: mean `|a - exact|`
  ≤ fixed 1e-2 error, 8 regimes still 1.00, ms/interval lower than fixed 1e-3, tolerance visibly
  widens on regime switches.

### 2.3 Array backend for the RK4 path
- **What:** `_derivative_compiled` is already pure array code (`bincount`, gathers). Parametrize
  the array module (`xp = numpy | cupy`) for `_compile_interval`, `_derivative_compiled`,
  `_rk4_interval`.
- **Why:** it's the hot loop above 400 Nethra and maps directly to the GPU on the Fedora runner.
- **Care:** GPU `bincount`/atomics are not order-deterministic. Keep numpy as the reference and
  bit-exact checkpoint path; declare GPU as tolerance-equivalent, or use sorted segment sums
  (`cupy` `reduceat` on pre-sorted indices) for determinism.
- **Test:** GPU vs numpy max abs diff < 1e-10 over 10k intervals of `binocular.py`; speed table
  at N = 300, 1k, 5k, 20k. Only worth it once N > ~2k (transfer overhead); measure the crossover.

### 2.4 Krylov exact passive step for N > 400
- **What:** ETD uses dense `eigh` (O(N^3)), so `auto` drops to RK4 above 400. Replace with
  `scipy.sparse.linalg.expm_multiply` (or a Lanczos phi-function action) on the sparse Laplacian.
- **Why:** keeps the exact passive part at any N; RK4 substeps grow with max weighted degree
  (stiffness), Krylov cost doesn't.
- **Test:** vs dense ETD at N ≤ 400 (diff < 1e-10), vs RK4 at N = 2k; ms/interval curve.

## Phase 3 — Capability gaps

### 3.1 Native gap-holding
- **Problem (notes §7, `waiting.py`):** nothing sustains activity; construction only joins
  consecutive intervals. Steady re-push works (1.00) but the harness picks what to rehearse.
  Loop-back by own activation fails because the amplitude never recurs exactly.
- **Recommended option:** loop back through **overlapping receptive Nethra** — the legal graded
  input from notes §3. An actuator reads a Nethra's activation each interval and pushes it to a
  small bank of receptive Nethra (e.g. 4–8 overlapping triangular bins over the activation range),
  each with its share. Similar amplitudes land on the same receptive Nethra, so the source
  *does* recur and construction reuses it. The actuator is just a physical binding (the core
  already allows any actuator to read any Nethra); no selector, no threshold in the core.
- **Alternatives:** (a) graded `source_support="min"/"product"` on raw loop-back — notes measured
  0.45, record as rejected unless 1.4 changes it; (b) a fixed-amplitude echo on every source
  Nethra (harness policy, not native — keep only as a control).
- **Why here:** biggest structural limit; needs 1.1 to judge whether rehearsal reduces surprise.
- **Test:** `waiting.py` table with a new row. Pass: cue's own outcome primed more ≥ 0.9 for gaps
  2–7, no harness-chosen rehearsal target, construction count bounded (no new Nethra per
  interval), ms/interval reported.

### 3.2 Combination (XOR) contexts
- **Problem:** XOR 0.43–0.65; configural lookup 1.00. Yet closure already recognizes complete
  known configurations crisply (6b: 2 refound for full context, 0 for partial).
- **Investigate before changing:** log, for XOR trials, the conjunctive Nethra's activation vs its
  members', its per-incidence outgoing evidence, and how the provisional MAX route summary sets
  `relation.routes[route][signature]`. Hypothesis: the member-level (linear) paths win the flow and
  the conjunctive Nethra's outgoing evidence is diluted by the shared incoming term.
- **Options (only after a logged failure, per the file header):**
  1. Replace the MAX route summary with the observed-failure-justified alternative (e.g. mean or
     sum) — the header explicitly allows this given a concrete failure.
  2. Tune `eta_out / eta_in` ratio (parameter, not mechanism).
  3. Higher seed for constructed Nethra only when admitted from a multi-member route — mechanism,
     last resort.
- **Test:** `human.py` XOR ≥ 0.85 over 10 seeds, and no regression on blocking ratio, continual
  learning (old regimes 1.00 after 400 blocks), 15-regime overlap (0.83). Ledger entry either way.

## Phase 4 — Real streams and action

### 4.1 Stochastic sequence world, log-loss scoring
- **What:** character or token stream with known transition probabilities (order-1 to order-3
  Markov, then a small natural-text corpus).
- **How to score (notes §2 forbid argmax):** normalize live activation over candidate Nethra into
  a distribution and report log-loss / calibration vs count n-gram and Rescorla-Wagner baselines.
  Probability matching (notes §6) says shares should track odds; this tests it at scale.
- **Why:** first test where Nethra must hold context longer than one interval (uses 3.1) on
  graded, noisy input (uses 1.4), at N in the thousands (uses Phase 2).
- **Test:** log-loss within 10% of the best n-gram of matching order; construction bounded;
  continual learning when the source switches corpora.

### 4.2 Market stream replay
- **What:** reuse the AAPL branches' cached data with the 4.1 scoring: shares over up/down/flat
  receptive Nethra, log-loss and calibration vs persistence and frequency baselines, strictly
  chronological, no lookahead.
- **Why:** real, non-stationary, noisy; the right honesty test. Expect near-baseline; the value is
  in calibration and regime-switch behaviour, not raw accuracy.
- **Test:** 10k chronological intervals, 5 non-overlapping windows, baselines on identical bytes
  (hash recorded in receipt).

### 4.3 Closed-loop feeding / hand-ball revisit
- **What:** rerun the reward-free feeding emergence experiment with Phase 2 cost bounds and a
  memory cap on the run; action via actuator reads of ordinary Nethra (no selector).
- **Why:** these runs were repeatedly cancelled for OOM and staleness; with bounded cost and
  receipts they can finish.
- **Test:** peak RSS below cap over 120k steps × 16 lineages; persistence metric per lineage;
  compare against a no-learning control with identical seeds.

### 4.4 Long-horizon persistence + checkpoint size
- **What:** days-long runs with periodic checkpoints; measure checkpoint size growth, reload time,
  and bit-identical resume (as in Phase 0) at scale. Add compact binary checkpoint (npz) alongside JSON if
  JSON size dominates.
- **Test:** 1M intervals, resume at 10 random points, bit-identical continuation; size vs N curve.

---

## Working rules for every task
- Read notes + ledger first; append a ledger entry for any mistake found.
- One change per branch; results (numbers, seeds, core sha256) in the commit message or PR.
- No task may change the field law, add a selector, or gate conduction. If a task seems to need
  that, stop and write the observed failure down first.
- Report results as intervals over seeds and against the baselines in `baselines.py`.
