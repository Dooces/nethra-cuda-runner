# Nethra roadmap

Source of truth: `nethra/nethra.py` and `nethra/NETHRA_OPERATING_NOTES.md`. Read the notes first.
`nethra/NETHRA_MISTAKE_LEDGER.md` in this repo is an old copy, kept for reference only.

## Ground rules

- Nethra is not a learner, predictor, network or model. Don't use ML terms or methods on it:
  no loss, accuracy, argmax, calibration, training/validation, gradient, batch, or tuning
  parameters toward a score.
- Readouts are only what the field already has:
  - live activation right after an interval;
  - closure (which constructed Nethra are refound);
  - topology and incidence evidence.
- The field law does not change. All earned incidences conduct. No gate or multiplier on `g`,
  no selector or winner, no new participant type, departure is not a negative push.
- No problem is opened without an observed failure, a contradiction or a request.
- Textbook methods (count table, Rescorla–Wagner, configural lookup in `tests/baselines.py`) are
  reference points for describing Nethra's profile. They are not targets.
- Checks are one-off and disposable unless a standing test is asked for.

## Next work, in order

### 1. Sub-pattern transfer
- **Question:** does recursive structure let a constructed chunk be reused in a new context, which
  flat pairwise associations can't do?
- **Design constraint:** flat pairwise associations must not be able to pass it.
  - A completed Q R S is followed by T.
  - Q R X is followed by U.
  - The deciding part is present in the input, not held across a gap.
  - Then check whether a new context takes up the constructed chunk faster than a novel chunk.
- **Read:**
  - live activation of T and U after the chunk;
  - whether the chunk's constructed Nethra is refound (closure) in the new context.
- **Reference points:** the three textbook methods in `tests/baselines.py`. That file is not in
  this repo yet.

### 2. Structural recognition
Structural recognition is already derivable from closure, with no new code:
- a complete known context refinds its constructed Nethra;
- partial or never-seen input refinds none;
- extra refinds show overlap with other known configurations.

The distinction worth recording: the same field gives graded activation and exact recognition of
a complete known configuration. Document where each shows up. Don't build a separate evaluator.

### 3. Knowing what to hold
- **Today:** holding re-pushes a Nethra across a gap.
  - `tests/spare_hold.py`: holding every cue 0.90, never holding 0.55, holding whatever just
    became present 0.60.
  - Holding already pauses itself while construction is active.
  - Choosing what to hold is what's missing.
- **Next:** mark a hold as paid off when the held Nethra became a member of a Nethra that is
  later refound. Prefer onsets whose past holds paid off.
  - Uses only closure and construction.
  - Where that record lives must be decided without adding a participant type or selector
    inside the core.

### 4. Replay during quiet periods
Re-present recent sequences when nothing new is being built, using the same construction-quiet
trigger. Check whether it consolidates the delayed links.

### 5. Holding as an actuator
Rehearsal becomes an ordinary Nethra whose activation is read and fed back as input. Holding then
becomes something the field's own structure carries, not a harness policy. This is the bridge to
outputs.

### 6. Residual scale
- **Observed:** what the input causes directly (`M`) is far larger than prior flow (`P`).
  - The unexplained fraction therefore sits near 1.
  - The adaptive frontier (`frontier_min`) stays at its low end.
- **Scope:** open only as far as observed. Don't invent a surprise metric.

### 7. Array-based evidence storage (execution only)
- Arrays instead of dicts for the same per-incidence evidence.
- Same local updates, once per interval, in the same order.
- Topology still grows by construction; arrays are reindexed when it does.
- Not gradient descent, a loss, batches or a fixed parameter vector.
- **Acceptance:** checkpoint hashes and every test in `tests/` match the current code to rounding.
- Other execution-only work falls under the same acceptance rule, for example the per-interval
  loops over all Nethra in `step`, `_move_evidence_and_construct` and `update_residuals`.

### Frontier
Fixed tolerance stays as a declared approximation until item 6 gives the adaptive tolerance a
usable signal.

## Observations recorded 2026-09-25

One-off checks on the current core. Measurements only, no conclusions drawn.

| Check | Result |
|---|---|
| `robust.py`, Nethra rows | 0.83 / 0.38 / 0.50 / 0.04, same as the notes |
| Same inputs twice | bit-identical |
| Checkpoint, reload, continue | bit-identical to the run without reload |
| `d(sum a)/dt = sum J - leak sum a` | holds: ETD 1e-14, RK4 1e-8 (Gamma sums to zero) |
| ETD vs RK4, one interval, 191 Nethra | max abs diff 1.1e-5; default ETD (2 pieces) 3.1e-5 vs 64 pieces |
| X then B (75%) or C (25%), then D; B share of B+C after X, seed 14, 3 world seeds | 0.66 / 0.62 / 0.61 / 0.52 / 0.51 / 0.52 after 5 / 10 / 20 / 50 / 100 / 200 blocks |
| Checkpoint contents | adaptive frontier tolerance `_tol` not saved |

Files not in this repo: `tests/baselines.py`, and the other scripts the notes cite.
