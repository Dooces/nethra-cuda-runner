# Nethra TODO — 2026-09-22

Reference freeze:
- `nethra-interval-boundary-frozen`: completed interval stores exact external source current and exact Nethra activation delta. Live `step()` has no automatic provisional learner.

## Dependency chain under investigation

1. **Interval-integrated local field flow — CLOSED: STORE PER-NETHRA INTEGRAL**
   - Purpose: retain what each incidence physically carried over a finite interval.
   - Resolved representation:
     - store one completed-interval activation integral per Nethra: `A_i = integral a_i dt`;
     - because conductance is fixed during a completed interval, recover any incidence potential integral as `Phi_ij = A_i - A_j`;
     - recover existing incidence charge exactly as `Q_ij = g_ij * Phi_ij`.
   - Fedora verification:
     - direct integrated edge charge vs reconstructed `g(A_i-A_j)`: worst error ~5.2e-18;
     - final activations unchanged exactly;
     - CPU overhead ~1.9–2.4% for 200–4000 edges;
     - transient storage O(N) activation-integral scalars instead of O(E) edge-flow scalars.
   - Decision: use per-Nethra interval integrals as the canonical transient representation if/when local field-flow plasticity is promoted. Do not store per-edge integrated flow unless another requirement appears.

2. **Local receiver residual / tension — OPEN / REFORMULATED**
   - Purpose: give each ordinary Nethra a local statistical tension from completed Nethra intervals without a probability ledger.
   - Earlier positive-flow S-P candidate passed frequency calibration, regime reversal, redundancy suppression and shared-residual tests, but remains a new statistical law.
   - Residual-squared / conductance-gradient proposal was vetted and is NOT valid as an exact field gradient:
     - `epsilon * Phi` is only the direct/frozen-trajectory derivative; changing conductance changes the full activation trajectory and all coupled flows;
     - full finite-difference gradient magnitude differed by ~1.4% to 57% in simple loaded passive tests;
     - 2500 random passive-network sign tests produced 8 genuine sign mismatches, so the local term is not a guaranteed descent direction;
     - virtual missing-edge finite-change formula retained ~7.9% first-order error even at delta-g=1e-6 and ~11.1% error at delta-g=.2.
   - External-source-only residual is NOT universal:
     - recursive-consequence control: future relation Nethra had external source U=0 but positive delta-a=0.11739; source residual therefore called a real positive manifestation an overprediction.
   - F61 convergence is omitted by `sum Q`:
     - test fixture: conductive charge 0.14517, convergence charge 0.07258, so one third of internal receiver drive was absent from the proposed prediction.
   - Conductance-corrected `Phi` support weighting remains promising:
     - with true incidence initially weak and nuisance incidences strong, Q weighting ended mean evidence 70.8 vs 60.9/60.6 nuisance;
     - Phi weighting ended 88.1 vs 52.5/52.2 nuisance; both recovered the true incidence in all 8 seeds, Phi separated it substantially better.
   - Evidence-update chain rule remains relevant if persistent variable is evidence e rather than conductance g: `dg/de` collapses near conductance saturation, so `Phi*epsilon` cannot be justified as a literal gradient step in e-space.
   - Next formulation must use a consequence quantity valid for every Nethra manifestation (not only external source), and must account for established F61 field contributions including convergence.

3. **Per-incidence evidence — CLOSED: RESTORE**
   - Purpose: persist which members of one arbitrary-arity Nethra actually carry its prospective relation.
   - Historical precedent: V56 stored relation-member incidence evidence independently; one-file route-wide evidence collapsed this.
   - Closure test passed on Fedora (run 35784621800):
     - graded continuous support: true A mean evidence 64.4375; nuisance B/C 0.000393 / 0.002088; A prospective response 0.008250 vs ~0.002208 nuisance;
     - recursive Nethra support: true supporting Nethra mean evidence 48.4446; nuisance B/C 10.6302 / 10.0116; true-support response 0.006103 vs 0.003793 / 0.003714;
     - route-wide controls could not store any context-member distinction under either environment.
   - Decision: persistent evidence must be available per relation-member incidence. Route-wide evidence is an information-losing simplification and should be removed when the live plasticity path is rebuilt.
   - This closes the representation question only; the final local evidence-update law remains coupled to TODO #2.

4. **Whole-support recruitment**
   - Purpose: let an existing Nethra add missing source support without subset search.
   - Candidate rule: when its current perspective receives positive tension, add the whole independently sourced preceding support as weak incidences; do not choose a subset.
   - Established so far: missing true A and nuisance D were both recruited; incidence-local credit later made A strong and D weak.
   - Status: OPEN; requires larger nuisance/recursive tests.

5. **Construction bootstrap**
   - Question: when no relevant Nethra exists, what earns the first weak whole-event Nethra without a candidate scanner or consequence chooser?
   - Status: OPEN / BLOCKING FULL PLASTICITY.

6. **F61 own residual**
   - Tension histories correctly make exact duplicates non-independent, but shared receiver coupling gave only partial independence for alternating predictors.
   - Status: OPEN; do not equate tension with F61 epsilon yet.

7. **Duplicate consolidation / sleep**
   - Existing policy: preserve ambiguity online; journal for later consolidation tests.
   - Status: DEFERRED. No current need to merge.

## Rejected / closed paths

- Dynamic external resolution/range chooser: REJECTED as outside learner machinery.
- Exact continuous matching: REJECTED as unnecessary and recurrence-fragmenting.
- Greedy binary range refinement: REJECTED; cancellation hides deeper structure.
- Clamped observation boundary as literal prediction-error current: REJECTED; added conductive loading defeated the proposed interpretation.
- Direct `p * epsilon` update using endpoint branch current: REJECTED; failed simplest frequency-order test.
- Residual-squared proposal as exact physical/field energy gradient: REJECTED AS STATED; squared source-charge residual is a statistical loss, `epsilon*Phi` omits trajectory sensitivity, and source-only consequences break recursive Nethra.
- External-source-only consequence residual `U_next - P_prior`: REJECTED as a universal Nethra consequence measure.
- Per-edge integrated-flow storage: SUPERSEDED by per-Nethra activation integrals `A_i`, from which `Phi` and `Q` reconstruct exactly during fixed-conductance intervals.
- Global automatic `history_count/support_count/outcome_count/conditional/baseline` authority in live step: REMOVED FROM LIVE PATH; retained only for explicit regression comparison.

## Current task

**Next target: #2 Local receiver residual / tension.**

Per-incidence representation and interval-flow representation are resolved. The next unresolved dependency is the consequence/residual quantity itself. The next candidate must:
- operate on completed Nethra manifestations, including internally manifested relation Nethra;
- retain original-source provenance separately;
- include established F61 field contribution rather than conductive Q alone;
- preserve local computability and avoid global probability ledgers;
- treat `Phi` as local conductance opportunity unless tests justify a stronger interpretation;
- be tested against full finite perturbations before any gradient language is used.
