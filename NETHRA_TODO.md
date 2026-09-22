# Nethra TODO — 2026-09-22

Reference freeze:
- `nethra-interval-boundary-frozen`: completed interval stores exact external source current and exact Nethra activation delta. Live `step()` has no automatic provisional learner.

## Dependency chain under investigation

1. **Interval-integrated local field flow**
   - Purpose: retain what each incidence physically carried over a finite interval.
   - Established so far: RK4-refinement stable; exact final activations preserved; O(E) transient storage; ~6.8–8.6% Python wall-time overhead in tested graphs.
   - Status: STRONG CANDIDATE; needs integration contract before promotion.

2. **Local receiver residual / S-P tension**
   - Candidate:
     - `P_m(k) = sum positive incoming integrated field charge into m`
     - `epsilon_m(k+1) = S_m(k+1) - P_m(k)`
     - `T_R = sum_m p_Rm(k) * epsilon_m(k+1)`
   - Purpose: give each ordinary Nethra a local statistical tension from its own field perspective.
   - Established so far: frequency calibration, regime reversal, redundancy suppression, shared residual all passed shadow tests.
   - Important limit: this is a new local statistical plasticity law; passive F61 does not imply the subtraction by itself.
   - Status: OPEN / HIGH PRIORITY.

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
- Global automatic `history_count/support_count/outcome_count/conditional/baseline` authority in live step: REMOVED FROM LIVE PATH; retained only for explicit regression comparison.

## Current task

**Next target: #2 Local receiver residual / S-P tension.**

Per-incidence representation is now resolved. The next unresolved dependency is whether S-P tension is the correct Nethra-local statistical law or only one successful proxy. Promotion requires a direct formulation over completed interval quantities, stable behavior with multiple consequences and recursive relations, and no global probability ledger or semantic target selection.
