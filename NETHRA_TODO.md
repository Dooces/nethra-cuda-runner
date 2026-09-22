# Nethra current status / TODO — 2026-09-22

This file is the current project status. It separates:
- what is frozen in the one-file core;
- what has been established experimentally but is not yet installed in that freeze;
- what has been falsified/rejected;
- what remains genuinely unresolved.

The temporary/candidate residual mechanism is treated as the intended subtraction-before-construction direction, not as something already implemented successfully.

## Frozen reference

Authoritative frozen field/boundary branch:
- `nethra-interval-boundary-frozen`
- head `c7737da52080f87605803bbc6b33d90e48e93591`

Frozen persistent ontology:
- one persistent object type: `Nethra`;
- no input/relation/perspective/temporal/candidate persistent subtype;
- arbitrary-size support routes;
- recursive relations and cycles are allowed;
- direct self-support is rejected;
- recursively refound descriptions containing their own target are omitted as tautological rather than registered as new support;
- longer cycles remain legal.

Frozen field:
- symmetric nonnegative incidences;
- external source current;
- leakage;
- bounded evidence -> conductance mapping;
- F61 conservative convergence redistribution;
- RK4 integration;
- learning/construction cannot write activation directly.

Frozen completed-interval boundary:
- `observe()` is gone;
- `_complete_interval(source_current, delta)` records exact external-source provenance and exact activation change;
- it does not discretize, round, classify, compare, construct, estimate probability, fabricate residuals, or alter conductance;
- live `step()` on the frozen branch does NOT invoke the old learner;
- `_consider_completed_interval_provisional()` remains only for explicit regression comparison.

Important branch warning:
- `nethra-field-tension-investigation` contains a stale experimental copy of `nethra.py` whose `step()` still invokes `_consider_completed_interval_provisional()`;
- do NOT treat that file as the authoritative live core;
- the authoritative runtime semantics are the `nethra-interval-boundary-frozen` branch above;
- investigation probes must be interpreted by what they actually call (many use direct/custom integration and do not exercise the stale live learner).

Important implementation distinction:
- the frozen core currently stores source current + Delta-a only;
- it does NOT yet store the experimentally established interval integral A_i;
- it still represents earned route evidence route-wide rather than with the per-incidence evidence that later tests established is required.

## Established outside the freeze

### Per-Nethra completed-interval integral

Established:
`A_i = integral_interval a_i(t) dt`

For conductance fixed during one interval:
`Phi_ij = A_i - A_j`
`Q_ij = g_ij * Phi_ij`

Fedora tests established:
- reconstructed Q matched directly integrated incidence charge to numerical precision;
- final activations were unchanged;
- Python overhead was about 1.9–2.4% in the tested graphs;
- transient storage is O(N), not O(E).

Current status:
- conceptually/experimentally accepted;
- not yet added to the frozen one-file interval record.

The strongest low-order completed-interval descriptor currently supported is:
`X_i = (Delta a_i, A_i)`

Keep external source integral/provenance separate:
`U_i = integral J_i dt`

`M_i = C*Delta a_i + lambda*A_i` is an exact native derived quantity, but tests did NOT justify using M as the universal prediction/residual target.

### Per-incidence persistent evidence

Established by graded and recursive-support tests:
- one arbitrary-arity relation must be able to retain different earned evidence on different relation-member incidences;
- route-wide evidence loses distinctions that the field can use.

Closure test:
- graded true support separated strongly from continuous nuisances only with per-incidence evidence;
- an already-constructed relation Nethra used as true support also separated from nuisance with per-incidence evidence;
- route-wide controls could not retain the support distinction.

Current status:
- representation decision CLOSED: restore per-incidence evidence;
- implementation in the frozen one-file core still pending;
- the final evidence update law is not frozen.

### Recursive ordinary-Nethra structure

Established:
- relation Nethra can participate in later relations without changing object type;
- recursive closure/refinding works;
- canonical reuse fixed the old duplicate-description blow-up in persistent structure;
- ablation tests showed learned recursive layers were behaviorally/field necessary, not unused descriptive handles.

Archived V74 frozen result reached literal relation depth 10 in three worlds.
Later relation-depth stress reached 12.
V68 closure sizes of 49, 58 and 67 are active closure sizes, not literal relation depth 49/58/67.

### Temporal prospective relations

Established:
- delayed prospective relations at lags 1,2,3,5,8 were recovered in the archived temporal work;
- after qualification, temporal distinction crystallized into ordinary Nethra structure;
- the earned temporal Nethra could participate recursively;
- sourcing/refinding the earned temporal relation produced consequence preference through ordinary field resonance.

Correct interpretation:
- lag is evidence used to establish the relation;
- lag is not required to become a literal propagation-delay parameter in F61;
- prediction does not require source-free reenactment of the original elapsed time.

### Adaptive expectation capability

A shadow test with an already-earned context Nethra R and already-available continuation relations established:
- persistent evidence -> conductance -> field resonance is sufficient to adapt expectation;
- repeated B increased B evidence and later B resonance;
- one contradictory D did not erase the established B relation;
- sustained D evidence eventually reversed the field preference;
- alternating B/D remained ambiguous rather than forcing a winner;
- no conditional-probability or marginal-baseline table was needed.

Important limit:
- the test was GIVEN which continuation occurred and applied signed +1/-1 evidence;
- it does NOT establish the Nethra-native mechanism that generates that signed update;
- established Nethra do not need a semantic operation that "registers agreement/disagreement."

### Whole-support recruitment

Shadow tests established a useful feasibility result:
- an existing relation missing true support A could weakly recruit the entire independently sourced support;
- this also recruited irrelevant D;
- subsequent incidence-local credit made A strong and D weak;
- no subset enumeration was required.

Important limit:
- those tests used a provisional local tension/credit law;
- therefore whole-support recruitment is supported as a construction/refinement pattern, not frozen as live plasticity.

### F61 convergence / independence

Established field behavior:
- simultaneous positive suppliers can receive a conservative convergence redistribution;
- pair-history independence modulates the extra convergence term;
- duplicate/correlated histories do not receive the same extra convergence bonus as independent histories;
- convergence is real field drive and cannot be omitted from field accounting.

Still unresolved:
- what native quantity should supply/update F61's own residual trace rho in the rebuilt learning path;
- do not identify the candidate residual below with F61 epsilon/rho without a separate derivation/test.

## Intended subtraction-before-construction mechanism

This is the conceptual mechanism to implement/test accurately.

Existing persistent Nethra get first opportunity to refind/close/account for what is already represented.

A temporary/unmaterialized candidate is allowed to carry the still-unaccounted prospective difference.

Its residual keeps that candidate primed across relevant completed intervals.

As existing structure accounts for part of the recurrence, that accounted component is subtracted from what remains attributable to the temporary candidate.

Whatever remains unresolved stays attributable to the difference.

If the unresolved structure recurs prospectively, the temporary candidate can acquire per-incidence evidence and eventually crystallize as an ordinary persistent Nethra.

Once persistent structure accounts for the recurrence, the temporary residual should collapse and should not create another redundant Nethra.

The resulting recursive pattern is:
`unresolved residual -> temporary candidate -> earned Nethra -> ordinary resonance/refinding -> later unresolved residual -> ...`

Important status:
- this is the intended mechanism;
- older V52/L77-style temporary candidates and gain tests are evidence that temporary testing/crystallization can work, but those implementations used external statistical qualification machinery;
- the current one-file frozen core does NOT yet implement a clean residual-bearing temporary candidate;
- therefore do not describe this carrier as already solved.

## What subtraction must preserve

The rebuilt mechanism must preserve all of these established constraints:
- complete existing-structure closure/refinding before construction;
- original-source provenance remains distinct from recursively manifested/refound Nethra;
- no candidate may influence behavior as persistent topology before crystallization;
- no subset scanner;
- no consequence chooser supplied by an evaluator;
- no conditional/marginal probability ledger;
- no semantic type for candidate/relation/input/perspective;
- no forced winner when multiple continuations remain supported;
- no hard gate that disconnects weak structure;
- cycles remain allowed;
- per-incidence evidence, once earned;
- graded/continuous interval information must not be collapsed to binary membership merely to make matching easy.

## Continuous precision / transient manifestation

This remains genuinely open and is coupled to the residual-bearing temporary candidate.

Established:
- exact continuous-value matching fragments recurrence and is not acceptable as the general solution;
- dynamic external range/resolution selection was rejected as outside learner machinery;
- greedy binary refinement was rejected;
- fixed rounding may be an engineering precision limit, but it is not a Nethra-native solution;
- signed participation/state was historically necessary for temporal distinctions; collapsing transient state to bare IDs destroyed them;
- Delta-a alone loses sustained stationary manifestation;
- A alone loses endpoint change;
- `(Delta a, A)` is the strongest low-order native descriptor currently established.

Still open:
- how a temporary candidate retains/reuses graded residual structure without exact-float identity;
- whether the candidate/field itself naturally controls effective precision;
- whether a higher-order within-interval temporal coordinate is needed beyond `(Delta a, A)`.

Do not install a matcher, clustering layer, adaptive quantizer or semantic precision policy to close this gap.

## Structural subtraction already present as regression evidence

The old/provisional one-file learner contains `_accounted(before, after)`:
- it checks existing persistent Nethra before minting another handle;
- it is a useful regression embodiment of "existing structure first";
- it is NOT the live solution because its before/after descriptions come from the provisional discrete event/probability path.

Likewise, archived gain/subset work demonstrated why proper-substructure subtraction matters, but the count/probability machinery is no longer accepted as the live implementation.

## Diagnostic work that remains useful but is not the mechanism

Keep these results as falsification/audit evidence:
- source-only residual `U_next - P_prior` is not universal because a relation Nethra can manifest internally with U=0;
- `epsilon*Phi` is not the exact full-field conductance gradient;
- residual-squared "energy" interpretation was not justified;
- clamped observation boundary produced loading artifacts;
- source-free replay/blocking/overexpectation probes demonstrate desirable subtraction behavior but are evaluator diagnostics, not the live residual carrier;
- counterfactual g+/-delta replay is a sensitivity audit only;
- `M=C*Delta a+lambda*A` is native but not established as a universal residual target;
- a short local temporal history can approximate some external-source sensitivity, but this does not define Nethra learning.

## Current unresolved implementation work

### Residual-bearing temporary candidate

Implement/test the temporary candidate so that:
- it is transient, not a second persistent ontology;
- it carries the unresolved prospective difference after existing Nethra subtraction;
- it can remain primed across intervals;
- accounted components actually reduce its residual;
- the remaining residual is what receives attribution to newly differing support;
- it can crystallize into an ordinary Nethra only after recurrent prospective support;
- once crystallized/accounted, repeated identical experience does not mint another relation.

No existing test currently proves all of this in the one-file/F61 core.

### Per-incidence evidence in the live core

Move the already-established representation into the rebuilt learning path:
- persistent strength must be relation-member/incidence local;
- keep symmetric field incidence;
- do not turn incidence evidence into semantic member roles.

### Whole-support recruitment under the real residual carrier

Retest the successful whole-support idea using the actual candidate residual:
- recruit whole independently sourced support weakly;
- allow nuisance support to enter;
- let per-incidence evidence determine what persists;
- no subset enumeration.

### Construction bootstrap

The first unresolved relation must become testable without:
- external Candidate/ResourceCloud machinery;
- pair manufacture;
- exhaustive subset enumeration;
- an externally chosen consequence;
- conditional-probability qualification.

This may be the same problem as creating the first residual-bearing temporary candidate from a completed interval. Do not split it into a separate learner unless the field forces that conclusion.

### F61 residual trace source

Keep F61 rho/convergence intact.
Determine later whether the rebuilt candidate/residual dynamics supply a native epsilon for rho.
Do not make rho define the candidate residual merely because the names are similar.

### Precision

Once the temporary candidate exists, stress it with:
- continuous graded input;
- nearby values;
- nuisance noise;
- recursive relation support;
- changing regimes.

Only then decide whether any explicit finite precision is still necessary.

## Deferred

Duplicate consolidation / sleep:
- preserve ambiguity online;
- no automatic merge/delete/rank;
- sleep/offline consolidation remains a later optional concern.

Decay:
- still not frozen in the live one-file core;
- only add a decay/competition law after the residual-bearing candidate/evidence update is clear enough to test without hiding mistakes.

## Rejected shortcuts

Do not reintroduce:
- Candidate/ResourceCloud as an independent learner substrate;
- direct pair-Nethra manufacture;
- binary source-presence residual as universal evidence;
- `history_count/support_count/outcome_count/conditional/baseline` as live authority;
- external adaptive precision/range chooser;
- exact-float recurrence keys;
- primitive-leaf flattening as universal evidence identity;
- raw activation ranking as a safe construction frontier;
- stop-at-first-gain-failure search;
- source-free autonomous lag reenactment as a requirement for prediction;
- semantic input/output/relation/temporal/perspective Nethra subtypes;
- external Rescorla-Wagner-style expectation/error as the Nethra mechanism.

## Immediate test target

Build the smallest faithful live test around the temporary candidate, not around an externally computed error:

1. establish an ordinary predictive relation through repeated experience;
2. present the same completed relation again and verify existing closure/subtraction leaves negligible candidate residual;
3. add an irrelevant X while the consequence is already accounted for and verify X does not acquire durable consequence evidence from the accounted component;
4. change the continuation so a genuine residual remains and verify the temporary candidate stays primed;
5. repeat that changed continuation and verify only the unresolved per-incidence support accumulates;
6. crystallize the candidate into an ordinary Nethra and verify it enters the same symmetric field;
7. repeat recursively with that new Nethra as support/consequence;
8. run graded/noisy versions without binary event projection.

The experiment must expose the candidate/residual directly. A test that computes the desired answer externally and then updates evidence is not sufficient.
