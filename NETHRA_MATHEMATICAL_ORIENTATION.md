# Nethra mathematical orientation — 2026-09-22

This note exists to prevent repeated category errors.  It is descriptive guidance for investigation,
not a replacement architecture.

## 1. Four mathematical questions that must remain separate

### A. Instantaneous field state
The live Nethra field is an ODE.  For fixed persistent topology/parameters and whatever F61 trace
variables are actually active, the instantaneous dynamical state is the collection required by the
ODE to determine the next derivative under a supplied external current.

This is a Markov/state-space question.  It is NOT the same thing as "what happened during the last
completed interval."

### B. Completed-interval manifestation
This is a path-observable question:
    what finite-interval facts about one persistent Nethra are retained after an interval completes?

Current strongest native low-order descriptor:
    X_i[k] = (Delta a_i[k], A_i[k])
    A_i[k] = integral_interval a_i(t) dt

External source:
    U_i[k] = integral_interval J_i(t) dt
is retained separately as provenance.

Established:
- Delta alone loses sustained stationary manifestation.
- A alone loses endpoint change.
- M = C Delta + lambda A is exact and native, but is a one-scalar projection of X, not a replacement.
- Fixed-conductance incidence quantities reconstruct exactly from A:
      Phi_ij = A_i - A_j
      Q_ij   = g_ij Phi_ij
- X plus separate U reconstructs endpoint, leak integral, M, net internal drive and net integrated
  F61 contribution by field balance.

NOT established:
- X is a complete description of arbitrary within-interval chronology.
- X is sufficient for every future refinement question.

This does NOT block adaptive expectation. If an existing relation can already be refound and a later
continuation can be registered, that is sufficient to update empirical support.

### C. Adaptive expectation / predictive relation
The immediate learning question is:
    when an earned relation R refounds, what continuation has worked here before,
    and did that continuation occur again?

This does not require proving a privileged causal/statistical relation or comparing against marginals.
A temporal lag may be part of the experience used to qualify the relation. Once earned, present
resonance of that learned relation can prime a continuation.

Persistent evidence may therefore summarize success/failure history directly:
    experience -> evidence -> conductance -> resonance -> expectation.

Ambiguity is not a failure:
- if R repeatedly precedes B, B becomes strongly expected;
- if R sometimes precedes D, D becomes a competing continuation;
- if additional existing context distinguishes the cases, refinement can form;
- if nothing distinguishes them, both continuations remain uncertain/ambiguous.

A complete predictive-state representation is a stronger mathematical question and should not be
silently imposed as a prerequisite for this empirical learning loop.

Nethra consequence:
- construction/reuse should ultimately be justified by prospectively different histories;
- no semantic label is needed;
- no requirement exists that one completed interval be compressed to one scalar;
- a learned temporal lag may be evidence used to qualify a prospective relation without becoming a
  physical propagation-delay parameter;
- present resonance of an earned temporal relation can express prediction of a later manifestation.
  Exact source-free reenactment after the historical lag is a separate generative-rollout question,
  not the criterion for whether the relation is predictive.

### D. Causal sensitivity / plasticity
Only after the consequence representation is fixed do we ask:
    how would future residual/loss have changed if g_ij had been different?

Gold-standard diagnostic remains identical replay at g+delta and g-delta.

Established:
- epsilon*Phi is NOT the exact full-field gradient.  It is the frozen-trajectory/direct local term.
- local temporal histories can nevertheless recover much of counterfactual sensitivity for the
  tested external-source objective.
- that success does not license treating the local term as an exact gradient or as the definition
  of manifestation.

## 2. Where older Nethra work already answered questions

Foundation 1 already established:
- persistent Nethra identity is separate from transient current instantiation;
- a relational Nethra's transient state is recursively derived from member states;
- current state never changes persistent identity;
- transient descriptive equality has no semantic authority.

Do not rediscover or replace this distinction.

Later audits established:
- a bare relation participation delta is too coarse: the same persistent relation can enter through
  different member-state configurations that carry different prospective information;
- F61 continuous live state retained temporal distinctions that the discrete learning boundary
  discarded;
- therefore the field->learning seam, not persistent identity, was the problem.

Per-incidence evidence is now closed:
- persistent statistical strength must be available per relation-member incidence;
- route-wide evidence loses distinctions that the field can support.

## 3. Relevant external mathematics

### State-space / realization theory
Use this only to answer "what variables determine future evolution?" or "what history summary is
predictively sufficient?"

Nonlinear observability asks whether hidden instantaneous state can be reconstructed from outputs.
It does not define completed-interval manifestation.

Minimal realization / predictive-state ideas are useful as a diagnostic principle:
two histories may be identified only when they imply the same future behavior.  Do not use that
criterion to define the raw interval observation.

References:
- Hermann & Krener, nonlinear observability / observability rank condition.
- H. J. Sussmann, minimal realizations of nonlinear systems.
- Littman, Sutton & Singh, "Predictive Representations of State", NIPS 2001.

### Path signatures / rough-path theory
This is directly relevant to the remaining manifestation question.

A finite interval is a path, not merely an endpoint.  The signature of a path is a hierarchy of
iterated integrals.  For bounded-variation paths the full signature characterizes the path up to
tree-like equivalence (Hambly & Lyons).

For a time-augmented one-Nethra path
    X(t) = (t, a_i(t)),
the low signature levels have a direct interpretation:
- level 1 contains Delta t and Delta a;
- level 2 contains time-state iterated integrals from which integral a(t) dt can be recovered
  together with boundary data.

Therefore the current descriptor (Delta a, A) is not an arbitrary pair: it corresponds to
low-order information in a principled hierarchy of path observables.

This gives the correct next test:
- search for physically realizable interval pairs with equal (Delta a, A);
- ask whether later Nethra-usable completed-interval quantities distinguish them;
- if yes, test the next time-augmented iterated-integral level rather than inventing an unrelated
  scalar.

Do NOT automatically install a path-signature model.  Use the hierarchy as a falsification tool for
manifestation sufficiency.

Reference:
- B. Hambly & T. Lyons, "Uniqueness for the signature of a path of bounded variation and the
  reduced path group", Annals of Mathematics 171 (2010), 109-167.

### Projection / Mori-Zwanzig
If a chosen reduced observable is not Markov-sufficient, exact reduced dynamics generically acquire
memory and a fluctuating/unresolved term.

This is important interpretation, not machinery to import:
- if (Delta a, A) fails to determine later behavior without several prior intervals, that does not
  automatically mean Nethra needs an externally designed memory system;
- it means the projection discarded dynamical information, which must reappear as history dependence
  unless a richer manifestation coordinate restores Markov closure.

The previously observed four-lag local sensitivity result is qualitatively consistent with this
general phenomenon.

### Linear response / susceptibility
The counterfactual g±delta experiment belongs here.

In dynamical systems, response to a parameter perturbation generally includes propagation through
the whole trajectory.  Local correlations can sometimes recover response through temporal kernels,
especially in linear-response regimes, but this is conditional and not an identity for arbitrary
nonequilibrium nonlinear dynamics.

This is the correct mathematical interpretation of:
    local Phi/residual history -> approximate true conductance sensitivity.

Do not call epsilon*Phi an exact gradient unless the full tangent/adjoint sensitivity calculation
agrees.

References:
- D. Ruelle, review of linear response theory for differentiable dynamical systems.
- fluctuation-dissipation / response theory literature.

### ODE sensitivity analysis
Exact d(output)/dg for a nonlinear ODE requires trajectory sensitivity (forward/tangent or
reverse/adjoint) or finite perturbation.

This explains the already-measured failure of the frozen-trajectory epsilon*Phi derivative.
Use g±delta replay as the ground-truth audit while investigating whether Nethra-local history
contains enough information to approximate that sensitivity.

### Eligibility traces
Biological/computational eligibility traces support one narrow point only:
local synaptic history can retain temporally delayed credit information.

They do NOT justify:
- a global reward signal in Nethra;
- a particular exponential decay;
- a particular learning equation;
- treating prediction residual as reward.

Nethra's experimentally recovered short temporal kernel should be studied on its own terms.

## 4. Immediate implication for manifestation

Do not ask "what is the one scalar manifestation?"

There may be no privileged scalar.

The present mathematically grounded candidate is a low-order path descriptor:
    (Delta a, A)
with U separate as provenance.

The next manifestation test should be a sufficiency/collision experiment, not a prediction test:

1. Construct physically realizable intervals whose relevant Nethra have identical (Delta a, A).
2. Keep persistent topology/parameters fixed.
3. Compare native completed-interval quantities that could be available to later Nethra:
   - time-resolved incidence ordering;
   - F61 convergence/supplier structure;
   - recursively derived member instantiation;
   - any higher time-state iterated integral.
4. If nothing Nethra-usable differs, (Delta a, A) survives that class.
5. If something differs, identify the lowest-order missing path coordinate.
6. Only after the manifestation representation is stable resume predictive equivalence tests.
7. Only after prediction is stable resume counterfactual plasticity tests.

## 5. Questions already answered; do not reopen without contradictory evidence

- Persistent identity and transient instantiation are separate.
- There is only ordinary Nethra; no semantic subtype is needed.
- Participation delta alone is insufficient as general transient manifestation.
- F61 live dynamics can retain temporal distinctions lost by the old learning interface.
- Exact continuous-value matching is not a viable recurrence mechanism.
- External adaptive precision/range selection is not justified.
- Per-incidence persistent evidence is required; route-wide evidence is too lossy.
- Per-Nethra A_i storage is sufficient to reconstruct fixed-g Phi and Q and is cheaper than storing
  one integrated Q per edge.
- M=C Delta+lambda A is native and useful, but is not justified as the universal prediction target.
- External-source-only consequence is not universal because relation Nethra can manifest internally.
- epsilon*Phi is not the exact conductance gradient.
- F61 convergence is part of field drive and must not be silently omitted from future accounting.
- A short local temporal history can approximate true conductance sensitivity for the tested
  external-source objective; arbitrary-Nethra manifestation remains unresolved.


## 6. Available passive one-step prediction readout

A passive one-step prediction workflow is now established and documented in
`NETHRA_PREDICTION_READOUT.md`.

The prediction target is the prospective live field at one requested future TIME from the complete
real state known now. Prediction does not require Nethra to recursively consume its own output.

Operationally:
- clone the complete live Nethra state;
- supply only genuinely known future inputs, normally TIME;
- evolve the disposable shadow to the requested horizon;
- rank externally bound input Nethra from the prospective field;
- record top-k and separation/confidence diagnostics;
- discard the shadow;
- only then reveal the real observation to the live Nethra and continue ordinary learning.

The primary finite-horizon input readout currently used by the passive audits is

    r_i(T) = a_i^shadow(T) - a_i^live(0) exp(-lambda T / C)

with

    margin = score(top1) - score(top2)

and optional normalized separation

    |top1-top2| / (|top1|+|top2|+epsilon).

The normalized separation is a field-separation diagnostic, not a calibrated probability.

This prediction procedure must remain causally separate from imaginative rollout. A multi-step
imagined trajectory may use a disposable simulation that feeds hypothetical predictions forward,
but hypothetical observations must never become real evidence in the live learner.
