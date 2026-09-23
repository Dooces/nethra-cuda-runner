# Nethra TODOs

## Intrinsic choice by prospective predictive reach

**Status:** Future work only. This is a design target. It is outside the frozen Nethra core and does not support any current benchmark claim.

### Scope

This applies once Nethra has consequential outputs/actions that can be taken and later observed through their consequences.

An output/action is represented as a Nethra in the same field as other Nethra. Taking an output makes that event part of experience. Later sensory/observational deltas can therefore become learned consequences of that output through the same temporal relation machinery used elsewhere.

No external reward scalar is required by this proposal.

### Idea

For each currently available output/action, use the learned prospective field to ask how far into future time its consequences remain coherently and confidently supported.

The intrinsic value of an action is its **predictive temporal reach**: the amount of future structure that remains supported after that action is prospectively primed.

Conceptually:

```
candidate action
    -> prospective activation of learned consequences
    -> supported consequence chain across increasing temporal displacement
    -> measure how far coherent/confident prediction persists
    -> prefer the action with the greatest supported predictive reach
```

A one-step prediction that agrees with the next observation contributes only the nearest-time part of this quantity. The target is persistence of supported prediction across future time.

A conceptual quantity is:

```
V(a) = integral over future displacement tau of C_a(tau) d tau
```

where `C_a(tau)` is the confidence/support carried by the prospective field for consequences of action `a` at future displacement `tau`.

This formula is only a design sketch. Do not implement it as an external evaluator unless the same quantity can be obtained from native Nethra field quantities.

### Integration

1. Allow consequential outputs to be experienced.
2. Record each taken output as part of the same interval history as other observations.
3. Let ordinary Nethra learning establish output -> consequence relations at the temporal separations actually observed.
4. At a later choice point, prospectively prime each available output candidate separately.
5. Use the resulting field to determine how far supported consequence structure persists through future temporal depth.
6. Select according to that native prospective reach once a field-native measure of reach/confidence has been identified.

### Required prior condition

This mechanism needs learned output consequences. An output with no experienced consequences has no established predictive reach yet.

Exploration therefore remains necessary for previously untried or weakly understood outputs. How exploration is selected is unresolved here.

### Boundaries

- Do not add a hand-authored reward value for outcomes.
- Do not equate the objective with one-step surprise minimization.
- Do not add a separate planner, rollout model, value network, evaluator, or policy learner.
- Do not treat recursive depth by itself as value; the relevant quantity is supported prospective prediction across time.
- Do not promote this into the frozen core until consequential-output learning exists and the reach quantity can be expressed through existing/native Nethra state.
- Preserve the same Nethra object type for outputs, observations, consequences, and learned relations.

### Open questions

- Which existing field quantity best represents `C_a(tau)`: activation magnitude, residual support, independent resonance overlap, persistence, or a combination already present in the field?
- How should confidence decay across increasing temporal separation?
- How should ambiguity between several supported futures contribute to reach?
- How should never-tried outputs acquire enough experience to become comparable?
- Can the criterion emerge directly from prospective field persistence without introducing a new scalar objective at all?
