# Nethra agreed-fixes freeze — 2026-09-23

This branch is rebuilt directly from the authoritative clean interval-boundary freeze
`bbc023c07a81ff5e17c8d7f66db4cd442600290d`.

## Frozen in this revision

- Exactly one persistent participant type: `Nethra`.
- Symmetric F61 field dynamics, leakage, conductance mapping, convergence redistribution and RK4
  integration are unchanged from the clean freeze.
- Live `step()` has no construction or provisional-probability authority.
- Completed intervals retain exact external-source provenance and exact activation delta.
- Completed intervals now also retain the accepted per-Nethra activation integral
  `A_i = integral a_i(t) dt`, using the same RK4 quadrature stages as field integration.
  With fixed conductance during an interval:
  `Phi_ij = A_i - A_j` and `Q_ij = g_ij * Phi_ij`.
- Existing route evidence is losslessly mirrored into persistent per-route/member incidence
  evidence.  At registration this is field-behaviour preserving; it permits later experiments to
  retain different evidence for different members without changing topology or adding another
  persistent object type.
- Recursive closure/refinding, arbitrary-size routes, cycles, direct-self-support rejection and
  subtraction-before-construction remain the frozen structural semantics.
- State-qualified refinding in the retained provisional comparison path now computes the current
  source event before closure; it no longer asks closure to interpret the new observation using an
  event from the preceding interval.
- The empty-projection/unqualified-route closure fix from the clean freeze remains present.

## Explicitly not changed

- `g_min` and the frozen evidence-to-conductance equation are unchanged.  Experiments with
  `g(0)=0` are evidence about later plasticity/admission behaviour, but changing the frozen field
  floor has not been promoted here.
- F61 residual-trace and convergence mathematics are unchanged.
- Pairwise residual-independence statistics are not optimized away when convergence gain is zero.
  That optimization has a later-history consequence if convergence is subsequently enabled and is
  therefore outside this semantic freeze.
- The old `_consider_completed_interval_provisional()` probability/counting learner remains
  manually callable only for regression comparison.  `step()` does not invoke it.
- No native learner, temporary candidate, selector, probability ledger, candidate cloud, subset
  search, semantic subtype or evaluator has been added to live execution.

## Still outside the freeze

The necessity of per-incidence evidence is settled; its final universal local update law is not.
The generic residual-bearing temporary carrier and arbitrary-structure construction bootstrap are
not frozen.  The exact generic consequence/residual quantity for internally manifested Nethra is
not frozen.  How that residual should supply F61 rho is not frozen.  Continuous residual identity /
precision for construction is not frozen.

Those unresolved pieces must be solved without changing this frozen field/boundary contract unless
a later revision is explicitly discussed and versioned.
