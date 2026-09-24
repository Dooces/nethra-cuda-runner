# Nethra temporal-route recursion repair freeze

Date: 2026-09-23 / 2026-09-24 UTC

## Frozen ancestry

Previous indexed freeze:
- commit: `b2ee46b79c8b0a9eadec22517613d63ac75c2e8b`
- `nethra.py` blob: `3edc7e2b08cf66d2de6177fe00144eed0062c880`

Repair lineage head before this freeze:
- commit: `eb8f2222ae32f27a59094ac9c54d5c7d5e271d7c`
- repaired `nethra.py` blob: `81b4b09816a12a5aaaee0eba5338328f736f4e9c`
- mistake ledger blob: `8ccdecb7f7e8cad85c8f2e23811a6344f0d57c2a`

Frozen branch:
- `nethra-temporal-route-frozen`

## Confirmed regression

Commit `d1ac8127c66c019405eaf4abb330e21236b51cb9` changed temporal admission from separately refindable before/after support routes into one simultaneous union route:

```text
wrong:
R route = previous_closure | current_closed
```

For `A -> B`, this produced only route `{A,B}`. Later `closure({A})` and `closure({B})` could not refind the learned relation, so native recursive composition stalled even though field conduction through the relation still worked.

The indexed closure optimization was not the cause; it reproduced the already-frozen behavior exactly.

## Repair

A learned temporal relation remains one ordinary Nethra. Its complete recursively refound temporal sides are stored as distinct routes:

```text
R:
  route = complete previous closure
  route = complete current closure
```

No proper subsets are enumerated.

Any individual route that contains the relation itself is skipped as tautological. The opposite non-self-containing temporal side remains eligible.

Source-pair indexing, native residual equations, manifestation replay, per-incidence plasticity, source similarity, magnitude-preserving smearing, F61 field equations, indexed closure, and per-interval edge caching are unchanged.

## Validation

Exact repaired source was tested through ordinary `NethraField.step()`; no provisional learner, alternate step, injected learned relation IDs, candidate scanner, evaluator, or harness-level learning semantics were used.

Validation workflow run:
- `35950800664`

Ubuntu:
- temporal sides separate/refindable: PASS
- simple learned sequence field prediction: PASS
- primitive-only recursive depth test: PASS
- depth receipt: `57`
- total Nethra: `169`
- learned Nethra: `112`

Fedora self-hosted:
- temporal sides separate/refindable: PASS
- simple learned sequence field prediction: PASS
- primitive-only recursive depth test: PASS
- depth receipt: `57`
- total Nethra: `169`
- learned Nethra: `112`

The depth curriculum supplied only primitive external Nethra. Learned Nethra had to be recursively refound by closure before they could participate in later learned structure.

## Mistakes logged during repair

The cumulative ledger now includes:
1. the original temporal-side union transcription error;
2. the first repair test demanding depth 50 after only 32 staged recursive additions, which correctly produced depth 33;
3. the repair workflow invoking an immutable-commit comparison from a shallow checkout.

## Freeze decision

The temporal-route representation repair is accepted and frozen.

Canonical source:

```text
nethra.py
blob 81b4b09816a12a5aaaee0eba5338328f736f4e9c
```

Any later source change requires a new test and freeze.
