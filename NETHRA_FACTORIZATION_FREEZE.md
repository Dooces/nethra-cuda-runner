# Nethra recursive factorization repair freeze

Previous temporal-route freeze:
- commit: `c0d7ac1a49d86c936ecd76e159d40834983fa7ed`
- nethra.py blob: `81b4b09816a12a5aaaee0eba5338328f736f4e9c`

Factorization repair test head:
- commit: `7bdfef4e7ac8cf73322e88b29b9bdf031567001e`
- nethra.py blob: `472d784af40dfebdbd993e4e3fc327eaade8bcc1`
- mistake ledger blob: `fb642e6c2c8eec7dc1a7db9f18dda35ded541ea7`

Frozen branch:
- `nethra-factorization-frozen`

## Restored rule

Recursive primitive factorization is used only for reuse/duplicate lookup.

For proposed recursive support S:

```text
canonical(S) = union primitive_leaves(member) for member in S
```

Stored persistent routes remain the original recursive members. They are never flattened.

A factor-equivalent description may reuse an existing relation only when that relation is already indexed by the same canonical source transition. Therefore primitive-domain equality is a reuse hint rather than universal semantic identity.

Factor-equivalent alternate parenthesizations do not add duplicate routes to the reused relation. Cross-domain support remains eligible to earn a new route through ordinary evidence. A different source transition is not collapsed merely because primitive support factors the same way.

Legal cycles are handled by finite grounded factorization. Back-edges contribute no repeated leaves; a completely ungrounded cycle remains opaque rather than being falsely merged.

## Regression tests

Workflow run: `35951923586`

Fedora self-hosted: PASS
Ubuntu: PASS

Seven tests passed on each platform:
- alternate recursive parenthesization reuses the same relation;
- factorization is not universal identity;
- grounded cyclic factorization terminates;
- many nested descriptions do not mint duplicate semantic relations;
- native primitive-only recursion still exceeds depth 50;
- temporal sides remain separately refindable;
- simple learned sequence prediction remains functional.

Depth receipt on Fedora:
- depth: 57
- total Nethra: 169
- learned Nethra: 112

## Freeze

Canonical source:
```text
nethra.py
blob 472d784af40dfebdbd993e4e3fc327eaade8bcc1
```

Any later source modification requires a new test and freeze.
