# Nethra indexed execution optimization freeze

Date: 2026-09-23 / 2026-09-24 UTC

## Frozen ancestry

This optimization started from the immutable smeared-source baseline:

- baseline commit: `8abeaf358b34aed6b18e10cc0f7fc17de6e7db93`
- baseline `nethra.py` blob: `5cef34cf74fe44e1e19413864aa7be758bf3d96c`

The validated optimization candidate is:

- validated commit: `636c8e9df5eb1a89224c36c21fef1e50ee05293c`
- optimized `nethra.py` blob: `3edc7e2b08cf66d2de6177fe00144eed0062c880`
- mistake ledger blob: `c11482b21abf76a0289f90d69ab21f903a941566`
- validation workflow run: `35946766037`

The frozen branch is `nethra-indexed-optimizations-frozen`.

## Scope of the source change

The source change is execution-only.

### Indexed recursive closure

A derived reverse index is maintained:

```text
member_to_routeuses[member] -> {(relation, route), ...}
```

Route registration through `_route()` updates this index.

Closure now:

1. visits state-qualified routes only when they are incident to a member represented in the fixed transient event;
2. visits unqualified routes only when a member becomes active;
3. counts active route members;
4. activates a relation when the same frozen route condition would have succeeded;
5. queues the newly active relation so recursive relations and cycles continue to fixed point.

The index has no activation, evidence, learning authority, semantic identity, or independent persistence meaning. It is derived from `Nethra.routes`.

### Per-interval field compilation

`step()` compiles the current edge tuple and neighbor map once while topology, evidence, and `current_event` are fixed.

Those exact structures are reused for:

- all four zero-source RK4 derivative evaluations;
- all four actual-source RK4 derivative evaluations;
- the residual-pair neighbor calculation after the outcome.

The cache ends at the interval boundary. Evidence/topology changes occur after the physical outcome.

### Explicitly excluded changes

No `g_epsilon` pruning was added.

No persistent active-edge cache was added.

No threshold, learning equation, source similarity rule, admission rule, evidence equation, F61 equation, source transduction rule, topology rule, or construction rule was changed.

## Exact-equivalence tests

Final workflow run `35946766037` passed on both Ubuntu Python 3.12 and the Fedora self-hosted Python 3.14 runner.

10/10 tests passed on each platform:

1. cached derivative equals uncached derivative exactly on the same field state;
2. cached neighbor map equals the directly compiled neighbor map;
3. cached residual-neighbor update equals direct residual update exactly;
4. cached RK4 zero-source and actual-source replays equal uncached replays exactly;
5. indexed closure equals the frozen global-scan closure exactly on the same field object over 200 randomized worlds;
6. independently instantiated frozen/optimized closure agrees on randomized mixed route/state cases;
7. a 1,200-route disconnected closure stress confirms the indexed path does not scan unrelated route dictionaries;
8. reverse-route index remains exactly derivable from `routes` through 320 live native-construction steps;
9. state-qualified transient-event refinding remains equivalent;
10. inherited historical tests retain the same pass/fail/error status map as the exact frozen baseline, preventing stale tests from being misread as current semantic requirements.

The optimized `step()` compiles `_edges()` once in the measured interval. The frozen implementation rebuilt it at least eight times.

## Performance receipts

### Fedora self-hosted

Physics benchmark:
- 900 relations
- fan-in 4
- 12 steps/sample
- frozen: 0.5905069590 s
- optimized: 0.3168253420 s
- speedup: 1.8638248925x

Sparse closure benchmark:
- 3,500 unrelated routes
- relevant recursive chain length 12
- 80 closures/sample
- frozen: 0.1773297880 s
- optimized: 0.0003112740 s
- speedup: 569.6903281x

### Ubuntu portable

Physics benchmark:
- frozen: 1.3536030140 s
- optimized: 0.6987990490 s
- speedup: 1.9370418662x

Sparse closure benchmark:
- frozen: 0.3648491250 s
- optimized: 0.0007252380 s
- speedup: 503.0750250x

These timings measure the exact frozen core and optimized core through the same benchmark harness.

## Mistakes made during this optimization

Two task mistakes were appended to `NETHRA_MISTAKE_LEDGER.md`.

First, the initial workflow treated two stale historical test files as current pass/fail authority. Running those files against the pinned frozen commit showed that their failing assertions referenced retired machinery already absent from the frozen baseline. The gate was corrected to frozen-baseline status parity plus direct optimization-equivalence tests.

Second, the first cross-module trajectory test demanded extremely tight floating equality across separately instantiated identity-hashed object graphs. Different set/frozenset iteration orders produced tiny floating-order differences that recurrent dynamics could amplify. The corrected tests compare the changed execution paths exactly on the same field state, while structural identities are checked separately.

## Expected assistant failure modes and guards

These are process risks grounded in the existing ledger and this task. They are not claims of new Nethra mechanism failures.

- **Selecting an old file or branch.** Guard: begin from an exact immutable commit and verify the `nethra.py` blob before work.
- **Trusting a branch name as semantic provenance.** Guard: record commit, source blob, and ledger blob together.
- **Treating an old test filename as current authority.** Guard: run inherited tests against the pinned baseline before interpreting candidate failures.
- **Letting a derived cache become semantic state.** Guard: every optimization index/cache must be reconstructible from Nethra topology/evidence and carry no independent authority.
- **Caching conductance across event changes.** Guard: the current optimization cache lives only inside one interval where `current_event` and evidence are fixed.
- **Adding conductance pruning because an edge is “small.”** Guard: `g_epsilon` remains absent; adding it would be an explicit semantic change requiring its own evidence and freeze.
- **Forgetting reverse-index maintenance when adding a new route-writing path.** Guard: route creation currently goes through `_route()`; the live-construction integrity test derives the index from `routes` and demands exact equality.
- **Using a benchmark harness that replaces core semantics.** Guard: this benchmark calls the unchanged public/core execution path and loads the frozen comparison source directly from its commit.
- **Mistaking floating-order drift between separate object graphs for semantic divergence.** Guard: changed operations are tested exactly on the same field state.
- **Changing source after validation.** Guard: the frozen source blob is `3edc7e2b08cf66d2de6177fe00144eed0062c880`; any different blob requires fresh validation and a new freeze.

## Freeze decision

The indexed closure and per-interval field compilation are accepted as behavior-preserving execution optimizations for this baseline.

The canonical source for this freeze is exactly:

```text
nethra.py
blob 3edc7e2b08cf66d2de6177fe00144eed0062c880
```

Any later semantic or execution change requires a new tested commit and freeze record.
