import json
import statistics
import subprocess
import time
import types

import nethra as optimized

BASE_COMMIT = "8abeaf358b34aed6b18e10cc0f7fc17de6e7db93"


def load_baseline():
    source = subprocess.check_output(["git", "show", f"{BASE_COMMIT}:nethra.py"], text=True)
    module = types.ModuleType("nethra_frozen_baseline_bench")
    module.__file__ = f"{BASE_COMMIT}:nethra.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


baseline = load_baseline()


def median_runtime(fn, repeats=5):
    rows = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        rows.append(time.perf_counter() - start)
    return statistics.median(rows)


def build_physics(module, relations=900, fanin=4):
    f = module.NethraField(native_learning=False, convergence_gain=.7)
    leaves = [f.new() for _ in range(180)]
    for i in range(relations):
        r = f.new()
        members = tuple(leaves[(i * 11 + j * 29) % len(leaves)] for j in range(fanin))
        f._route(r, members, frozenset(), 15 + (i % 30))
    return f, leaves


def physics_batch(module):
    f, leaves = build_physics(module)

    def run():
        for k in range(12):
            leaves[(k * 13) % len(leaves)].push(.37 + .01 * (k % 3))
            f.step(.02)

    return run


def build_closure(module, unrelated=3500, chain_len=12):
    f = module.NethraField(native_learning=False)
    seed = f.new()
    chain = seed
    for _ in range(chain_len):
        r = f.new()
        f._route(r, (chain,), frozenset(), 4)
        chain = r

    leaves = [f.new() for _ in range(400)]
    for i in range(unrelated):
        r = f.new()
        a = leaves[(i * 7) % len(leaves)]
        b = leaves[(i * 17 + 3) % len(leaves)]
        f._route(r, (a, b), frozenset(), 2)
    return f, seed, chain


def closure_batch(module):
    f, seed, last = build_closure(module)

    def run():
        for _ in range(80):
            closed = f.closure((seed,), frozenset())
            if last not in closed:
                raise RuntimeError("closure lost the indexed chain")

    return run


old_physics = median_runtime(physics_batch(baseline), 5)
new_physics = median_runtime(physics_batch(optimized), 5)
old_closure = median_runtime(closure_batch(baseline), 5)
new_closure = median_runtime(closure_batch(optimized), 5)

print(json.dumps({
    "baseline_commit": BASE_COMMIT,
    "physics": {
        "relations": 900,
        "fanin": 4,
        "steps_per_sample": 12,
        "baseline_seconds": old_physics,
        "optimized_seconds": new_physics,
        "speedup": old_physics / new_physics if new_physics else None,
    },
    "closure": {
        "unrelated_routes": 3500,
        "chain_length": 12,
        "closures_per_sample": 80,
        "baseline_seconds": old_closure,
        "optimized_seconds": new_closure,
        "speedup": old_closure / new_closure if new_closure else None,
    },
}, indent=2))
