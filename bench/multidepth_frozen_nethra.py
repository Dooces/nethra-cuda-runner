#!/usr/bin/env python3
"""Parallel Fedora probe: frozen Nethra prediction under fixed scalar quantization depths.

The quantizer is test equipment.  Each worker receives one fixed depth for the entire run.
Nothing inspects prediction quality to change depth, thresholds, routes, or Nethra state.
Training uses the frozen Nethra observe()/construction path.  Evaluation drives only the learned
Nethra field and scores it externally.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import math
import os
import random
import resource
import time

from nethra import NethraField


DEPTHS = (0, 1, 2, 3, 4, 5, 6, 8)
CASES = ("broad", "narrow", "islands", "smooth")
NTRAIN = int(os.environ.get("NETHRA_DEPTH_NTRAIN", "4000"))
NTEST = int(os.environ.get("NETHRA_DEPTH_NTEST", "12000"))
FIELD_STEPS = int(os.environ.get("NETHRA_DEPTH_FIELD_STEPS", "10"))
FIELD_DT = float(os.environ.get("NETHRA_DEPTH_FIELD_DT", "0.05"))


def outcome_probability(kind, x):
    if kind == "broad":
        return 0.90 if x >= 0.5 else 0.10
    if kind == "narrow":
        return 0.90 if 0.47 <= x < 0.53 else 0.10
    if kind == "islands":
        return 0.90 if (0.15 <= x < 0.35 or 0.65 <= x < 0.85) else 0.10
    if kind == "smooth":
        return 0.10 + 0.80 * x
    raise ValueError(kind)


def rows(kind, n, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        x = rng.random()
        y = int(rng.random() < outcome_probability(kind, x))
        out.append((x, y))
    return out


def qindex(x, depth):
    bins = 1 << depth
    return min(int(x * bins), bins - 1)


def independent_episode(field, source, outcome):
    """Present one source->consequence episode while retaining accumulated Nethra evidence.

    Resetting only the three transient interval cursors makes samples independent physical trials.
    It does not clear topology, route evidence, prospective counts, field statistics, or Nethra.
    """
    field.previous_closure = frozenset()
    field.previous_event = frozenset()
    field.current_event = frozenset()
    field.observe((source,))
    field.observe((source, outcome))


def integrate_without_learning(field, source, steps=FIELD_STEPS, dt=FIELD_DT):
    """Drive learned topology with source current using the frozen F61 derivative only."""
    for n in field.nethra:
        n.activation = 0.0
        n.external = 0.0

    source.external = 1.0
    field.current_event = frozenset(((source, +1),))

    for _ in range(steps):
        a0 = {n: n.activation for n in field.nethra}
        k1 = field._derivative_at(a0)
        a1 = {n: a0[n] + 0.5 * dt * k1[n] for n in field.nethra}
        k2 = field._derivative_at(a1)
        a2 = {n: a0[n] + 0.5 * dt * k2[n] for n in field.nethra}
        k3 = field._derivative_at(a2)
        a3 = {n: a0[n] + dt * k3[n] for n in field.nethra}
        k4 = field._derivative_at(a3)
        for n in field.nethra:
            n.activation = a0[n] + dt * (
                k1[n] + 2.0 * k2[n] + 2.0 * k3[n] + k4[n]
            ) / 6.0

    source.external = 0.0


def run_case(depth, kind):
    train = rows(kind, NTRAIN, 100_000 + len(kind) * 100 + depth)
    test = rows(kind, NTEST, 200_000 + len(kind) * 100)

    field = NethraField()
    sources = [field.new() for _ in range(1 << depth)]
    y0 = field.new()
    y1 = field.new()

    t0 = time.perf_counter()
    for x, y in train:
        independent_episode(field, sources[qindex(x, depth)], y1 if y else y0)
    train_s = time.perf_counter() - t0

    base_ones = sum(y for _, y in test)
    baseline_accuracy = max(base_ones, len(test) - base_ones) / len(test)

    t1 = time.perf_counter()
    margins = []
    for source in sources:
        integrate_without_learning(field, source)
        margins.append(y1.activation - y0.activation)
    eval_s = time.perf_counter() - t1

    correct = 0
    resolved = 0
    ties = 0
    margin_abs_sum = 0.0
    for x, y in test:
        margin = margins[qindex(x, depth)]
        margin_abs_sum += abs(margin)
        if abs(margin) <= 1e-15:
            ties += 1
            continue
        pred = int(margin > 0.0)
        resolved += 1
        correct += int(pred == y)

    weighted_accuracy = (correct + 0.5 * ties) / len(test)
    resolved_accuracy = correct / resolved if resolved else None

    input_count = len(sources) + 2
    relation_count = len(field.nethra) - input_count
    route_count = sum(len(n.routes) for n in field.nethra)
    edge_count = len(field._edges())
    learned_bins = sum(abs(m) > 1e-15 for m in margins)

    return {
        "case": kind,
        "depth": depth,
        "bins": 1 << depth,
        "ntrain": NTRAIN,
        "ntest": NTEST,
        "baseline_accuracy": baseline_accuracy,
        "weighted_accuracy": weighted_accuracy,
        "resolved_accuracy": resolved_accuracy,
        "unresolved_fraction": ties / len(test),
        "learned_bins": learned_bins,
        "mean_abs_field_margin": margin_abs_sum / len(test),
        "relations": relation_count,
        "routes": route_count,
        "edges": edge_count,
        "histories": len(field.history_relation),
        "prospective_rows": len(field.history_count),
        "train_seconds": train_s,
        "field_eval_seconds": eval_s,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


def run_depth(depth):
    started = time.perf_counter()
    cases = [run_case(depth, kind) for kind in CASES]
    return {
        "depth": depth,
        "pid": os.getpid(),
        "elapsed_seconds": time.perf_counter() - started,
        "cases": cases,
    }


def main():
    started = time.perf_counter()
    workers = min(len(DEPTHS), max(1, os.cpu_count() or 1))
    print(json.dumps({
        "event": "start",
        "depths": DEPTHS,
        "cases": CASES,
        "workers": workers,
        "cpu_count": os.cpu_count(),
        "ntrain": NTRAIN,
        "ntest": NTEST,
        "field_steps": FIELD_STEPS,
        "field_dt": FIELD_DT,
    }), flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_depth = {pool.submit(run_depth, d): d for d in DEPTHS}
        for fut in as_completed(future_depth):
            result = fut.result()
            results.append(result)
            print(json.dumps({"event": "depth_complete", **result}), flush=True)

    results.sort(key=lambda r: r["depth"])
    payload = {
        "contract": "NETHRA_FIXED_DEPTH_FEDORA_PARALLEL_PROBE_1",
        "frozen_core_commit": "721870d516a82eb29b722dc67c937f140c54038e",
        "frozen_core_blob": "944fbd1f34ad4042fbf797824c19ed5cf2e97d43",
        "depths": list(DEPTHS),
        "cases": list(CASES),
        "workers": workers,
        "ntrain": NTRAIN,
        "ntest": NTEST,
        "field_steps": FIELD_STEPS,
        "field_dt": FIELD_DT,
        "wall_seconds": time.perf_counter() - started,
        "results": results,
    }
    Path("multidepth_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("=== SUMMARY ===")
    for result in results:
        for row in result["cases"]:
            print(
                f"depth={row['depth']:2d} bins={row['bins']:4d} "
                f"case={row['case']:7s} "
                f"acc={row['weighted_accuracy']:.4f} "
                f"resolved={row['resolved_accuracy'] if row['resolved_accuracy'] is not None else 'NA'} "
                f"unresolved={row['unresolved_fraction']:.4f} "
                f"relations={row['relations']:4d} "
                f"train_s={row['train_seconds']:.3f} "
                f"eval_s={row['field_eval_seconds']:.3f} "
                f"rss_mb={row['max_rss_mb']:.1f}"
            )
    print(f"parallel_wall_seconds={payload['wall_seconds']:.3f}")
    print("all_depths_complete")


if __name__ == "__main__":
    main()
