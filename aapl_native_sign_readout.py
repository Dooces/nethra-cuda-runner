#!/usr/bin/env python3
"""Native sign readout from primed Nethra continuation structure.

Prediction is read before the next price is revealed.

At each scored interval:
  1. preserve the live carried field;
  2. evolve a copy with TIME only for the actual elapsed gap;
  3. ask the already-learned current-event closure what existing relations are specific to a
     hypothetical P+ continuation and what existing relations are specific to a hypothetical P-
     continuation;
  4. compare how strongly those sign-specific continuation Nethra are already primed in the
     TIME-only prospective field;
  5. reveal the real price and let ordinary plasticity/construction continue.

The hypothetical closures do not update topology, evidence, activation, or plasticity.
There is no learned decoder, transition table, fitted threshold, or outcome-fed calibration.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

import aapl_residual_recursive_native as base
from aapl_residual_recursive_native import NativeReplay, fetch_aapl, intervals, integrate

TRAIN = int(os.environ.get("NETHRA_SIGN_TRAIN", "10000"))
ONLINE = int(os.environ.get("NETHRA_SIGN_ONLINE", "10000"))
EPS = 1e-18
FRACS = (0.10, 0.25, 0.50)


def summary(margins, truth, kinds):
    m = np.asarray(margins, np.float64)
    y = np.asarray(truth, np.int8)
    knd = np.asarray(kinds, np.int8)
    resolved = np.abs(m) > EPS
    pred = np.where(m >= 0.0, 1, -1)

    out = {
        "n": int(len(m)),
        "resolved": int(np.sum(resolved)),
        "coverage": float(np.mean(resolved)) if len(m) else 0.0,
        "accuracy": float(np.mean(pred[resolved] == y[resolved])) if np.any(resolved) else None,
        "margin_rms": float(np.sqrt(np.mean(m * m))) if len(m) else 0.0,
    }

    ix = np.flatnonzero(resolved)
    order = ix[np.argsort(-np.abs(m[ix]))] if len(ix) else ix
    for frac in FRACS:
        n = max(1, int(len(order) * frac)) if len(order) else 0
        chosen = order[:n]
        key = int(frac * 100)
        out[f"top_{key}pct_n"] = int(n)
        out[f"top_{key}pct_accuracy"] = (
            float(np.mean(pred[chosen] == y[chosen])) if n else None
        )

    for kval, name in ((0, "open"), (1, "close")):
        q = np.flatnonzero(resolved & (knd == kval))
        out[f"{name}_n"] = int(len(q))
        out[f"{name}_accuracy"] = float(np.mean(pred[q] == y[q])) if len(q) else None
        if len(q):
            qq = q[np.argsort(-np.abs(m[q]))]
            n = max(1, int(len(qq) * 0.10))
            chosen = qq[:n]
            out[f"{name}_top10_n"] = int(n)
            out[f"{name}_top10_accuracy"] = float(np.mean(pred[chosen] == y[chosen]))
        else:
            out[f"{name}_top10_n"] = 0
            out[f"{name}_top10_accuracy"] = None
    return out


def segment(margins, truth, kinds):
    n = len(truth)
    h = n // 2
    return {
        "all": summary(margins, truth, kinds),
        "first_half": summary(margins[:h], truth[:h], kinds[:h]),
        "second_half": summary(margins[h:], truth[h:], kinds[h:]),
    }


def score_set(values, members):
    if not members:
        return 0.0, 0.0, 0.0
    vals = [float(values[i]) for i in members]
    return max(vals), sum(vals), sum(vals) / len(vals)


def main():
    base.FAST_SYNC = True
    points = fetch_aapl()
    currents, elapsed, kinds, raw = intervals(points)
    need = TRAIN - 1 + ONLINE
    if need > len(currents):
        raise RuntimeError(f"need {need} intervals, have {len(currents)}")

    model = NativeReplay()
    model.reset_transient()

    for i in range(TRAIN - 1):
        model.interval(float(currents[i]), float(elapsed[i]), True, True)

    train_relations = len(model.birth_members)
    train_depth = max(model.depth.values(), default=0)

    margins = {
        "activation_max": [],
        "activation_sum": [],
        "activation_mean": [],
        "gain_max": [],
        "gain_sum": [],
        "gain_mean": [],
    }
    truth = []
    scored_kinds = []
    plus_counts = []
    minus_counts = []

    for i in range(TRAIN - 1, TRAIN - 1 + ONLINE):
        if model.topology_dirty:
            model.sync_topology()
        if len(model.state) != len(model.f.nethra):
            model._sync_indices()

        start = model.state.copy()
        prospect = start.copy()
        ext = np.zeros(len(prospect), np.float64)
        ext[model.index[model.time]] = 1.0
        integrate(prospect, ext, model.er, model.em, model.ee, float(elapsed[i]))
        gain = prospect - start

        prev = set(model.f.previous_closure)

        plus_closed, _ = model.recursive_current_closure(
            frozenset((model.time, model.pplus))
        )
        minus_closed, _ = model.recursive_current_closure(
            frozenset((model.time, model.pminus))
        )

        plus_only = {
            model.index[r]
            for r in plus_closed
            if r in model.birth_members and r not in minus_closed and r not in prev
        }
        minus_only = {
            model.index[r]
            for r in minus_closed
            if r in model.birth_members and r not in plus_closed and r not in prev
        }

        plus_counts.append(len(plus_only))
        minus_counts.append(len(minus_only))

        pa_max, pa_sum, pa_mean = score_set(prospect, plus_only)
        ma_max, ma_sum, ma_mean = score_set(prospect, minus_only)
        pg_max, pg_sum, pg_mean = score_set(gain, plus_only)
        mg_max, mg_sum, mg_mean = score_set(gain, minus_only)

        margins["activation_max"].append(pa_max - ma_max)
        margins["activation_sum"].append(pa_sum - ma_sum)
        margins["activation_mean"].append(pa_mean - ma_mean)
        margins["gain_max"].append(pg_max - mg_max)
        margins["gain_sum"].append(pg_sum - mg_sum)
        margins["gain_mean"].append(pg_mean - mg_mean)

        truth.append(1 if currents[i] >= 0.0 else -1)
        scored_kinds.append(int(kinds[i]))

        # Only now reveal the actual next price and permit normal learning/construction.
        model.interval(float(currents[i]), float(elapsed[i]), True, True)

    y = np.asarray(truth, np.int8)
    knd = np.asarray(scored_kinds, np.int8)

    result = {
        "training_prices": TRAIN,
        "train_intervals": TRAIN - 1,
        "online_intervals": ONLINE,
        "training_start": points[0].date,
        "training_end": points[TRAIN - 1].date,
        "online_end": points[TRAIN - 1 + ONLINE].date,
        "learning_live": True,
        "construction_live": True,
        "relations_train": train_relations,
        "relations_final": len(model.birth_members),
        "depth_train": train_depth,
        "depth_final": max(model.depth.values(), default=0),
        "actual_up_fraction": float(np.mean(y == 1)),
        "mean_plus_specific_relations": float(np.mean(plus_counts)),
        "mean_minus_specific_relations": float(np.mean(minus_counts)),
        "steps_with_both_sign_sets": int(sum(a > 0 and b > 0 for a, b in zip(plus_counts, minus_counts))),
        "steps_with_any_sign_set": int(sum(a > 0 or b > 0 for a, b in zip(plus_counts, minus_counts))),
        "readouts": {
            name: segment(vals, y, knd)
            for name, vals in margins.items()
        },
    }

    print("RESULT", json.dumps(result, sort_keys=True), flush=True)
    print("all_assertions_passed", flush=True)


if __name__ == "__main__":
    main()
