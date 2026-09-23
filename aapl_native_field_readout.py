#!/usr/bin/env python3
"""Read the live Nethra field directly before each next AAPL observation.

No external predictor, classifier, transition table, route lookup, or frozen learning.

For every scored interval:
  1. preserve the carried field left by the preceding real observation;
  2. evolve a copy with TIME current only for the actual elapsed wall-clock gap;
  3. read P+ and P- directly from that prospective field;
  4. reveal the true next price delta;
  5. advance the real model through the same interval with plasticity and construction live.

Three native readouts are reported:
  activation:       prospective P+ activation - prospective P- activation
  time_gain:        TIME-induced change in P+ - TIME-induced change in P-
  incoming_flow:    prospective conductive flow into P+ - flow into P-

The first two are literal field state. The third is the existing local field-flow readout.
"""

from __future__ import annotations

import json
import os
import statistics

import numpy as np

import aapl_residual_recursive_native as base
from aapl_residual_recursive_native import NativeReplay, fetch_aapl, intervals, integrate, preflow

TRAIN = int(os.environ.get("NETHRA_FIELD_TRAIN", "1200"))
ONLINE = int(os.environ.get("NETHRA_FIELD_ONLINE", "5000"))
EPS = 1e-18
FRACS = (0.10, 0.25, 0.50)


def summarize(margins, truth, kinds):
    margins = np.asarray(margins, np.float64)
    truth = np.asarray(truth, np.int8)
    kinds = np.asarray(kinds, np.int8)

    resolved = np.abs(margins) > EPS
    pred = np.where(margins >= 0.0, 1, -1)

    out = {
        "n": int(len(margins)),
        "resolved": int(np.sum(resolved)),
        "coverage": float(np.mean(resolved)) if len(margins) else 0.0,
        "accuracy": float(np.mean(pred[resolved] == truth[resolved])) if np.any(resolved) else None,
        "margin_rms": float(np.sqrt(np.mean(margins * margins))) if len(margins) else 0.0,
    }

    resolved_ix = np.flatnonzero(resolved)
    order = resolved_ix[np.argsort(-np.abs(margins[resolved_ix]))] if len(resolved_ix) else resolved_ix
    for frac in FRACS:
        k = max(1, int(len(order) * frac)) if len(order) else 0
        ix = order[:k]
        out[f"top_{int(frac*100)}pct_n"] = int(k)
        out[f"top_{int(frac*100)}pct_accuracy"] = (
            float(np.mean(pred[ix] == truth[ix])) if k else None
        )

    for kind, name in ((0, "open"), (1, "close")):
        ix = np.flatnonzero(resolved & (kinds == kind))
        out[f"{name}_n"] = int(len(ix))
        out[f"{name}_accuracy"] = float(np.mean(pred[ix] == truth[ix])) if len(ix) else None
        if len(ix):
            ranked = ix[np.argsort(-np.abs(margins[ix]))]
            k = max(1, int(len(ranked) * 0.10))
            top = ranked[:k]
            out[f"{name}_top10_n"] = int(k)
            out[f"{name}_top10_accuracy"] = float(np.mean(pred[top] == truth[top]))
        else:
            out[f"{name}_top10_n"] = 0
            out[f"{name}_top10_accuracy"] = None

    return out


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

    activation = []
    time_gain = []
    incoming_flow = []
    truth = []
    scored_kinds = []
    nonzero_start = 0

    pplus = model.pplus
    pminus = model.pminus
    time_node = model.time

    for i in range(TRAIN - 1, TRAIN - 1 + ONLINE):
        if model.topology_dirty:
            model.sync_topology()
        if len(model.state) != len(model.f.nethra):
            model._sync_indices()

        ip = model.index[pplus]
        im = model.index[pminus]
        it = model.index[time_node]

        start = model.state.copy()
        if float(np.linalg.norm(start)) > EPS:
            nonzero_start += 1

        prospect = start.copy()
        ext = np.zeros(len(prospect), np.float64)
        ext[it] = 1.0
        integrate(prospect, ext, model.er, model.em, model.ee, float(elapsed[i]))

        activation.append(float(prospect[ip] - prospect[im]))
        time_gain.append(float((prospect[ip] - start[ip]) - (prospect[im] - start[im])))

        _pout, _pin, pred, _supply = preflow(prospect, model.er, model.em, model.ee)
        incoming_flow.append(float(pred[ip] - pred[im]))

        truth.append(1 if currents[i] >= 0.0 else -1)
        scored_kinds.append(int(kinds[i]))

        # Reveal only after every prospective readout has been recorded.
        model.interval(float(currents[i]), float(elapsed[i]), True, True)

    truth_arr = np.asarray(truth, np.int8)
    result = {
        "training_prices": TRAIN,
        "train_intervals": TRAIN - 1,
        "online_intervals": ONLINE,
        "training_start": points[0].date,
        "training_end": points[TRAIN - 1].date,
        "online_end": points[TRAIN - 1 + ONLINE].date,
        "learning_live": True,
        "construction_live": True,
        "nonzero_start_fraction": nonzero_start / ONLINE,
        "relations_train": train_relations,
        "relations_final": len(model.birth_members),
        "depth_train": train_depth,
        "depth_final": max(model.depth.values(), default=0),
        "actual_up_fraction": float(np.mean(truth_arr == 1)),
        "activation": summarize(activation, truth_arr, scored_kinds),
        "time_gain": summarize(time_gain, truth_arr, scored_kinds),
        "incoming_flow": summarize(incoming_flow, truth_arr, scored_kinds),
    }

    print("RESULT", json.dumps(result, sort_keys=True), flush=True)
    print("all_assertions_passed", flush=True)


if __name__ == "__main__":
    main()
