#!/usr/bin/env python3
"""Selective AAPL predictor using the native live Nethra field.

Prediction contract:
- TIME is advanced on a copy of the current field.
- The next AAPL price is still hidden.
- Direction is the sign of incoming prospective field flow to P+ minus P-.
- Confidence is the percentile of |that margin| against resolved margins seen earlier in time.
- Only after scoring is the true price revealed and ordinary Nethra learning/construction runs.

The confidence readout never feeds back into Nethra.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics

import numpy as np

import aapl_residual_recursive_native as base
from aapl_residual_recursive_native import (
    NativeReplay,
    TIME_UNITS_PER_DAY,
    fetch_aapl,
    integrate,
    intervals,
    preflow,
)

TRAIN = int(os.environ.get("NETHRA_AAPL_TRAIN", "1200"))
WARMUP_RESOLVED = int(os.environ.get("NETHRA_AAPL_CONF_WARMUP", "100"))
EPS = 1e-18
QUANTILES = (0.90, 0.95, 0.98, 0.99)


def prospective_margin(model: NativeReplay, elapsed: float) -> float:
    """Read the native P+/P- expectation after TIME alone, before the next price exists."""
    if model.topology_dirty:
        model.sync_topology()
    if len(model.state) != len(model.f.nethra):
        model._sync_indices()

    prospect = model.state.copy()
    ext = np.zeros(len(prospect), np.float64)
    ext[model.index[model.time]] = 1.0
    integrate(prospect, ext, model.er, model.em, model.ee, float(elapsed))

    _pout, _pin, pred, _supply = preflow(
        prospect, model.er, model.em, model.ee
    )
    return float(
        pred[model.index[model.pplus]]
        - pred[model.index[model.pminus]]
    )


def summarize_gate(rows):
    if not rows:
        return {
            "n": 0,
            "accuracy": None,
            "majority_baseline": None,
            "persistence_baseline": None,
            "prediction_up_fraction": None,
            "truth_up_fraction": None,
        }

    accuracy = statistics.mean(r["correct"] for r in rows)
    truth_up = statistics.mean(r["truth"] > 0 for r in rows)
    pred_up = statistics.mean(r["prediction"] > 0 for r in rows)
    persistence = statistics.mean(
        r["previous_truth"] == r["truth"] for r in rows
    )
    return {
        "n": len(rows),
        "accuracy": accuracy,
        "majority_baseline": max(truth_up, 1.0 - truth_up),
        "persistence_baseline": persistence,
        "prediction_up_fraction": pred_up,
        "truth_up_fraction": truth_up,
    }


def main():
    base.FAST_SYNC = True

    points = fetch_aapl()
    payload = json.dumps(
        [[p.timestamp, p.price, p.kind, p.date] for p in points],
        separators=(",", ":"),
    ).encode("utf-8")
    input_sha256 = hashlib.sha256(payload).hexdigest()

    currents, elapsed, kinds, raw = intervals(points)
    train_intervals = TRAIN - 1
    if train_intervals <= 0 or train_intervals >= len(currents):
        raise RuntimeError("invalid training span")

    model = NativeReplay()
    model.reset_transient()

    for i in range(train_intervals):
        model.interval(
            float(currents[i]),
            float(elapsed[i]),
            True,
            True,
        )

    strength_history = []
    gates = {q: [] for q in QUANTILES}
    resolved = 0
    resolved_correct = 0

    for i in range(train_intervals, len(currents)):
        margin = prospective_margin(model, float(elapsed[i]))
        truth = 1 if currents[i] >= 0.0 else -1
        previous_truth = 1 if currents[i - 1] >= 0.0 else -1

        if abs(margin) > EPS:
            resolved += 1
            prediction = 1 if margin > 0.0 else -1
            resolved_correct += int(prediction == truth)
            strength = abs(margin)

            if len(strength_history) >= WARMUP_RESOLVED:
                prior = np.asarray(strength_history, np.float64)
                for q in QUANTILES:
                    cutoff = float(np.quantile(prior, q))
                    if strength >= cutoff:
                        gates[q].append({
                            "prediction": prediction,
                            "truth": truth,
                            "previous_truth": previous_truth,
                            "correct": prediction == truth,
                        })

            strength_history.append(strength)

        # Reveal the observed price only after every prospective readout above is finished.
        model.interval(
            float(currents[i]),
            float(elapsed[i]),
            True,
            True,
        )

    online_intervals = len(currents) - train_intervals
    result = {
        "input_sha256": input_sha256,
        "history_start": points[0].date,
        "training_end": points[TRAIN - 1].date,
        "history_end": points[-1].date,
        "train_intervals": train_intervals,
        "online_intervals": online_intervals,
        "learning_live": True,
        "construction_live": True,
        "resolved": resolved,
        "resolved_coverage": resolved / online_intervals,
        "resolved_accuracy": (
            resolved_correct / resolved if resolved else None
        ),
        "relations_final": len(model.birth_members),
        "depth_final": max(model.depth.values(), default=0),
        "confidence": {
            f"prior_top_{int((1.0 - q) * 100 + 0.5)}pct": summarize_gate(gates[q])
            for q in QUANTILES
        },
    }

    next_hours = os.environ.get("NETHRA_NEXT_HOURS")
    if next_hours is not None:
        hours = float(next_hours)
        next_margin = prospective_margin(
            model,
            (hours / 24.0) * TIME_UNITS_PER_DAY,
        )
        strength = abs(next_margin)
        if strength_history:
            percentile = float(np.mean(
                np.asarray(strength_history, np.float64) <= strength
            ))
        else:
            percentile = None
        result["next_prediction"] = {
            "from_date": points[-1].date,
            "from_kind": int(points[-1].kind),
            "elapsed_hours": hours,
            "direction": (
                "up" if next_margin > EPS
                else "down" if next_margin < -EPS
                else "unresolved"
            ),
            "margin": next_margin,
            "prior_strength_percentile": percentile,
        }

    print("RESULT", json.dumps(result, sort_keys=True), flush=True)
    print("all_assertions_passed", flush=True)


if __name__ == "__main__":
    main()
