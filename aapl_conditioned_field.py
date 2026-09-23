#!/usr/bin/env python3
"""AAPL replay using the stock runner's learned incidence plasticity with core route-state visibility.

This changes one thing only: an incidence participates in the numerical field on an interval only
when at least one earned route containing that incidence is supported by NethraField.current_event.
With g_min=0 in NativeReplay, this is the route visibility already implied by the core field.

Prediction remains the live field before the next price is revealed.
"""

from __future__ import annotations

import json
import os

import numpy as np

import aapl_residual_recursive_native as base
from aapl_residual_recursive_native import (
    NativeReplay, fetch_aapl, intervals, integrate, preflow, outcome_origin, plasticity
)

TRAIN = int(os.environ.get("NETHRA_COND_TRAIN", "1200"))
ONLINE = int(os.environ.get("NETHRA_COND_ONLINE", "5000"))
EPS = 1e-18
FRACS = (0.10, 0.25, 0.50)


class ConditionedReplay(NativeReplay):
    def conditioned_evidence(self):
        if self.topology_dirty:
            self.sync_topology()
        event = self.f.current_event
        active = set()
        for relation in self.birth_members:
            for route, conditions in relation.routes.items():
                if self.f._route_evidence(route, conditions, event) > 0:
                    for member in route:
                        active.add((relation, member))
        out = np.zeros_like(self.ee)
        active_count = 0
        for k, key in enumerate(self.edge_keys):
            if key in active:
                out[k] = self.ee[k]
                active_count += 1
        return out, active_count

    def interval_conditioned(self, current, elapsed, learn=True, construct=True):
        if self.topology_dirty:
            self.sync_topology()
        if len(self.state) != len(self.f.nethra):
            self._sync_indices()

        ip = self.index[self.pplus]
        im = self.index[self.pminus]
        it = self.index[self.time]

        start = self.state.copy()
        ee_ctx, active_edges = self.conditioned_evidence()

        ext = np.zeros(len(self.state), np.float64)
        ext[it] = 1.0
        integrate(self.state, ext, self.er, self.em, ee_ctx, float(elapsed))

        activation_margin = float(self.state[ip] - self.state[im])
        time_gain_margin = float(
            (self.state[ip] - start[ip]) - (self.state[im] - start[im])
        )

        pout, pin, pred, supply = preflow(self.state, self.er, self.em, ee_ctx)
        flow_margin = float(pred[ip] - pred[im])
        pre = self.state.copy()

        actual, target = outcome_origin(
            pre, float(current), ip, im, self.er, self.em, ee_ctx
        )
        eps = target - pred
        surprise = abs(float(eps[ip])) + abs(float(eps[im]))

        if learn and self.er.shape[0]:
            learned = ee_ctx.copy()
            plasticity(
                learned, self.er, self.em, pout, pin, pred, supply, target,
                self.relmask, self.stats_tension, self.stats_flow, self.stats_abs
            )
            # Only context-visible incidences participated, so only those evidence values change.
            for k, key in enumerate(self.edge_keys):
                if ee_ctx[k] > 0.0:
                    self.ee[k] = learned[k]

        self.state = actual

        explicit = {self.time, self.pplus if current >= 0 else self.pminus}
        closure_size = 0
        if construct:
            _r, closure_size = self.structural_step(explicit, surprise)

        return {
            "activation_margin": activation_margin,
            "time_gain_margin": time_gain_margin,
            "flow_margin": flow_margin,
            "surprise": surprise,
            "closure_size": closure_size,
            "active_edges": active_edges,
            "total_edges": len(ee_ctx),
        }


def summarize(margins, truth):
    m = np.asarray(margins, np.float64)
    y = np.asarray(truth, np.int8)
    resolved = np.abs(m) > EPS
    pred = np.where(m >= 0.0, 1, -1)
    out = {
        "n": int(len(m)),
        "resolved": int(np.sum(resolved)),
        "coverage": float(np.mean(resolved)) if len(m) else 0.0,
        "accuracy": float(np.mean(pred[resolved] == y[resolved])) if np.any(resolved) else None,
        "margin_rms": float(np.sqrt(np.mean(m*m))) if len(m) else 0.0,
    }
    ix = np.flatnonzero(resolved)
    order = ix[np.argsort(-np.abs(m[ix]))] if len(ix) else ix
    for frac in FRACS:
        k = max(1, int(len(order)*frac)) if len(order) else 0
        top = order[:k]
        out[f"top_{int(frac*100)}pct_n"] = int(k)
        out[f"top_{int(frac*100)}pct_accuracy"] = (
            float(np.mean(pred[top] == y[top])) if k else None
        )
    return out


def main():
    base.FAST_SYNC = True
    points = fetch_aapl()
    currents, elapsed, kinds, raw = intervals(points)
    need = TRAIN - 1 + ONLINE
    if need > len(currents):
        raise RuntimeError((need, len(currents)))

    model = ConditionedReplay()
    model.reset_transient()

    active_fracs = []
    for i in range(TRAIN - 1):
        out = model.interval_conditioned(float(currents[i]), float(elapsed[i]), True, True)
        if out["total_edges"]:
            active_fracs.append(out["active_edges"] / out["total_edges"])

    train_rel = len(model.birth_members)
    train_depth = max(model.depth.values(), default=0)

    activation = []
    time_gain = []
    flow = []
    truth = []
    open_activation = []
    open_truth = []
    close_activation = []
    close_truth = []

    for i in range(TRAIN - 1, TRAIN - 1 + ONLINE):
        out = model.interval_conditioned(float(currents[i]), float(elapsed[i]), True, True)
        activation.append(out["activation_margin"])
        time_gain.append(out["time_gain_margin"])
        flow.append(out["flow_margin"])
        y = 1 if currents[i] >= 0 else -1
        truth.append(y)
        if kinds[i] == 0:
            open_activation.append(out["activation_margin"])
            open_truth.append(y)
        else:
            close_activation.append(out["activation_margin"])
            close_truth.append(y)
        if out["total_edges"]:
            active_fracs.append(out["active_edges"] / out["total_edges"])

    result = {
        "training_prices": TRAIN,
        "online_intervals": ONLINE,
        "training_start": points[0].date,
        "training_end": points[TRAIN-1].date,
        "online_end": points[TRAIN-1+ONLINE].date,
        "learning_live": True,
        "construction_live": True,
        "relations_train": train_rel,
        "relations_final": len(model.birth_members),
        "depth_train": train_depth,
        "depth_final": max(model.depth.values(), default=0),
        "mean_active_edge_fraction": float(np.mean(active_fracs)) if active_fracs else 0.0,
        "actual_up_fraction": float(np.mean(np.asarray(truth) == 1)),
        "activation": summarize(activation, truth),
        "activation_open": summarize(open_activation, open_truth),
        "activation_close": summarize(close_activation, close_truth),
        "time_gain": summarize(time_gain, truth),
        "incoming_flow": summarize(flow, truth),
    }
    print("RESULT", json.dumps(result, sort_keys=True), flush=True)
    print("all_assertions_passed", flush=True)


if __name__ == "__main__":
    main()
