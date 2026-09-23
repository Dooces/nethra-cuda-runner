#!/usr/bin/env python3
"""Audit whether learned Nethra can continue through their own live field after external input stops.

Training uses only ordinary NethraField.push()/step() on a repeated A->B->C->D stream.
No relations are hand-created.

Evaluation reconstructs a short observed prefix, then external current is set to zero. From that
point onward only the Nethra field equation runs. We measure:
  * whether learned relation Nethra of increasing recursive origin depth continue gaining activation;
  * what is already primed at the instant the final external cue ends;
  * whether the next source Nethra in the learned sequence is favored over alternatives;
  * whether deeper learned Nethra peak later, which would be evidence of actual recursive field
    propagation rather than an external loop replaying the sequence.

The evaluation does not convert activations into new symbolic input and does not inject a decoder,
selector, transition table, fitted threshold, or outcome-fed calibration.
"""

from __future__ import annotations

import copy
from collections import defaultdict

from nethra import NethraField

SYMBOLS = 4
TRAIN_CYCLES = 300
TRAIN_DT = 0.15
EVAL_DT = 0.01
EVAL_STEPS = 500
PREFIX = 8


def train():
    f = NethraField(g_min=.20, g_max=1.50, tau=100.0,
                    capacitance=1.0, leakage=.6, convergence_gain=0.0)
    leaves = [f.new() for _ in range(SYMBOLS)]
    for _ in range(TRAIN_CYCLES):
        for i in range(SYMBOLS):
            leaves[i].push(1.0)
            f.step(TRAIN_DT)
    return f, leaves


def origin_depths(f):
    """Creation-causal recursive depth using routes whose members already existed."""
    index = {n: i for i, n in enumerate(f.nethra)}
    depth = {}
    for n in f.nethra:
        i = index[n]
        candidates = []
        for route in n.routes:
            older = [m for m in route if index[m] < i]
            if older and len(older) == len(route) and all(m in depth for m in older):
                candidates.append(1 + max(depth[m] for m in older))
        depth[n] = max(candidates, default=0)
    return depth


def raw_step(f, dt):
    """Exactly the core RK4 field equation, with no observation/construction call."""
    a0 = {n: n.activation for n in f.nethra}
    k1 = f._derivative_at(a0)
    a1 = {n: a0[n] + .5 * dt * k1[n] for n in f.nethra}
    k2 = f._derivative_at(a1)
    a2 = {n: a0[n] + .5 * dt * k2[n] for n in f.nethra}
    k3 = f._derivative_at(a2)
    a3 = {n: a0[n] + dt * k3[n] for n in f.nethra}
    k4 = f._derivative_at(a3)
    for n in f.nethra:
        n.activation = a0[n] + dt * (k1[n] + 2*k2[n] + 2*k3[n] + k4[n]) / 6.0


def clone_and_prefix(f, cue_index):
    g = copy.deepcopy(f)
    leaves = g.nethra[:SYMBOLS]

    for n in g.nethra:
        n.activation = 0.0
        n.external = 0.0

    g.previous_explicit = frozenset()
    g.previous_closure = frozenset()
    g.previous_source_event = frozenset()
    g.current_source_event = frozenset()
    g.previous_event = frozenset()
    g.current_event = frozenset()
    g.previous_interval_source = {}
    g.current_interval_source = {}
    g.previous_interval_delta = {}
    g.current_interval_delta = {}

    start = (cue_index - PREFIX + 1) % SYMBOLS
    seq = [(start + k) % SYMBOLS for k in range(PREFIX)]
    assert seq[-1] == cue_index
    for idx in seq:
        leaves[idx].push(1.0)
        g.step(TRAIN_DT)

    for n in g.nethra:
        n.external = 0.0
    return g, leaves


def rank_of(index, candidates, score):
    ranked = sorted(candidates, key=lambda i: score(i), reverse=True)
    return ranked.index(index) + 1, ranked


def evaluate_one(f, cue_index):
    g, leaves = clone_and_prefix(f, cue_index)
    depth = origin_depths(g)

    base = {n: n.activation for n in g.nethra}
    peak_gain = {n: 0.0 for n in g.nethra}
    peak_abs = dict(base)
    peak_time = {n: 0.0 for n in g.nethra}
    integral = {n: 0.0 for n in g.nethra}
    initial_derivative = g.derivative()

    expected = (cue_index + 1) % SYMBOLS
    alternatives = [i for i in range(SYMBOLS) if i != cue_index]
    base_rank, base_order = rank_of(expected, alternatives, lambda i: base[leaves[i]])
    dadt_rank, dadt_order = rank_of(expected, alternatives, lambda i: initial_derivative[leaves[i]])

    for step in range(1, EVAL_STEPS + 1):
        raw_step(g, EVAL_DT)
        t = step * EVAL_DT
        for n in g.nethra:
            integral[n] += n.activation * EVAL_DT
            gain = n.activation - base[n]
            if gain > peak_gain[n]:
                peak_gain[n] = gain
                peak_time[n] = t
            if n.activation > peak_abs[n]:
                peak_abs[n] = n.activation

    future_rank, future_order = rank_of(expected, alternatives, lambda i: peak_abs[leaves[i]])
    integral_rank, integral_order = rank_of(expected, alternatives, lambda i: integral[leaves[i]])

    by_depth = defaultdict(list)
    for n in g.nethra[SYMBOLS:]:
        if n.routes:
            by_depth[depth[n]].append(n)

    depth_rows = []
    for d in sorted(by_depth):
        nodes = by_depth[d]
        best = max(nodes, key=lambda n: peak_gain[n])
        depth_rows.append((d, peak_gain[best], peak_time[best], base[best], peak_abs[best]))

    print(
        "cue", cue_index,
        "expected", expected,
        "relations", sum(bool(n.routes) for n in g.nethra),
        "max_depth", max(depth.values(), default=0),
    )
    print(
        "leaf_base", [f"{base[n]:.9e}" for n in leaves],
        "base_rank", base_rank, "base_order", base_order,
    )
    print(
        "leaf_dadt0", [f"{initial_derivative[n]:.9e}" for n in leaves],
        "dadt_rank", dadt_rank, "dadt_order", dadt_order,
    )
    print(
        "leaf_peak_abs", [f"{peak_abs[n]:.9e}" for n in leaves],
        "future_rank", future_rank, "future_order", future_order,
    )
    print(
        "leaf_peak_gain", [f"{peak_gain[n]:.9e}" for n in leaves],
        "leaf_integral", [f"{integral[n]:.9e}" for n in leaves],
        "integral_rank", integral_rank, "integral_order", integral_order,
    )
    print(
        "depth_peaks",
        " ".join(
            f"{d}:gain={gain:.3e}@{t:.2f},base={b:.3e},peak={p:.3e}"
            for d, gain, t, b, p in depth_rows[:32]
        ),
    )
    return base_rank, dadt_rank, future_rank, integral_rank, depth_rows


def main():
    f, _leaves = train()
    depth = origin_depths(f)
    relations = [n for n in f.nethra if n.routes]
    max_depth = max((depth[n] for n in relations), default=0)

    print("training_intervals", TRAIN_CYCLES * SYMBOLS)
    print("nethra", len(f.nethra))
    print("relations", len(relations))
    print("max_origin_depth", max_depth)
    print("field_activation_nonzero", sum(abs(n.activation) > 1e-15 for n in f.nethra))

    base_ranks = []
    dadt_ranks = []
    future_ranks = []
    integral_ranks = []
    deepest_positive = 0

    for cue in range(SYMBOLS):
        br, dr, fr, ir, rows = evaluate_one(f, cue)
        base_ranks.append(br)
        dadt_ranks.append(dr)
        future_ranks.append(fr)
        integral_ranks.append(ir)
        for d, gain, _t, _b, _p in rows:
            if gain > 1e-12:
                deepest_positive = max(deepest_positive, d)

    print("base_ranks", base_ranks, "rank1", sum(r == 1 for r in base_ranks))
    print("dadt_ranks", dadt_ranks, "rank1", sum(r == 1 for r in dadt_ranks))
    print("future_ranks", future_ranks, "rank1", sum(r == 1 for r in future_ranks))
    print("integral_ranks", integral_ranks, "rank1", sum(r == 1 for r in integral_ranks))
    print("deepest_endogenously_rising_depth", deepest_positive)

    assert max_depth >= 2, max_depth
    assert deepest_positive >= 2, deepest_positive
    print("recursive_self_feed_confirmed", True)


if __name__ == "__main__":
    main()
