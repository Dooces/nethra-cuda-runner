#!/usr/bin/env python3
"""Audit whether learned Nethra can continue through their own live field after external input stops.

Training uses only ordinary NethraField.push()/step() on a repeated A->B->C->D stream.
No relations are hand-created.

Evaluation reconstructs a short observed prefix, then external current is set to zero. From that
point onward only the Nethra field equation runs. We measure:
  * whether learned relation Nethra of increasing recursive origin depth continue gaining activation;
  * whether the next source Nethra in the learned sequence gains more activation than alternatives;
  * whether deeper learned Nethra peak later, which would be evidence of actual recursive field
    propagation rather than an external loop replaying the sequence.

The evaluation does not convert activations into new symbolic input and does not inject a decoder,
selector, transition table, or threshold.
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
    """Creation-causal recursive depth: only routes into already-existing members can raise depth."""
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

    # Clear physical state, retain learned topology/evidence.
    for n in g.nethra:
        n.activation = 0.0
        n.external = 0.0

    # Clear only transient chronological boundary so every evaluation begins identically.
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

    # Present enough real source history to refind the learned context, ending at cue_index.
    start = (cue_index - PREFIX + 1) % SYMBOLS
    seq = [(start + k) % SYMBOLS for k in range(PREFIX)]
    assert seq[-1] == cue_index
    for idx in seq:
        leaves[idx].push(1.0)
        g.step(TRAIN_DT)

    # Evaluation begins here. No further external source is ever applied.
    for n in g.nethra:
        n.external = 0.0
    return g, leaves


def evaluate_one(f, depth, cue_index):
    g, leaves = clone_and_prefix(f, cue_index)
    gdepth = {g.nethra[i]: depth[f.nethra[i]] for i in range(len(f.nethra))}

    base = {n: n.activation for n in g.nethra}
    peak_gain = {n: 0.0 for n in g.nethra}
    peak_time = {n: 0.0 for n in g.nethra}

    initial_derivative = g.derivative()

    for step in range(1, EVAL_STEPS + 1):
        raw_step(g, EVAL_DT)
        t = step * EVAL_DT
        for n in g.nethra:
            gain = n.activation - base[n]
            if gain > peak_gain[n]:
                peak_gain[n] = gain
                peak_time[n] = t

    expected = (cue_index + 1) % SYMBOLS
    alternatives = [i for i in range(SYMBOLS) if i != cue_index]
    ranked = sorted(alternatives, key=lambda i: peak_gain[leaves[i]], reverse=True)
    next_rank = ranked.index(expected) + 1

    by_depth = defaultdict(list)
    for n in g.nethra[SYMBOLS:]:
        if n.routes:
            by_depth[gdepth[n]].append(n)

    depth_rows = []
    for d in sorted(by_depth):
        nodes = by_depth[d]
        best = max(nodes, key=lambda n: peak_gain[n])
        depth_rows.append((d, peak_gain[best], peak_time[best]))

    print(
        "cue", cue_index,
        "expected", expected,
        "next_rank", next_rank,
        "expected_gain", f"{peak_gain[leaves[expected]]:.9e}",
        "expected_dadt0", f"{initial_derivative[leaves[expected]]:.9e}",
        "leaf_gains", [f"{peak_gain[n]:.6e}" for n in leaves],
    )
    print(
        "depth_peaks",
        " ".join(f"{d}:{gain:.3e}@{t:.2f}" for d, gain, t in depth_rows[:32]),
    )
    return next_rank, depth_rows


def main():
    f, leaves = train()
    depth = origin_depths(f)
    relations = [n for n in f.nethra if n.routes]
    max_depth = max((depth[n] for n in relations), default=0)

    print("training_intervals", TRAIN_CYCLES * SYMBOLS)
    print("nethra", len(f.nethra))
    print("relations", len(relations))
    print("max_origin_depth", max_depth)
    print("field_activation_nonzero", sum(abs(n.activation) > 1e-15 for n in f.nethra))

    ranks = []
    deepest_positive = 0
    for cue in range(SYMBOLS):
        rank, rows = evaluate_one(f, depth, cue)
        ranks.append(rank)
        for d, gain, _t in rows:
            if gain > 1e-12:
                deepest_positive = max(deepest_positive, d)

    print("next_ranks", ranks)
    print("rank1_count", sum(r == 1 for r in ranks), "of", len(ranks))
    print("deepest_endogenously_rising_depth", deepest_positive)

    # This assertion is purely ontological: recursive learned Nethra must genuinely receive field
    # activation with no external current after the observed prefix. Sequence prediction is reported
    # as a result rather than forced as an assertion.
    assert max_depth >= 2, max_depth
    assert deepest_positive >= 2, deepest_positive
    print("recursive_self_feed_confirmed", True)


if __name__ == "__main__":
    main()
