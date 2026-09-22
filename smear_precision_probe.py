#!/usr/bin/env python3
"""Fixed overlapping-current precision probe.

This is an investigation harness, not a promoted learner.  It tests whether numeric precision
needs one identity per bin at all.

The transducers below are fixed functions.  They do not inspect consequences, learned topology,
prediction quality, or requested precision.
"""

from math import sqrt
import random

from nethra import NethraField


def pair_smear(x):
    x = float(x)
    if not 0.0 <= x <= 1.0:
        raise ValueError("x outside [0,1]")
    return (1.0 - x, x)


def hat3_smear(x):
    x = float(x)
    if not 0.0 <= x <= 1.0:
        raise ValueError("x outside [0,1]")
    if x <= 0.5:
        return (1.0 - 2.0*x, 2.0*x, 0.0)
    return (0.0, 2.0*(1.0-x), 2.0*x - 1.0)


def integrate_fixed_current(field, currents, steps=200, dt=0.05):
    for n in field.nethra:
        n.activation = 0.0
        n.external = 0.0
    for n, value in currents.items():
        n.external = float(value)

    for _ in range(steps):
        a0 = {n: n.activation for n in field.nethra}
        k1 = field._derivative_at(a0)
        a1 = {n: a0[n] + 0.5*dt*k1[n] for n in field.nethra}
        k2 = field._derivative_at(a1)
        a2 = {n: a0[n] + 0.5*dt*k2[n] for n in field.nethra}
        k3 = field._derivative_at(a2)
        a3 = {n: a0[n] + dt*k3[n] for n in field.nethra}
        k4 = field._derivative_at(a3)
        for n in field.nethra:
            n.activation = a0[n] + dt*(k1[n] + 2*k2[n] + 2*k3[n] + k4[n])/6.0


def test_exact_pair_encoding():
    rng = random.Random(81001)
    max_reconstruction_error = 0.0
    max_axis_error = 0.0
    target = (-1.0/sqrt(2.0), 1.0/sqrt(2.0))

    for _ in range(100000):
        x0 = rng.random()
        x1 = rng.random()
        a0, b0 = pair_smear(x0)
        a1, b1 = pair_smear(x1)

        reconstructed = b0 / (a0 + b0)
        max_reconstruction_error = max(
            max_reconstruction_error, abs(reconstructed - x0)
        )

        da, db = a1-a0, b1-b0
        norm = sqrt(da*da + db*db)
        if norm:
            u = (da/norm, db/norm)
            e1 = sqrt((u[0]-target[0])**2 + (u[1]-target[1])**2)
            e2 = sqrt((u[0]+target[0])**2 + (u[1]+target[1])**2)
            max_axis_error = max(max_axis_error, min(e1, e2))

    assert max_reconstruction_error <= 1e-15
    assert max_axis_error <= 1e-12
    return max_reconstruction_error, max_axis_error


def test_learning_boundary_discards_amplitude():
    field = NethraField()
    a = field.new()
    b = field.new()

    a.external, b.external = pair_smear(0.10)
    d_lo = field.derivative()
    a.external, b.external = pair_smear(0.90)
    d_hi = field.derivative()

    # The field equation receives different graded currents.
    assert abs(d_lo[a] - d_hi[a]) > 0.5
    assert abs(d_lo[b] - d_hi[b]) > 0.5

    # But the present one-file learning boundary receives only nonzero membership.
    explicit_lo = frozenset(n for n in (a,b) if n.external != 0.0)
    a.external, b.external = pair_smear(0.10)
    explicit_hi = frozenset(n for n in (a,b) if n.external != 0.0)
    assert explicit_lo == explicit_hi == frozenset((a,b))

    return d_lo[a], d_lo[b], d_hi[a], d_hi[b]


def test_field_uses_pair_ratio():
    field = NethraField(leakage=0.4)
    a = field.new()
    b = field.new()
    y0 = field.new()
    y1 = field.new()
    r0 = field.new()
    r1 = field.new()

    # Probe topology only: two ordinary relation Nethra couple opposite smear components
    # to opposite outputs.  This does not claim the current learner can discover it yet.
    field._route(r0, (a, y0), frozenset(), 80)
    field._route(r1, (b, y1), frozenset(), 80)

    margins = []
    for x in (0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9):
        ja, jb = pair_smear(x)
        integrate_fixed_current(field, {a: ja, b: jb})
        margins.append((x, y1.activation - y0.activation))

    assert margins[0][1] < 0.0
    assert margins[-1][1] > 0.0
    assert abs(margins[3][1]) < 1e-10
    assert all(margins[i][1] < margins[i+1][1] for i in range(len(margins)-1))
    return margins


def test_field_can_form_local_band_from_three_fixed_smear_channels():
    field = NethraField(leakage=0.4)
    left = field.new()
    center = field.new()
    right = field.new()
    y0 = field.new()
    y1 = field.new()
    inside = field.new()
    outside = field.new()

    field._route(inside, (center, y1), frozenset(), 80)
    field._route(outside, (left, right, y0), frozenset(), 80)

    rows = []
    for i in range(101):
        x = i / 100.0
        jl, jc, jr = hat3_smear(x)
        integrate_fixed_current(field, {left: jl, center: jc, right: jr})
        rows.append((x, y1.activation - y0.activation))

    positive = [x for x, margin in rows if margin > 0.0]
    assert positive
    assert min(positive) > 0.0
    assert max(positive) < 1.0
    assert rows[50][1] > rows[0][1]
    assert rows[50][1] > rows[-1][1]
    return min(positive), max(positive), rows[0][1], rows[50][1], rows[-1][1]


def test_local_hat_delta_axes():
    # Inside either half-interval, every movement is one fixed signed local axis.
    rng = random.Random(81002)
    axes = {
        "left": (-1.0/sqrt(2.0), 1.0/sqrt(2.0), 0.0),
        "right": (0.0, -1.0/sqrt(2.0), 1.0/sqrt(2.0)),
    }
    worst = 0.0
    for side, lo, hi in (("left", 0.0, 0.5), ("right", 0.5, 1.0)):
        target = axes[side]
        for _ in range(50000):
            x0 = lo + (hi-lo)*rng.random()
            x1 = lo + (hi-lo)*rng.random()
            j0 = hat3_smear(x0)
            j1 = hat3_smear(x1)
            d = tuple(j1[i]-j0[i] for i in range(3))
            norm = sqrt(sum(v*v for v in d))
            if not norm:
                continue
            u = tuple(v/norm for v in d)
            e1 = sqrt(sum((u[i]-target[i])**2 for i in range(3)))
            e2 = sqrt(sum((u[i]+target[i])**2 for i in range(3)))
            worst = max(worst, min(e1,e2))
    assert worst <= 1e-12
    return worst



def test_existing_f61_convergence_extracts_overlap_from_pair_smear():
    def make(gain):
        field = NethraField(leakage=0.4, convergence_gain=gain)
        a = field.new()
        b = field.new()
        r = field.new()
        field._route(r, (a, b), frozenset(), 80)
        # Probe setup only: under existing F61 math, zero residual correlation means
        # full supplier independence. No new learning rule is introduced here.
        field.pair_stats[frozenset((a, b))] = (0.0, 1.0, 1.0, 1)
        return field, a, b, r

    control = make(0.0)
    active = make(1.0)
    control_rows = []
    active_rows = []
    for x in (0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95):
        ja, jb = pair_smear(x)
        cf, ca, cb, cr = control
        af, aa, ab, ar = active
        integrate_fixed_current(cf, {ca: ja, cb: jb})
        integrate_fixed_current(af, {aa: ja, ab: jb})
        control_rows.append((x, cr.activation))
        active_rows.append((x, ar.activation))

    control_span = max(v for _, v in control_rows) - min(v for _, v in control_rows)
    assert control_span < 1e-10

    active_map = dict(active_rows)
    assert active_map[0.50] > active_map[0.05]
    assert active_map[0.50] > active_map[0.95]
    assert abs(active_map[0.05] - active_map[0.95]) < 1e-10
    return control_rows, active_rows


def main():
    recon, axis = test_exact_pair_encoding()
    boundary = test_learning_boundary_discards_amplitude()
    monotonic = test_field_uses_pair_ratio()
    band = test_field_can_form_local_band_from_three_fixed_smear_channels()
    local_axis = test_local_hat_delta_axes()
    convergence_control, convergence_active = test_existing_f61_convergence_extracts_overlap_from_pair_smear()

    print("pair_reconstruction_max_error", recon)
    print("pair_delta_axis_max_error", axis)
    print("learning_boundary_derivatives", boundary)
    print("pair_field_margins", monotonic)
    print("three_hat_positive_band", band[:2])
    print("three_hat_edge_center_edge_margins", band[2:])
    print("three_hat_local_axis_max_error", local_axis)
    print("f61_overlap_control", convergence_control)
    print("f61_overlap_active", convergence_active)
    print("pair_active_inputs", 2)
    print("hat3_max_active_inputs", 2)
    print("all_assertions_passed")


if __name__ == "__main__":
    main()
