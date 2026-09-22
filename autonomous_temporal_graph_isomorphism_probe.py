#!/usr/bin/env python3
"""Test whether the old crystallized temporal structure expresses learned delay autonomously.

This reproduces the connected F61 graph created by the archived crystallization contract:

    source_context = {A,B}
    temporal_tau   = {source_context, delta_t[tau]}
    perspective    = {temporal_tau, Y}

The archived contract gives the learned high perspective conductance only when temporal_tau is
present in current support. At autonomous probe time we remove the temporal provenance transducer,
so only A and B receive external current and that applicability condition is absent.

Important: delta_t[tau] is an ordinary Nethra. The integer tau changes only which lag-coordinate
node is connected; tau is not a capacitance, conductance, leakage, delay, or differential parameter.

Therefore lag-1/3/5/8 connected components should be dynamically isomorphic when only X={A,B} is
sourced. This script verifies that under the current frozen F61 equation and also verifies the old
manual temporal-handle control can drive Y when temporal_tau is directly sourced.

No learning, prediction equation, temporal transducer, or source after X is used in the autonomous
probe.
"""

from nethra import NethraField

DT=.01
PULSE=.30
FREE_STEP=.25
FREE_INTERVALS=32
EVIDENCE=1600.0


def make(lag:int, learned:bool=True):
    f=NethraField(leakage=1.0,capacitance=1.0,convergence_gain=0.0)
    # roots: A,B,N1,N2,Y,Z
    A=f.new();B=f.new();N1=f.new();N2=f.new();Y=f.new();Z=f.new()
    lag_nodes=[f.new() for _ in range(lag)]
    lag_nid=lag_nodes[-1]
    source_context=f.new()
    temporal=f.new()
    perspective=f.new()

    g0=f.conductance(0)
    gh=f.conductance(EVIDENCE)

    # Old crystallization creates these ordinary structural incidences.
    base_edges=[
        (source_context,A,g0),
        (source_context,B,g0),
        (temporal,source_context,g0),
        (temporal,lag_nid,g0),
    ]

    # Learned temporal evidence strengthened the perspective relation, but its high applicability
    # was conditional on current support containing temporal itself. With X-only and no temporal
    # transducer, that condition is absent => g_min. Manual temporal control below activates it.
    auto_perspective_g=g0
    manual_perspective_g=gh if learned else g0

    return {
        "field":f,"A":A,"B":B,"Y":Y,"Z":Z,"lag":lag,
        "lag_nid":lag_nid,"source_context":source_context,"temporal":temporal,
        "perspective":perspective,"base_edges":tuple(base_edges),
        "auto_g":auto_perspective_g,"manual_g":manual_perspective_g,
    }


def set_edges(obj,mode):
    g=obj["auto_g"] if mode=="auto" else obj["manual_g"]
    edges=obj["base_edges"]+(
        (obj["perspective"],obj["temporal"],g),
        (obj["perspective"],obj["Y"],g),
    )
    obj["field"]._edges=lambda:edges


def rk4(field,seconds):
    steps=round(seconds/DT)
    for _ in range(steps):
        a0={n:n.activation for n in field.nethra}
        k1=field._derivative_at(a0)
        a1={n:a0[n]+.5*DT*k1[n] for n in field.nethra};k2=field._derivative_at(a1)
        a2={n:a0[n]+.5*DT*k2[n] for n in field.nethra};k3=field._derivative_at(a2)
        a3={n:a0[n]+DT*k3[n] for n in field.nethra};k4=field._derivative_at(a3)
        for n in field.nethra:
            n.activation=a0[n]+DT*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6


def zero_sources(f):
    for n in f.nethra:n.external=0.0


def autonomous_curve(lag,learned=True):
    o=make(lag,learned)
    f=o["field"];set_edges(o,"auto")
    o["A"].external=1.0;o["B"].external=1.0
    rk4(f,PULSE)
    zero_sources(f)

    rows=[(0.0,o["Y"].activation,o["temporal"].activation,o["source_context"].activation)]
    elapsed=0.0
    for _ in range(FREE_INTERVALS):
        rk4(f,FREE_STEP);elapsed+=FREE_STEP
        rows.append((elapsed,o["Y"].activation,o["temporal"].activation,o["source_context"].activation))
    return rows


def manual_temporal_control(lag,learned=True):
    o=make(lag,learned);f=o["field"];set_edges(o,"manual")
    o["temporal"].external=1.0
    rk4(f,.75)
    zero_sources(f)
    return (o["Y"].activation,o["Z"].activation,o["Y"].activation-o["Z"].activation)


def peak(curve):
    return max(curve,key=lambda r:r[1])[:2]


def maxdiff(a,b,col=1):
    return max(abs(x[col]-y[col]) for x,y in zip(a,b))


def main():
    lags=(1,3,5,8)

    print("g_min",NethraField().conductance(0),"g_learned",NethraField().conductance(EVIDENCE))

    print("=== MANUAL TEMPORAL HANDLE CONTROL ===")
    for lag in lags:
        print("lag",lag,"learned",manual_temporal_control(lag,True),
              "unlearned",manual_temporal_control(lag,False))

    print("=== AUTONOMOUS X ONLY; NO TEMPORAL PROVENANCE ===")
    learned={lag:autonomous_curve(lag,True) for lag in lags}
    unlearned={lag:autonomous_curve(lag,False) for lag in lags}

    for lag in lags:
        c=learned[lag]
        print("lag",lag,"peak",peak(c),"nominal_row",c[min(lag,len(c)-1)])
        print("curve",[(round(t,2),y) for t,y,_,_ in c])

    print("=== ISOMORPHISM / LEARNING CONTROLS ===")
    for lag in (3,5,8):
        print("lag1_vs",lag,"max_Y_diff",maxdiff(learned[1],learned[lag],1))
    for lag in lags:
        print("learned_vs_unlearned",lag,"max_Y_diff",maxdiff(learned[lag],unlearned[lag],1))

    peak_times={lag:peak(curve)[0] for lag,curve in learned.items()}
    max_cross=max(maxdiff(learned[a],learned[b],1)
                  for i,a in enumerate(lags) for b in lags[i+1:])
    max_learning=max(maxdiff(learned[x],unlearned[x],1) for x in lags)

    print("peak_times",peak_times)
    print("max_cross_lag_curve_difference",max_cross)
    print("max_learned_vs_unlearned_autonomous_difference",max_learning)

    # Manual learned perspective must have an effect.
    assert all(manual_temporal_control(l,True)[0] > manual_temporal_control(l,False)[0] for l in lags)
    # With provenance removed, learned lag and evidence must not alter the X-only connected dynamics.
    assert max_cross < 1e-14
    assert max_learning < 1e-14
    print("VERDICT learned_temporal_delay_not_encoded_in_autonomous_field")
    print("all_assertions_passed")


if __name__=="__main__":
    main()
