#!/usr/bin/env python3
"""Check whether permissive creation of weak zero-evidence relations is only a compute cost.

A useful strong A-Y relation is held fixed. Increasing numbers of unrelated zero-evidence
relations are attached to the same receiver Y. Under current Nethra conductance semantics,
zero evidence still means g_min conductance, so every false persistent relation is physically
present.

The probe compares current g_min against g_min=0 control. No learning occurs.
"""

from nethra import NethraField

STEPS=60
DT=.01


def set_evidence(r,e):
    for c in r.routes.values():
        c.clear()
        c[frozenset()]=float(e)


def integrate(f,currents):
    for n in f.nethra:
        n.activation=0.0
        n.external=0.0
    for n,j in currents.items():
        n.external=float(j)
    for _ in range(STEPS):
        a0={n:n.activation for n in f.nethra}
        k1=f._derivative_at(a0)
        a1={n:a0[n]+.5*DT*k1[n] for n in f.nethra}; k2=f._derivative_at(a1)
        a2={n:a0[n]+.5*DT*k2[n] for n in f.nethra}; k3=f._derivative_at(a2)
        a3={n:a0[n]+DT*k3[n] for n in f.nethra}; k4=f._derivative_at(a3)
        for n in f.nethra:
            n.activation=a0[n]+DT*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.0
    for n in f.nethra:
        n.external=0.0


def run(junk_count,g_min):
    f=NethraField(leakage=.6,convergence_gain=0.0,g_min=g_min)
    a=f.new(); y=f.new()
    true=f.new()
    f._route(true,(a,y),frozenset(),1)
    set_evidence(true,50.0)

    for _ in range(junk_count):
        x=f.new()
        r=f.new()
        f._route(r,(x,y),frozenset(),0)
        set_evidence(r,0.0)

    integrate(f,{a:1.0})
    return {
        "junk":junk_count,
        "g_min":g_min,
        "y_activation":y.activation,
        "true_activation":true.activation,
        "nethra":len(f.nethra),
        "edges":len(f._edges()),
    }


def main():
    counts=(0,1,5,10,25,50,100,250)
    default=[run(n,.2) for n in counts]
    zero=[run(n,0.0) for n in counts]

    base=default[0]["y_activation"]
    # Under current g_min, enough false persistent relations must measurably load Y.
    assert default[-1]["y_activation"] < base*.5

    base0=zero[0]["y_activation"]
    # With no baseline conductance, zero-evidence junk is computational structure but has no
    # field effect.
    assert max(abs(r["y_activation"]-base0) for r in zero) < 1e-12

    print("gmin_default",default)
    print("gmin_zero",zero)
    print("default_retained_fraction",default[-1]["y_activation"]/base)
    print("zero_retained_fraction",zero[-1]["y_activation"]/base0)
    print("all_assertions_passed")


if __name__=="__main__":
    main()
