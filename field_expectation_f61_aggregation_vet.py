#!/usr/bin/env python3
"""Check whether F61 supplies the aggregate expectation operator for established Nethra.

Uses the paired field-derived residual from field_expectation_blocking_vet:
    context_end = field(context)
    expected_end = field_continue(context_end, no new source)
    manifestation = field(context_end, actual new source) - expected_end
    residual = manifestation - expected_end

This probe isolates aggregation:
- two predictor relations converge on B;
- their ordinary conductive topology is identical;
- F61 pair-history independence is set either 0 (duplicate histories) or 1 (independent histories);
- compare source-free expected B and overexpectation residual.

No learning law is installed.
"""

from nethra import NethraField

T=.6
STEPS=120


def integrate(field,start,currents):
    for n in field.nethra:
        n.activation=float(start[n])
        n.external=0.0
    for n,j in currents.items(): n.external=float(j)
    dt=T/STEPS
    for _ in range(STEPS):
        a0={n:n.activation for n in field.nethra}
        k1=field._derivative_at(a0)
        a1={n:a0[n]+.5*dt*k1[n] for n in field.nethra}; k2=field._derivative_at(a1)
        a2={n:a0[n]+.5*dt*k2[n] for n in field.nethra}; k3=field._derivative_at(a2)
        a3={n:a0[n]+dt*k3[n] for n in field.nethra}; k4=field._derivative_at(a3)
        for n in field.nethra:
            n.activation=a0[n]+dt*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6
    out={n:n.activation for n in field.nethra}
    for n in field.nethra:n.external=0.0
    return out


def build(independence):
    f=NethraField(leakage=.6,convergence_gain=1.0)
    a=f.new(); ra=f.new(); c=f.new(); rc=f.new(); b=f.new()
    g=1.5
    f._edges=lambda:((a,ra,g),(ra,b,g),(c,rc,g),(rc,b,g))
    if independence==1:
        f.pair_stats[frozenset((ra,rc))]=(0.0,1.0,1.0,20)
    elif independence==0:
        f.pair_stats[frozenset((ra,rc))]=(1.0,1.0,1.0,20)
    else:
        raise ValueError
    return f,a,ra,c,rc,b


def measure(independence):
    f,a,ra,c,rc,b=build(independence)
    z={n:0.0 for n in f.nethra}
    ctx=integrate(f,z,{a:1.0,c:1.0})
    expected=integrate(f,ctx,{})
    actual=integrate(f,ctx,{b:.2})
    manifest=actual[b]-expected[b]
    residual=manifest-expected[b]
    return expected[b],manifest,residual


def single():
    f=NethraField(leakage=.6,convergence_gain=1.0)
    a=f.new(); r=f.new(); b=f.new(); g=1.5
    f._edges=lambda:((a,r,g),(r,b,g))
    z={n:0.0 for n in f.nethra}
    ctx=integrate(f,z,{a:1.0})
    expected=integrate(f,ctx,{})
    actual=integrate(f,ctx,{b:.2})
    return expected[b],actual[b]-expected[b],(actual[b]-expected[b])-expected[b]


def main():
    s=single()
    dup=measure(0)
    ind=measure(1)
    print("single",s)
    print("duplicate_history_pair",dup)
    print("independent_history_pair",ind)

    assert dup[0] > s[0]             # ordinary conductive combination still sums
    assert ind[0] > dup[0]           # F61 adds only independent convergence bonus
    assert ind[2] < dup[2]           # more aggregate expectation -> more negative shared residual
    assert dup[2] < 0.0 and ind[2] < 0.0
    print("F61_independence_modulates_aggregate_expectation")
    print("all_assertions_passed")


if __name__=="__main__":
    main()
