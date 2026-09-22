#!/usr/bin/env python3
"""Ordinary-Nethra local perspective probe.

For any ordinary relation Nethra R, its current field state defines signed current from R to each
neighbor m:

    c_Rm(k) = g_Rm * (a_R(k) - a_m(k))

No member is assigned a semantic role.  Positive c means R is currently tending to drive that
neighbor; negative c means that neighbor is currently supplying R.

On the next completed interval, exact independent external source current J_m(k+1) is already
available from the frozen interval boundary.  The audit-only local prospective receipt is:

    q_R = sum_m c_Rm(k) * J_m(k+1)

This is a dot product between one Nethra's local field-current pattern and the next independent
source pattern over its actual neighbors.  It is not a learning update and creates no direction.
"""

from nethra import NethraField


def rk4(field,currents,steps=30,dt=.03):
    for n in field.nethra: n.external=0.0
    for n,j in currents.items(): n.external=float(j)
    for _ in range(steps):
        a0={n:n.activation for n in field.nethra}
        k1=field._derivative_at(a0)
        a1={n:a0[n]+.5*dt*k1[n] for n in field.nethra}; k2=field._derivative_at(a1)
        a2={n:a0[n]+.5*dt*k2[n] for n in field.nethra}; k3=field._derivative_at(a2)
        a3={n:a0[n]+dt*k3[n] for n in field.nethra}; k4=field._derivative_at(a3)
        for n in field.nethra:
            n.activation=a0[n]+dt*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.0
    for n in field.nethra: n.external=0.0


def set_strength(r,e):
    for c in r.routes.values():
        c.clear(); c[frozenset()]=float(e)


def neighbor_currents(field,r):
    out={}
    for a,b,g in field._edges():
        if a is r:
            out[b]=out.get(b,0.0)+g*(r.activation-b.activation)
        elif b is r:
            out[a]=out.get(a,0.0)+g*(r.activation-a.activation)
    return out


def receipt(field,r,next_source):
    c=neighbor_currents(field,r)
    return sum(c.get(n,0.0)*float(j) for n,j in next_source.items()),c


def make_pair(evidence=1.0):
    f=NethraField(leakage=.6,convergence_gain=0.0)
    a=f.new(); y=f.new(); z=f.new(); r=f.new()
    f._route(r,(a,y),frozenset(),1); set_strength(r,evidence)
    return f,a,y,z,r


def test_temporal_orientation_emerges_from_state():
    f,a,y,z,r=make_pair()
    rk4(f,{a:1.0})
    ay,c1=receipt(f,r,{y:.02})
    aa,_=receipt(f,r,{a:.02})
    az,_=receipt(f,r,{z:.02})

    f2,a2,y2,z2,r2=make_pair()
    rk4(f2,{y2:1.0})
    ya,c2=receipt(f2,r2,{a2:.02})

    assert ay>0.0
    assert ya>0.0
    assert aa<0.0
    assert az==0.0
    return ay,aa,az,ya,c1,c2


def test_simultaneous_from_zero_has_no_prior_perspective():
    f,a,y,z,r=make_pair()
    q,c=receipt(f,r,{a:1.0,y:1.0})
    assert q==0.0
    assert all(v==0.0 for v in c.values())
    return q,c


def test_multi_member_relation_grades_incomplete_vs_complete_support():
    def run(inputs):
        f=NethraField(leakage=.6,convergence_gain=0.0)
        a=f.new(); b=f.new(); y=f.new(); r=f.new()
        f._route(r,(a,b,y),frozenset(),1); set_strength(r,1.0)
        currents={}
        if "a" in inputs: currents[a]=1.0
        if "b" in inputs: currents[b]=1.0
        rk4(f,currents)
        q,c=receipt(f,r,{y:.02})
        return q,r.activation,y.activation,c[y]

    a_only=run({"a"})
    b_only=run({"b"})
    both=run({"a","b"})
    assert both[0] > a_only[0] > 0.0
    assert both[0] > b_only[0] > 0.0
    return a_only,b_only,both


def test_competing_relations_share_target_without_selector():
    f=NethraField(leakage=.6,convergence_gain=0.0)
    a=f.new(); b=f.new(); y=f.new()
    r1=f.new(); r2=f.new()
    f._route(r1,(a,y),frozenset(),1); set_strength(r1,1.0)
    f._route(r2,(b,y),frozenset(),1); set_strength(r2,1.0)
    rk4(f,{a:1.0,b:1.0})
    q1,c1=receipt(f,r1,{y:.02})
    q2,c2=receipt(f,r2,{y:.02})

    solo,sa,sy,sz,sr=make_pair()
    rk4(solo,{sa:1.0})
    qs,cs=receipt(solo,sr,{sy:.02})

    assert q1>0.0 and q2>0.0
    assert abs(q1-q2)<1e-12
    assert q1<qs
    return qs,q1,q2,q1+q2


def test_existing_strong_relation_loads_weaker_perspective():
    rows=[]
    for strong in (1.0,5.0,20.0,100.0,1000.0):
        f=NethraField(leakage=.6,convergence_gain=0.0)
        a=f.new(); c=f.new(); y=f.new()
        weak=f.new(); base=f.new()
        f._route(weak,(a,y),frozenset(),1); set_strength(weak,1.0)
        f._route(base,(c,y),frozenset(),1); set_strength(base,strong)
        rk4(f,{a:1.0,c:1.0})
        qw,cw=receipt(f,weak,{y:.02})
        qb,cb=receipt(f,base,{y:.02})
        rows.append((strong,qw,qb,y.activation))
    assert all(rows[i][1]>=rows[i+1][1]-1e-12 for i in range(len(rows)-1))
    return rows


def test_signed_source_is_symmetric():
    f,a,y,z,r=make_pair()
    rk4(f,{a:1.0})
    pos,_=receipt(f,r,{y:.02})

    f2,a2,y2,z2,r2=make_pair()
    rk4(f2,{a2:-1.0})
    neg,_=receipt(f2,r2,{y2:-.02})

    assert abs(pos-neg)<1e-12
    return pos,neg


def main():
    print("state_emergent_orientation",test_temporal_orientation_emerges_from_state())
    print("simultaneous_zero",test_simultaneous_from_zero_has_no_prior_perspective())
    print("multi_member_support",test_multi_member_relation_grades_incomplete_vs_complete_support())
    print("competing_relations",test_competing_relations_share_target_without_selector())
    print("strong_relation_loading",test_existing_strong_relation_loads_weaker_perspective())
    print("signed_source_symmetry",test_signed_source_is_symmetric())
    print("all_assertions_passed")


if __name__=="__main__":
    main()
