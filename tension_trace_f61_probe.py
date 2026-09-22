#!/usr/bin/env python3
"""Feed field-derived Nethra tension into the existing F61 residual-trace machinery.

This is a shadow compatibility test, not a promoted residual definition.

Each ordinary Nethra r receives a scalar local prospective tension:
    T_r = sum_m positive_flow(r->m, interval k) * epsilon_m(interval k+1)

where receiver epsilon uses exact next source charge minus total positive incoming field charge.

The probe asks whether using T_r as the already-existing F61 "own predictive residual" produces the
intended historical independence behavior without source IDs:
- duplicate predictors with identical field experience -> W ~= 0
- predictors with nonoverlapping/independent experience -> W ~= 1
- when later simultaneously active, only the historically independent pair gets convergence bonus.
"""

import random
from nethra import NethraField


DURATION=.6
STEPS=30
DT=DURATION/STEPS
Y_CURRENT=.02


def set_strength(r,e):
    for c in r.routes.values():
        c.clear(); c[frozenset()]=float(e)


def integrate(field,currents):
    for n in field.nethra:
        n.activation=0.0; n.external=0.0
    for n,j in currents.items(): n.external=float(j)
    es=field._edges()
    q=[0.0]*len(es)

    def flows(state):
        return [g*(state[a]-state[b]) for a,b,g in es]

    for _ in range(STEPS):
        a0={n:n.activation for n in field.nethra}
        k1=field._derivative_at(a0)
        a1={n:a0[n]+.5*DT*k1[n] for n in field.nethra}; k2=field._derivative_at(a1)
        a2={n:a0[n]+.5*DT*k2[n] for n in field.nethra}; k3=field._derivative_at(a2)
        a3={n:a0[n]+DT*k3[n] for n in field.nethra}; k4=field._derivative_at(a3)
        f0,f1,f2,f3=flows(a0),flows(a1),flows(a2),flows(a3)
        for i in range(len(es)):
            q[i]+=DT*(f0[i]+2*f1[i]+2*f2[i]+f3[i])/6.0
        for n in field.nethra:
            n.activation=a0[n]+DT*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.0

    for n in field.nethra: n.external=0.0
    return es,q


def positive_flows(field,es,q):
    out={n:{} for n in field.nethra}
    inc={n:0.0 for n in field.nethra}
    for (a,b,g),charge in zip(es,q):
        if charge>0:
            out[a][b]=out[a].get(b,0.0)+charge
            inc[b]+=charge
        elif charge<0:
            z=-charge
            out[b][a]=out[b].get(a,0.0)+z
            inc[a]+=z
    return out,inc


def tensions(field,out,inc,next_source):
    source={n:0.0 for n in field.nethra}
    for n,j in next_source.items():
        source[n]=DURATION*float(j)
    eps={n:source[n]-inc[n] for n in field.nethra}
    return {
        r:sum(q*eps[m] for m,q in out[r].items())
        for r in field.nethra
    }


def build():
    f=NethraField(leakage=.6,convergence_gain=1.0,trace_decay=.9)
    a=f.new(); b=f.new(); y=f.new()
    r1=f.new(); r2=f.new()
    f._route(r1,(a,y),frozenset(),1); set_strength(r1,1.0)
    f._route(r2,(b,y),frozenset(),1); set_strength(r2,1.0)
    return f,a,b,y,r1,r2


def train_history(mode,seed=8801,n=500):
    rng=random.Random(seed)
    f,a,b,y,r1,r2=build()
    samples=[]
    for _ in range(n):
        if mode=="duplicate":
            cur={a:1.0,b:1.0}
        elif mode=="independent":
            cur={a:1.0} if rng.random()<.5 else {b:1.0}
        else:
            raise ValueError(mode)

        es,q=integrate(f,cur)
        out,inc=positive_flows(f,es,q)
        next_source={y:Y_CURRENT} if rng.random()<.5 else {}
        t=tensions(f,out,inc,next_source)
        f.update_residuals(t)
        samples.append((t[r1],t[r2]))

    w=f._independence(r1,r2)
    return f,a,b,y,r1,r2,w,samples


def convergence_component(field,state,receiver,suppliers):
    # Compare actual derivative with same field state after zeroing pair evidence, so the
    # difference at receiver is exactly the F61 convergence redistribution contribution.
    d_on=field._derivative_at(state)[receiver]
    saved=dict(field.pair_stats)
    field.pair_stats.clear()
    d_off=field._derivative_at(state)[receiver]
    field.pair_stats.update(saved)
    return d_on-d_off


def test_duplicate_tension_histories_are_not_independent():
    f,a,b,y,r1,r2,w,s=train_history("duplicate")
    max_diff=max(abs(x-yv) for x,yv in s)
    assert max_diff < 1e-15
    assert w < 1e-12
    return w,max_diff,s[-5:]


def test_nonoverlapping_tension_histories_are_independent():
    f,a,b,y,r1,r2,w,s=train_history("independent")
    products=sum(abs(x*yv) for x,yv in s)
    assert products < 1e-18
    assert w > .95
    return w,products,s[-5:]


def test_same_present_combination_gets_different_convergence_from_history():
    dup=train_history("duplicate")
    ind=train_history("independent")

    rows=[]
    for name,data in (("duplicate",dup),("independent",ind)):
        f,a,b,y,r1,r2,w,s=data
        # Produce one common present state with both contexts active.
        integrate(f,{a:1.0,b:1.0})
        state={n:n.activation for n in f.nethra}
        conv=convergence_component(f,state,y,(r1,r2))
        rows.append((name,w,conv,y.activation))
    assert rows[0][2] < 1e-12
    assert rows[1][2] > 0.0
    return rows


def main():
    print("duplicate_history",test_duplicate_tension_histories_are_not_independent())
    print("independent_history",test_nonoverlapping_tension_histories_are_independent())
    print("history_controls_convergence",test_same_present_combination_gets_different_convergence_from_history())
    print("all_assertions_passed")


if __name__=="__main__":
    main()
