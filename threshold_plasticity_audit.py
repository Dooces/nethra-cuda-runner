#!/usr/bin/env python3
"""Threshold-gated native-plasticity audit.

This is a shadow test over the existing F61/Nethra field. It deliberately does not search for an
optimal threshold.

A fixed scalar threshold theta has only one authority:
    if abs(local_tension) <= theta: do nothing
    otherwise: allow the signed local tension to alter incidence evidence.

For support recruitment, a missing currently-sourced incidence may be added weakly only when the
relation's positive tension exceeds theta. No subset is selected.

The threshold never declares a relation true/false, never compares exact states, and never chooses
a consequence. Lower thresholds are allowed to spend more computation and create/recruit more weak
support. The audit asks whether there is a broad region in which learned behavior is stable.

All tests use the exact RK4 field and integrated incidence current.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
import random
import time

from nethra import NethraField

T = 0.6
STEPS = 20
DT = T / STEPS
TARGET = 0.02
ETA_OUT = 1200.0
ETA_IN = 2400.0

THRESHOLDS = (
    0.0,
    1e-8,
    3e-8,
    1e-7,
    3e-7,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
)


def conductance(field, evidence):
    e=max(0.0,float(evidence))
    return field.g_min + (field.g_max-field.g_min)*(1.0-math.exp(-e/field.tau))


def integrate(field, currents):
    for n in field.nethra:
        n.activation=0.0
        n.external=0.0
    for n,j in currents.items():
        n.external=float(j)

    es=field._edges()
    charge=[0.0]*len(es)

    def flows(state):
        return [g*(state[a]-state[b]) for a,b,g in es]

    for _ in range(STEPS):
        a0={n:n.activation for n in field.nethra}
        k1=field._derivative_at(a0)
        a1={n:a0[n]+.5*DT*k1[n] for n in field.nethra}; k2=field._derivative_at(a1)
        a2={n:a0[n]+.5*DT*k2[n] for n in field.nethra}; k3=field._derivative_at(a2)
        a3={n:a0[n]+DT*k3[n] for n in field.nethra}; k4=field._derivative_at(a3)
        q0,q1,q2,q3=flows(a0),flows(a1),flows(a2),flows(a3)

        for i in range(len(es)):
            charge[i]+=DT*(q0[i]+2*q1[i]+2*q2[i]+q3[i])/6.0
        for n in field.nethra:
            n.activation=a0[n]+DT*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.0

    for n in field.nethra:
        n.external=0.0
    return es,charge


def flow_between(es,charge,src,dst):
    for (a,b,g),q in zip(es,charge):
        if a is src and b is dst:
            return q
        if a is dst and b is src:
            return -q
    return 0.0


def gated(evidence,tension,theta,eta):
    if abs(tension) <= theta:
        return evidence,False
    return max(0.0,evidence + eta*tension),True


def set_route_evidence(relation,evidence):
    for conditions in relation.routes.values():
        conditions.clear()
        conditions[frozenset()] = max(0.0,float(evidence))


def get_route_evidence(relation):
    vals=[]
    for conditions in relation.routes.values():
        vals.extend(conditions.values())
    return max((float(v) for v in vals),default=0.0)


def add_relation(field,*members,evidence=0.0):
    r=field.new()
    field._route(r,members,frozenset(),1)
    set_route_evidence(r,evidence)
    return r


def existing_relation_frequency(theta,probability,seed,trials=2500):
    rng=random.Random(seed)
    f=NethraField(leakage=.6,convergence_gain=0.0)
    a=f.new(); y=f.new(); r=add_relation(f,a,y,evidence=0.0)
    updates=0

    for _ in range(trials):
        es,q=integrate(f,{a:1.0})
        py=max(0.0,flow_between(es,q,r,y))
        sy=T*TARGET if rng.random()<probability else 0.0
        tension=py*(sy-py)
        e,changed=gated(get_route_evidence(r),tension,theta,ETA_OUT)
        if changed:
            set_route_evidence(r,e)
            updates+=1

    es,q=integrate(f,{a:1.0})
    py=max(0.0,flow_between(es,q,r,y))
    return {
        "evidence":get_route_evidence(r),
        "prediction":py,
        "update_fraction":updates/trials,
    }


def regime_change(theta,seed,trials=2000):
    rng=random.Random(seed)
    f=NethraField(leakage=.6,convergence_gain=0.0)
    a=f.new(); y=f.new(); r=add_relation(f,a,y,evidence=0.0)
    updates=[0,0]

    for phase,prob in enumerate((.9,.1)):
        for _ in range(trials):
            es,q=integrate(f,{a:1.0})
            py=max(0.0,flow_between(es,q,r,y))
            sy=T*TARGET if rng.random()<prob else 0.0
            tension=py*(sy-py)
            e,changed=gated(get_route_evidence(r),tension,theta,ETA_OUT)
            if changed:
                set_route_evidence(r,e); updates[phase]+=1
        if phase==0:
            high_e=get_route_evidence(r)
            high_p=py

    es,q=integrate(f,{a:1.0})
    low_p=max(0.0,flow_between(es,q,r,y))
    return {
        "high_evidence":high_e,
        "low_evidence":get_route_evidence(r),
        "high_prediction":high_p,
        "low_prediction":low_p,
        "update_fraction_high":updates[0]/trials,
        "update_fraction_low":updates[1]/trials,
    }


def redundant_predictor(theta,seed,trials=2200):
    rng=random.Random(seed)
    f=NethraField(leakage=.6,convergence_gain=0.0)
    c=f.new(); a=f.new(); y=f.new()
    base=add_relation(f,c,y,evidence=0.0)

    for _ in range(trials):
        es,q=integrate(f,{c:1.0})
        pb=max(0.0,flow_between(es,q,base,y))
        sy=T*TARGET if rng.random()<.9 else 0.0
        tb=pb*(sy-pb)
        e,changed=gated(get_route_evidence(base),tb,theta,ETA_OUT)
        if changed: set_route_evidence(base,e)

    base_before=get_route_evidence(base)
    cand=add_relation(f,a,y,evidence=0.0)
    cand_updates=0

    for _ in range(trials):
        aon=rng.random()<.5
        cur={c:1.0}
        if aon: cur[a]=1.0
        es,q=integrate(f,cur)
        pb=max(0.0,flow_between(es,q,base,y))
        pc=max(0.0,flow_between(es,q,cand,y)) if aon else 0.0
        sy=T*TARGET if rng.random()<.9 else 0.0
        eps=sy-(pb+pc)

        eb,cb=gated(get_route_evidence(base),pb*eps,theta,ETA_OUT)
        if cb: set_route_evidence(base,eb)
        if aon:
            ec,cc=gated(get_route_evidence(cand),pc*eps,theta,ETA_OUT)
            if cc:
                set_route_evidence(cand,ec); cand_updates+=1

    return {
        "base_before":base_before,
        "base_after":get_route_evidence(base),
        "candidate":get_route_evidence(cand),
        "candidate_update_fraction":cand_updates/trials,
    }


def whole_support(theta,seed,trials=5000,nuisance=8):
    """One overinclusive Nethra, one true source A, many independent nuisance members."""
    rng=random.Random(seed)
    f=NethraField(leakage=.6,convergence_gain=0.0)
    a=f.new(); nuis=[f.new() for _ in range(nuisance)]; y=f.new(); r=f.new()
    members=(a,*nuis,y)
    f._route(r,members,frozenset(),0)
    ev={frozenset((r,m)):0.0 for m in members}

    def custom_edges():
        return tuple((r,m,conductance(f,ev[frozenset((r,m))])) for m in members)
    f._edges=custom_edges

    updates=0
    for _ in range(trials):
        aon=rng.random()<.5
        cur={}
        if aon: cur[a]=1.0
        for m in nuis:
            if rng.random()<.5: cur[m]=1.0
        yon=rng.random()<(.85 if aon else .15)

        es,q=integrate(f,cur)
        py=max(0.0,flow_between(es,q,r,y))
        sy=T*TARGET if yon else 0.0
        tension=py*(sy-py)

        keyy=frozenset((r,y))
        ey,cy=gated(ev[keyy],tension,theta,ETA_OUT)
        if cy:
            ev[keyy]=ey; updates+=1

        supplies={}
        for m in (a,*nuis):
            p=max(0.0,flow_between(es,q,m,r))
            if p>0.0: supplies[m]=p
        total=sum(supplies.values())
        if total and abs(tension)>theta:
            for m,p in supplies.items():
                key=frozenset((r,m))
                ev[key]=max(0.0,ev[key]+ETA_IN*tension*(p/total))

    ea=ev[frozenset((r,a))]
    en=[ev[frozenset((r,m))] for m in nuis]

    def response(m):
        es,q=integrate(f,{m:1.0})
        return max(0.0,flow_between(es,q,r,y))

    pa=response(a)
    pn=[response(m) for m in nuis]
    return {
        "true_evidence":ea,
        "nuisance_mean_evidence":sum(en)/len(en),
        "nuisance_max_evidence":max(en),
        "true_response":pa,
        "nuisance_mean_response":sum(pn)/len(pn),
        "nuisance_max_response":max(pn),
        "update_fraction":updates/trials,
        "edges":len(members),
    }


def recursive_relation(theta,seed,trials=3200):
    """Use an already-learned Nethra itself as context for a later relation."""
    rng=random.Random(seed)
    f=NethraField(leakage=.6,convergence_gain=0.0)
    a=f.new(); b=f.new(); y=f.new()
    lower=add_relation(f,a,b,evidence=30.0)
    upper=add_relation(f,lower,y,evidence=0.0)
    updates=0

    for _ in range(trials):
        # Driving a and b together activates the ordinary lower relation through the field.
        es,q=integrate(f,{a:1.0,b:1.0})
        py=max(0.0,flow_between(es,q,upper,y))
        # Consequence follows the lower relation's context with noise.
        sy=T*TARGET if rng.random()<.8 else 0.0
        tension=py*(sy-py)
        e,ch=gated(get_route_evidence(upper),tension,theta,ETA_OUT)
        if ch:
            set_route_evidence(upper,e); updates+=1

    es,q=integrate(f,{a:1.0,b:1.0})
    py=max(0.0,flow_between(es,q,upper,y))
    return {
        "upper_evidence":get_route_evidence(upper),
        "prediction":py,
        "update_fraction":updates/trials,
        "lower_activation":lower.activation,
    }


def run_threshold(theta):
    t0=time.perf_counter()
    low=existing_relation_frequency(theta,.1,10001)
    mid=existing_relation_frequency(theta,.5,10002)
    high=existing_relation_frequency(theta,.9,10003)
    regime=regime_change(theta,11001)
    redundant=redundant_predictor(theta,12001)
    support=whole_support(theta,13001)
    recursive=recursive_relation(theta,14001)

    # Flexible success criteria: ordering and useful separation, not exact convergence.
    checks={
        "frequency_order": high["prediction"] > mid["prediction"] > low["prediction"],
        "regime_flexible": regime["low_prediction"] < regime["high_prediction"]*.6,
        "redundant_suppressed": redundant["candidate"] < max(1.0,redundant["base_after"]*.2),
        "support_delineated": support["true_response"] > support["nuisance_max_response"]*1.5,
        "recursive_learns": recursive["prediction"] > 0.002,
    }
    return {
        "threshold":theta,
        "checks":checks,
        "pass_count":sum(checks.values()),
        "frequency":{"low":low,"mid":mid,"high":high},
        "regime":regime,
        "redundant":redundant,
        "support":support,
        "recursive":recursive,
        "seconds":time.perf_counter()-t0,
    }


def main():
    workers=min(len(THRESHOLDS),max(1,os.cpu_count() or 1))
    results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs={pool.submit(run_threshold,t):t for t in THRESHOLDS}
        for fut in as_completed(futs):
            row=fut.result()
            results.append(row)
            print("THRESHOLD_RESULT",json.dumps(row,sort_keys=True),flush=True)
    results.sort(key=lambda x:x["threshold"])

    broad=[r for r in results if r["pass_count"]==5]
    print("=== SUMMARY ===")
    for r in results:
        print(
            f"theta={r['threshold']:.1e} pass={r['pass_count']}/5 "
            f"updates_hi={r['frequency']['high']['update_fraction']:.3f} "
            f"hi/mid/lo={r['frequency']['high']['prediction']:.6f}/"
            f"{r['frequency']['mid']['prediction']:.6f}/"
            f"{r['frequency']['low']['prediction']:.6f} "
            f"regime={r['regime']['high_prediction']:.6f}->{r['regime']['low_prediction']:.6f} "
            f"support_ratio={r['support']['true_response']/max(r['support']['nuisance_max_response'],1e-30):.2f} "
            f"recursive={r['recursive']['prediction']:.6f}"
        )
    print("full_pass_thresholds",[r["threshold"] for r in broad])
    print("all_thresholds_complete")


if __name__=="__main__":
    main()
