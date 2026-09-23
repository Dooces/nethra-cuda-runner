#!/usr/bin/env python3
"""Efficient recursive-depth probe for admission-gated Nethra plasticity.

The test deliberately separates admission from ongoing plasticity:

1. A new recursive relation is admitted only when the already-active preceding Nethra carries
   enough field activation into a consistently observed future source. The admission test is one
   permissive scalar threshold; it does not classify or search for an exact match.
2. The admitted relation receives one finite weak seed.
3. Thereafter its evidence changes continuously from its own signed local prospective tension.
   No threshold is applied to ordinary plasticity.
4. Only the two bottom Nethra are externally driven during context. Every recursive Nethra above
   them must therefore be reached through the Nethra field itself.

The probe stops only when the preceding learned Nethra can no longer clear the permissive admission
threshold or the newly admitted recursive relation cannot increase its own prediction.

For efficiency the RK4 derivative caches the edge list inside one interval. This is exactly
equivalent when convergence_gain == 0 because evidence/topology do not change during an interval.
The script verifies the cached derivative against NethraField._derivative_at before using it.
"""

from __future__ import annotations

import math
import random
import time
from nethra import NethraField

T = 0.60
STEPS = 10
DT = T / STEPS
TARGET_CURRENT = 0.02
TARGET_CHARGE = T * TARGET_CURRENT
ETA = 2400.0
ADMISSION_THRESHOLDS = (1e-12, 1e-15, 1e-18, 0.0)
ADMISSION_SEED = 5.0
MAX_DEPTH = 128
MAX_TRIALS_PER_DEPTH = 5000
MIN_TRIALS_PER_DEPTH = 500
CHECK_EVERY = 250
STABLE_REL_CHANGE = 1e-4


def set_evidence(relation, value):
    value=max(0.0,float(value))
    for conditions in relation.routes.values():
        conditions.clear()
        conditions[frozenset()] = value


def evidence(relation):
    return max(
        (float(v) for conditions in relation.routes.values() for v in conditions.values()),
        default=0.0,
    )


def add_relation(field, *members, seed):
    relation=field.new()
    field._route(relation, members, frozenset(), 1)
    set_evidence(relation, seed)
    return relation


def cached_derivative(field, activation, edges):
    """Exact F61 derivative for convergence_gain=0 using already-compiled interval edges."""
    current={n:n.external-field.leakage*activation[n] for n in field.nethra}
    for a,b,g in edges:
        q=g*(activation[a]-activation[b])
        current[a]-=q
        current[b]+=q
    return {n:v/field.capacitance for n,v in current.items()}


def verify_cached_derivative():
    rng=random.Random(41001)
    for _ in range(20):
        f=NethraField(g_min=0.0, leakage=.6, convergence_gain=0.0)
        nodes=[f.new() for _ in range(7)]
        for i in range(1,7):
            r=f.new()
            f._route(r,(nodes[i-1],nodes[i]),frozenset(),1)
            set_evidence(r,rng.uniform(.01,80.0))
        for n in f.nethra:
            n.activation=rng.uniform(-.2,1.0)
            n.external=rng.uniform(-.1,.3)
        state={n:n.activation for n in f.nethra}
        edges=f._edges()
        a=f._derivative_at(state)
        b=cached_derivative(f,state,edges)
        err=max(abs(a[n]-b[n]) for n in f.nethra)
        assert err < 1e-13, err


def integrate(field, currents, watch_edge=None):
    """RK4 one finite interval; edge compilation occurs once because topology is fixed within it."""
    for n in field.nethra:
        n.activation=0.0
        n.external=0.0
    for n,j in currents.items():
        n.external=float(j)

    edges=field._edges()
    charge=0.0
    watch_index=None
    watch_sign=1.0
    if watch_edge is not None:
        src,dst=watch_edge
        for i,(a,b,g) in enumerate(edges):
            if a is src and b is dst:
                watch_index=i; watch_sign=1.0; break
            if a is dst and b is src:
                watch_index=i; watch_sign=-1.0; break

    def watched_flow(state):
        if watch_index is None:
            return 0.0
        a,b,g=edges[watch_index]
        return watch_sign*g*(state[a]-state[b])

    for _ in range(STEPS):
        a0={n:n.activation for n in field.nethra}
        k1=cached_derivative(field,a0,edges)
        a1={n:a0[n]+.5*DT*k1[n] for n in field.nethra}
        k2=cached_derivative(field,a1,edges)
        a2={n:a0[n]+.5*DT*k2[n] for n in field.nethra}
        k3=cached_derivative(field,a2,edges)
        a3={n:a0[n]+DT*k3[n] for n in field.nethra}
        k4=cached_derivative(field,a3,edges)

        if watch_index is not None:
            charge += DT*(
                watched_flow(a0)+2*watched_flow(a1)+2*watched_flow(a2)+watched_flow(a3)
            )/6.0

        for n in field.nethra:
            n.activation=a0[n]+DT*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.0

    for n in field.nethra:
        n.external=0.0
    return charge


def microbenchmark():
    f=NethraField(g_min=0.0, leakage=.6, convergence_gain=0.0)
    roots=[f.new() for _ in range(2)]
    prev=add_relation(f,*roots,seed=50.0)
    for _ in range(30):
        y=f.new()
        prev=add_relation(f,prev,y,seed=50.0)

    # Compare actual core derivative against cached equivalent repeatedly on the same state.
    for n in f.nethra:
        n.activation=0.01
    roots[0].external=1.0
    roots[1].external=1.0
    state={n:n.activation for n in f.nethra}

    t0=time.perf_counter()
    for _ in range(1000):
        f._derivative_at(state)
    core_s=time.perf_counter()-t0

    edges=f._edges()
    t0=time.perf_counter()
    for _ in range(1000):
        cached_derivative(f,state,edges)
    cached_s=time.perf_counter()-t0

    for n in f.nethra:
        n.external=0.0

    return {
        "core_seconds":core_s,
        "cached_seconds":cached_s,
        "speedup":core_s/cached_s if cached_s else math.inf,
        "nodes":len(f.nethra),
        "edges":len(edges),
    }


def learn_depth(admission_threshold, phase=0, admission_seed=ADMISSION_SEED):
    f=NethraField(g_min=0.0, leakage=.6, convergence_gain=0.0)
    a=f.new(); b=f.new()

    # First already-earned relation gives recursion a real Nethra rather than a raw source handle.
    prev=add_relation(f,a,b,seed=50.0)
    roots={a:1.0,b:1.0}

    rows=[]
    stop_reason="max_depth"

    for depth in range(1,MAX_DEPTH+1):
        # No direct current enters prev. This is the actual field activation available from below.
        integrate(f,roots)
        upstream_activation=max(0.0,prev.activation)

        # Permissive admission gate over the preceding live Nethra and the witnessed next source.
        admission_score=upstream_activation*TARGET_CHARGE
        if admission_threshold > 0.0 and admission_score <= admission_threshold:
            stop_reason="admission_threshold"
            break

        target=f.new()
        relation=add_relation(f,prev,target,seed=admission_seed)

        initial_e=evidence(relation)
        initial_p=max(0.0,integrate(f,roots,(relation,target)))
        prior_check=initial_p
        trials=0

        # Continuous signed local plasticity. The admission threshold is deliberately absent here.
        for k in range(1,MAX_TRIALS_PER_DEPTH+1):
            p=max(0.0,integrate(f,roots,(relation,target)))
            source=TARGET_CHARGE if ((k+phase)%10)!=0 else 0.0
            tension=p*(source-p)
            set_evidence(relation,evidence(relation)+ETA*tension)
            trials=k

            if k>=MIN_TRIALS_PER_DEPTH and k%CHECK_EVERY==0:
                now=max(0.0,integrate(f,roots,(relation,target)))
                scale=max(abs(now),abs(prior_check),1e-300)
                rel_change=abs(now-prior_check)/scale
                if rel_change < STABLE_REL_CHANGE:
                    break
                prior_check=now

        final_p=max(0.0,integrate(f,roots,(relation,target)))
        final_e=evidence(relation)
        learned=(final_e > initial_e and final_p > initial_p)

        rows.append({
            "depth":depth,
            "upstream_activation":upstream_activation,
            "admission_score":admission_score,
            "initial_prediction":initial_p,
            "final_prediction":final_p,
            "evidence":final_e,
            "trials":trials,
            "learned":learned,
        })

        if not learned or final_p <= 0.0 or not math.isfinite(final_p):
            stop_reason="plasticity_failed"
            break

        # This learned Nethra, not its target and not an external proxy, is the sole recursive
        # structural support handed to the next depth.
        prev=relation

    return {
        "rows":rows,
        "depth_reached":rows[-1]["depth"] if rows else 0,
        "learned_depth":max((row["depth"] for row in rows if row["learned"]),default=0),
        "stop_reason":stop_reason,
        "nodes":len(f.nethra),
        "edges":len(f._edges()),
    }


def main():
    suite_start=time.perf_counter()
    verify_cached_derivative()
    print("cached_derivative_equivalence=PASS")

    bench=microbenchmark()
    print("microbenchmark",bench)

    runs=[]
    for theta in ADMISSION_THRESHOLDS:
        for phase in (0,3,7):
            t0=time.perf_counter()
            result=learn_depth(theta,phase)
            result["seconds"]=time.perf_counter()-t0
            result["threshold"]=theta
            result["phase"]=phase
            runs.append(result)
            print(
                "run",
                "theta",theta,
                "phase",phase,
                "attempted_depth",result["depth_reached"],
                "learned_depth",result["learned_depth"],
                "stop",result["stop_reason"],
                "seconds",round(result["seconds"],4),
                "nodes",result["nodes"],
                "edges",result["edges"],
            )
            for row in result["rows"]:
                d=row["depth"]
                if (d & (d-1))==0 or d==result["depth_reached"]:
                    print(
                        "depth_row",
                        theta,phase,d,
                        "up",f'{row["upstream_activation"]:.6e}',
                        "gate",f'{row["admission_score"]:.6e}',
                        "p0",f'{row["initial_prediction"]:.6e}',
                        "p1",f'{row["final_prediction"]:.6e}',
                        "e",f'{row["evidence"]:.6e}',
                        "trials",row["trials"],
                    )

    by_threshold={}
    for theta in ADMISSION_THRESHOLDS:
        depths=[r["learned_depth"] for r in runs if r["threshold"]==theta]
        by_threshold[theta]=depths
        print("threshold_learned_depths",theta,depths,"minimum",min(depths))

    seed_rows={}
    for seed_e in (1.0,5.0,20.0,50.0):
        t0=time.perf_counter()
        result=learn_depth(0.0,0,seed_e)
        seed_rows[seed_e]=result["learned_depth"]
        print(
            "seed_depth",
            seed_e,
            "learned_depth",result["learned_depth"],
            "attempted_depth",result["depth_reached"],
            "stop",result["stop_reason"],
            "seconds",round(time.perf_counter()-t0,4),
        )

    total=time.perf_counter()-suite_start
    print("suite_seconds",total)

    # Efficiency is itself part of the test contract.
    assert total < 55.0, total
    assert min(min(v) for v in by_threshold.values()) >= 1
    print("all_assertions_passed")


if __name__=="__main__":
    main()
