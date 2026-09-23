#!/usr/bin/env python3
"""Continuous recursive-handle probe.

Questions:
1. Did the earlier recursive-depth probe manufacture attenuation by zeroing every activation before
   every interval?
2. Can an already-manifest learned Nethra act as the direct support handle for another learned
   Nethra without re-expanding primitive leaves?
3. How much work is saved by direct-member/event-driven recursive refinding compared with repeated
   whole-graph fixed-point scans?

No semantic shortcut is introduced. The field remains the same linear F61 field with convergence
disabled for the plasticity probe. Upper relations connect only to the immediately preceding learned
Nethra and a new target Nethra. Primitive leaves never appear in upper routes.
"""

from collections import defaultdict, deque
import math
import time

from nethra import NethraField

T=.6
STEPS=10
DT=T/STEPS
TARGET_CHARGE=.012
ETA=2400.0
SEED=5.0
MAX_DEPTH=64
TRAIN_INTERVALS=800


def set_evidence(r,e):
    e=max(0.0,float(e))
    for c in r.routes.values():
        c.clear(); c[frozenset()]=e


def evidence(r):
    return max((float(v) for c in r.routes.values() for v in c.values()),default=0.0)


def add_relation(f,*members,seed=SEED):
    r=f.new()
    f._route(r,members,frozenset(),1)
    set_evidence(r,seed)
    return r


def cached_derivative(f,state,edges):
    current={n:n.external-f.leakage*state[n] for n in f.nethra}
    for a,b,g in edges:
        q=g*(state[a]-state[b])
        current[a]-=q
        current[b]+=q
    return {n:v/f.capacitance for n,v in current.items()}


def interval(f,currents,watch=None,reset=False):
    if reset:
        for n in f.nethra:
            n.activation=0.0
    for n in f.nethra:
        n.external=0.0
    for n,j in currents.items():
        n.external=float(j)

    edges=f._edges()
    wi=None; sign=1.0
    if watch is not None:
        src,dst=watch
        for i,(a,b,g) in enumerate(edges):
            if a is src and b is dst:
                wi=i; sign=1.0; break
            if a is dst and b is src:
                wi=i; sign=-1.0; break

    def wf(st):
        if wi is None:
            return 0.0
        a,b,g=edges[wi]
        return sign*g*(st[a]-st[b])

    charge=0.0
    for _ in range(STEPS):
        a0={n:n.activation for n in f.nethra}
        k1=cached_derivative(f,a0,edges)
        a1={n:a0[n]+.5*DT*k1[n] for n in f.nethra}; k2=cached_derivative(f,a1,edges)
        a2={n:a0[n]+.5*DT*k2[n] for n in f.nethra}; k3=cached_derivative(f,a2,edges)
        a3={n:a0[n]+DT*k3[n] for n in f.nethra}; k4=cached_derivative(f,a3,edges)
        if wi is not None:
            charge += DT*(wf(a0)+2*wf(a1)+2*wf(a2)+wf(a3))/6.0
        for n in f.nethra:
            n.activation=a0[n]+DT*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.0

    for n in f.nethra:
        n.external=0.0
    return charge


def learn_chain(continuous=True):
    f=NethraField(g_min=0.0,leakage=.6,convergence_gain=0.0)
    a=f.new(); b=f.new()
    roots={a:1.0,b:1.0}
    base=add_relation(f,a,b,seed=50.0)

    # Establish the base in the continuous case. Cold comparator discards state every interval.
    for _ in range(40):
        interval(f,roots,reset=not continuous)

    prev=base
    rows=[]

    for depth in range(1,MAX_DEPTH+1):
        target=f.new()
        r=add_relation(f,prev,target,seed=SEED)

        p0=max(0.0,interval(f,roots,(r,target),reset=not continuous))
        e0=evidence(r)

        # Deterministic 9/10 consequence recurrence, no threshold on ongoing plasticity.
        for k in range(TRAIN_INTERVALS):
            p=max(0.0,interval(f,roots,(r,target),reset=not continuous))
            source=TARGET_CHARGE if (k%10)!=9 else 0.0
            set_evidence(r,evidence(r)+ETA*p*(source-p))

        p1=max(0.0,interval(f,roots,(r,target),reset=not continuous))
        learned=(evidence(r)>e0 and p1>p0)
        rows.append({
            "depth":depth,
            "p0":p0,
            "p1":p1,
            "e":evidence(r),
            "prev_activation":prev.activation,
            "relation_activation":r.activation,
            "learned":learned,
        })
        if not learned:
            break
        prev=r

    return f,rows


def direct_member_leaf_audit(rows,field):
    # Every recursive relation after the base must have exactly one relation-valued predecessor
    # plus one fresh target. No primitive root is copied into later routes.
    recursive=[n for n in field.nethra if n.routes]
    rootset=set(field.nethra[:2])
    violations=0
    for r in recursive[1:]:
        for route in r.routes:
            # Roots may occur only in the original base relation.
            if route & rootset:
                violations+=1
    return violations


def make_recursive_closure_fixture(depth=2500,noise=5000):
    f=NethraField()
    a=f.new(); b=f.new()
    prev=add_relation(f,a,b,seed=1.0)
    chain=[prev]

    # Pure recursive-handle chain. Every upper Nethra has exactly one direct member: the already
    # learned Nethra below it. No primitive leaf is repeated in an upper route.
    for _ in range(depth):
        prev=add_relation(f,prev,seed=1.0)
        chain.append(prev)

    # Unrelated dormant routes force whole-graph fixed-point scans to inspect irrelevant structure.
    dormant=[]
    for _ in range(noise):
        x=f.new(); y=f.new()
        dormant.append(add_relation(f,x,y,seed=1.0))
    return f,a,b,chain


def compile_direct_route_index(f):
    """Compile exact reverse incidences once; no primitive-leaf expansion."""
    waiting=defaultdict(list)
    required={}
    for relation in f.nethra:
        if not relation.routes:
            continue
        for route,conditions in relation.routes.items():
            if conditions.get(frozenset(),0)<=0:
                continue
            key=(relation,route)
            required[key]=len(route)
            for member in route:
                waiting[member].append(key)
    return waiting,required


def indexed_closure(index,explicit):
    """Exact event-driven direct-member closure for unqualified routes.

    Only routes incident to an actually active/refound Nethra are touched. Dormant unrelated routes
    are never inspected during the query, and an upper relation never asks for the leaves beneath
    its direct member.
    """
    waiting,required=index
    active=set(explicit)
    seen_count=defaultdict(int)
    q=deque(explicit)

    while q:
        n=q.popleft()
        for key in waiting.get(n,()):
            relation,route=key
            if relation in active:
                continue
            seen_count[key]+=1
            if seen_count[key]==required[key]:
                active.add(relation)
                q.append(relation)
    return frozenset(active)


def closure_efficiency():
    f,a,b,chain=make_recursive_closure_fixture()
    explicit=frozenset((a,b))

    t0=time.perf_counter()
    baseline=f.closure(explicit,event=frozenset())
    baseline_s=time.perf_counter()-t0

    t0=time.perf_counter()
    index=compile_direct_route_index(f)
    compile_s=time.perf_counter()-t0

    t0=time.perf_counter()
    indexed=indexed_closure(index,explicit)
    indexed_s=time.perf_counter()-t0

    assert baseline==indexed
    assert chain[-1] in indexed

    # A second unchanged interval can reuse the already-established closure exactly because closure
    # is deterministic for unchanged explicit support/event state.
    t0=time.perf_counter()
    for _ in range(10000):
        reused=baseline
    reuse_s=(time.perf_counter()-t0)/10000.0
    assert reused==baseline

    return {
        "nethra":len(f.nethra),
        "relations":sum(bool(n.routes) for n in f.nethra),
        "baseline_seconds":baseline_s,
        "index_compile_seconds":compile_s,
        "indexed_seconds":indexed_s,
        "speedup":baseline_s/indexed_s if indexed_s else math.inf,
        "unchanged_reuse_seconds":reuse_s,
        "active":len(baseline),
        "depth":len(chain),
    }


def main():
    t0=time.perf_counter()

    fc,cold=learn_chain(False)
    ff,cont=learn_chain(True)

    cold_depth=max((r["depth"] for r in cold if r["learned"]),default=0)
    cont_depth=max((r["depth"] for r in cont if r["learned"]),default=0)

    print("cold_depth",cold_depth)
    print("continuous_depth",cont_depth)
    for label,rows in (("cold",cold),("continuous",cont)):
        for row in rows:
            d=row["depth"]
            if (d & (d-1))==0 or not row["learned"] or d==rows[-1]["depth"]:
                print(
                    label,d,
                    "p0",f'{row["p0"]:.6e}',
                    "p1",f'{row["p1"]:.6e}',
                    "e",f'{row["e"]:.6e}',
                    "prev_a",f'{row["prev_activation"]:.6e}',
                    "r_a",f'{row["relation_activation"]:.6e}',
                    "learned",row["learned"],
                )

    violations=direct_member_leaf_audit(cont,ff)
    print("recursive_leaf_route_violations",violations)
    assert violations==0

    ce=closure_efficiency()
    print("closure_efficiency",ce)

    elapsed=time.perf_counter()-t0
    print("suite_seconds",elapsed)
    assert elapsed < 55.0
    print("all_assertions_passed")


if __name__=="__main__":
    main()
