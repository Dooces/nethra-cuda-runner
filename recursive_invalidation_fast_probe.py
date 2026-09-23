#!/usr/bin/env python3
"""Fast exact incremental invalidation for direct-handle recursive Nethra.

This is a performance refinement of recursive_invalidation_probe.py. It changes no closure
semantics. It uses only precompiled direct route incidences:

    member -> routes that directly depend on member
    target -> routes that can directly refind target

On a support change:
1. discover only the downstream dependency cone;
2. discard stale activation only inside that cone;
3. seed from current explicit support + unaffected still-active handles;
4. rebuild the affected cone by queue propagation.

No primitive leaves are expanded. Cycles are evaluated from the new support boundary, so an
unsupported cycle cannot bootstrap itself from stale activation.
"""

from collections import defaultdict, deque
import random
import time

from nethra import NethraField
from recursive_invalidation_probe import add_relation, add_route


def compile_index(field):
    routes=[]
    reverse=defaultdict(list)
    by_target=defaultdict(list)
    for target in field.nethra:
        if not target.routes:
            continue
        for members,bucket in target.routes.items():
            if bucket.get(frozenset(),0)<=0:
                continue
            rid=len(routes)
            members=frozenset(members)
            routes.append((target,members))
            by_target[target].append(rid)
            for member in members:
                reverse[member].append(rid)
    return routes,reverse,by_target


def full_closure(index, explicit, counters=None):
    routes,reverse,_by_target=index
    active=set(explicit)
    satisfied=defaultdict(int)
    q=deque(explicit)

    while q:
        member=q.popleft()
        for rid in reverse.get(member,()):
            if counters is not None:
                counters["route_touches"]+=1
            target,members=routes[rid]
            if target in active:
                continue
            satisfied[rid]+=1
            if satisfied[rid]==len(members):
                active.add(target)
                q.append(target)
    return frozenset(active)


def downstream_cone(index, changed, counters=None):
    routes,reverse,_by_target=index
    affected=set(changed)
    q=deque(changed)
    while q:
        member=q.popleft()
        for rid in reverse.get(member,()):
            if counters is not None:
                counters["dependency_touches"]+=1
            target,_members=routes[rid]
            if target not in affected:
                affected.add(target)
                q.append(target)
    return frozenset(affected)


def incremental_reclosure(index, old_active, new_explicit, changed, counters=None):
    routes,reverse,by_target=index
    affected=downstream_cone(index,changed,counters)

    # Everything outside the dependency cone is provably unchanged.
    active=set(old_active)-set(affected)
    active.update(new_explicit)

    affected_rids=set()
    for target in affected:
        affected_rids.update(by_target.get(target,()))

    missing={}
    ready=deque()

    # One pass over only affected direct routes. No whole-graph scan.
    for rid in affected_rids:
        _target,members=routes[rid]
        m=sum(member not in active for member in members)
        missing[rid]=m
        if counters is not None:
            counters["route_member_checks"]+=len(members)
        if m==0:
            ready.append(rid)

    while ready:
        rid=ready.popleft()
        target,_members=routes[rid]
        if target in active:
            continue

        active.add(target)
        if counters is not None:
            counters["activated"]+=1

        # Only routes directly depending on this newly re-earned handle can change next.
        for drid in reverse.get(target,()):
            if drid not in affected_rids:
                continue
            if missing[drid]<=0:
                continue
            missing[drid]-=1
            if counters is not None:
                counters["route_updates"]+=1
            if missing[drid]==0:
                ready.append(drid)

    return frozenset(active),affected


def deep_chain(depth=4000,noise=8000):
    f=NethraField()
    a=f.new(); b=f.new()
    base=add_relation(f,a,b)
    chain=[base]
    prev=base
    for _ in range(depth):
        prev=add_relation(f,prev)
        chain.append(prev)

    noise_roots=[]
    for _ in range(noise):
        x=f.new(); y=f.new()
        add_relation(f,x,y)
        noise_roots.extend((x,y))

    idx=compile_index(f)
    explicit=frozenset((a,b,*noise_roots))
    old=full_closure(idx,explicit)
    new_explicit=frozenset((b,*noise_roots))

    c={"dependency_touches":0,"route_member_checks":0,"route_updates":0,"activated":0}
    t0=time.perf_counter()
    inc,affected=incremental_reclosure(idx,old,new_explicit,{a},c)
    inc_s=time.perf_counter()-t0

    fc={"route_touches":0}
    t0=time.perf_counter()
    full=full_closure(idx,new_explicit,fc)
    full_s=time.perf_counter()-t0

    assert inc==full
    assert all(r not in inc for r in chain)

    return {
        "nethra":len(f.nethra),
        "relations":len(idx[0]),
        "chain_depth":len(chain),
        "affected":len(affected),
        "incremental_seconds":inc_s,
        "full_seconds":full_s,
        "speedup":full_s/inc_s if inc_s else float("inf"),
        "incremental_counts":c,
        "full_route_touches":fc["route_touches"],
    }


def localized_change(noise=20000,branch_depth=16):
    """Large world, tiny affected cone."""
    f=NethraField()

    # Main unrelated population.
    noise_roots=[]
    for _ in range(noise):
        x=f.new(); y=f.new()
        add_relation(f,x,y)
        noise_roots.extend((x,y))

    a=f.new(); b=f.new(); c=f.new()
    ab=add_relation(f,a,b)
    ac=add_relation(f,a,c)
    shared=add_relation(f,ab,ac)
    prev=shared
    branch=[ab,ac,shared]
    for _ in range(branch_depth):
        prev=add_relation(f,prev)
        branch.append(prev)

    idx=compile_index(f)
    explicit=frozenset((*noise_roots,a,b,c))
    old=full_closure(idx,explicit)
    new_explicit=frozenset((*noise_roots,a,b))

    t0=time.perf_counter()
    inc,affected=incremental_reclosure(idx,old,new_explicit,{c})
    inc_s=time.perf_counter()-t0

    t0=time.perf_counter()
    full=full_closure(idx,new_explicit)
    full_s=time.perf_counter()-t0

    assert inc==full
    assert ab in inc
    assert all(x not in inc for x in branch[1:])

    return {
        "nethra":len(f.nethra),
        "relations":len(idx[0]),
        "affected":len(affected),
        "incremental_seconds":inc_s,
        "full_seconds":full_s,
        "speedup":full_s/inc_s if inc_s else float("inf"),
    }


def cycle_and_alternate():
    f=NethraField()
    a=f.new(); z=f.new()
    r1=add_relation(f,a)
    add_route(f,r1,z)
    r2=add_relation(f,r1)
    add_route(f,r1,r2)  # r1 <-> r2 cycle plus independent anchors a/z
    upper=add_relation(f,r2)

    idx=compile_index(f)
    old=full_closure(idx,frozenset((a,z)))
    assert {r1,r2,upper}.issubset(old)

    # Remove A; Z independently preserves the cycle and upper handle.
    got,_=incremental_reclosure(idx,old,frozenset((z,)),{a})
    assert got==full_closure(idx,frozenset((z,)))
    assert {r1,r2,upper}.issubset(got)

    # Remove Z too; no stale cycle self-support is allowed.
    got2,_=incremental_reclosure(idx,got,frozenset(),{z})
    assert got2==full_closure(idx,frozenset())
    assert r1 not in got2 and r2 not in got2 and upper not in got2

    return {"alternate_anchor_retained":True,"unsupported_cycle_retracted":True}


def fuzz(worlds=80,changes_per_world=12):
    """Random exactness stress including multi-route relations and cycles."""
    rng=random.Random(99173)
    checked=0

    for _ in range(worlds):
        f=NethraField()
        primitives=[f.new() for _ in range(24)]
        relations=[]

        # Build recursively reusable structure.
        available=list(primitives)
        for _j in range(55):
            k=rng.randint(1,min(4,len(available)))
            members=rng.sample(available,k)
            r=add_relation(f,*members)
            relations.append(r)
            available.append(r)

        # Add alternative routes, including routes through later relations to create cycles.
        for _j in range(25):
            target=rng.choice(relations)
            pool=[n for n in available if n is not target]
            k=rng.randint(1,min(3,len(pool)))
            try:
                add_route(f,target,*rng.sample(pool,k))
            except ValueError:
                pass

        idx=compile_index(f)
        explicit=set(rng.sample(primitives,rng.randint(5,18)))
        active=full_closure(idx,explicit)

        for _k in range(changes_per_world):
            p=rng.choice(primitives)
            if p in explicit:
                explicit.remove(p)
            else:
                explicit.add(p)

            inc,_=incremental_reclosure(idx,active,frozenset(explicit),{p})
            full=full_closure(idx,frozenset(explicit))
            assert inc==full
            active=inc
            checked+=1

    return {"worlds":worlds,"changes":checked}


def main():
    t0=time.perf_counter()

    d=deep_chain()
    print("deep_chain_fast",d)

    l=localized_change()
    print("localized_change",l)

    c=cycle_and_alternate()
    print("cycles",c)

    z=fuzz()
    print("fuzz",z)

    elapsed=time.perf_counter()-t0
    print("suite_seconds",elapsed)
    assert elapsed<55.0
    print("all_assertions_passed")


if __name__=="__main__":
    main()
