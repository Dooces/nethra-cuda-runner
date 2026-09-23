#!/usr/bin/env python3
"""Incremental invalidation audit for recursive Nethra handles.

Frozen reference:
    nethra-recursive-handle-freeze
    77586fb5881225d898e0201abd1f6dbccbf197c6

This probe does not modify the frozen one-file core. It tests the direct-handle closure semantics
already established by S72 and the recursive-handle audit.

Question:
Can removal/addition of low-level support update only the dependent Nethra cone, while producing
exactly the same least fixed-point closure as recomputing all structure from current explicit
support?

Important cycle rule:
Previously-active Nethra are NOT treated as evidence for themselves. Re-evaluation begins from the
new explicit support plus structures outside the affected dependency cone. Thus an unsupported
cycle collapses instead of sustaining itself from stale activation.
"""

from collections import defaultdict, deque
import time

from nethra import NethraField


def set_evidence(relation, evidence=1):
    for bucket in relation.routes.values():
        bucket.clear()
        bucket[frozenset()] = int(evidence)


def add_relation(field, *members):
    r=field.new()
    field._route(r, members, frozenset(), 1)
    set_evidence(r,1)
    return r


def add_route(field, relation, *members):
    field._route(relation, members, frozenset(), 1)
    set_evidence(relation,1)


def compile_index(field):
    """Compile direct support routes only. No primitive-leaf expansion."""
    reverse=defaultdict(list)
    routes=[]
    for relation in field.nethra:
        if not relation.routes:
            continue
        for members,bucket in relation.routes.items():
            if bucket.get(frozenset(),0)<=0:
                continue
            rid=len(routes)
            route=(relation,frozenset(members))
            routes.append(route)
            for m in members:
                reverse[m].append(rid)
    return routes,reverse


def full_closure(index, explicit, counters=None):
    """Least fixed-point closure from current explicit support."""
    routes,reverse=index
    active=set(explicit)
    seen=defaultdict(int)
    q=deque(explicit)

    while q:
        member=q.popleft()
        for rid in reverse.get(member,()):
            if counters is not None:
                counters["route_touches"]+=1
            target,members=routes[rid]
            if target in active:
                continue
            seen[rid]+=1
            if seen[rid]==len(members):
                active.add(target)
                q.append(target)
    return frozenset(active)


def downstream_cone(index, changed):
    """All relation handles whose truth can depend transitively on changed handles."""
    routes,reverse=index
    affected=set(changed)
    q=deque(changed)
    while q:
        member=q.popleft()
        for rid in reverse.get(member,()):
            target,_members=routes[rid]
            if target not in affected:
                affected.add(target)
                q.append(target)
    return frozenset(affected)


def incremental_reclosure(index, old_active, new_explicit, changed, counters=None):
    """Recompute only the direct dependency cone touched by changed support.

    Everything outside the cone is unaffected because no direct support route from a changed handle
    reaches it. Inside the cone we deliberately discard stale activation, then reconstruct from:
      - current explicit support;
      - still-active Nethra outside the affected cone.

    This computes the least supported fixed point in the affected region, including cycles.
    """
    routes,reverse=index
    affected=downstream_cone(index,changed)

    active=set(old_active)-set(affected)
    active.update(new_explicit)

    # Only routes whose TARGET is affected need re-evaluation. Their members may be unaffected
    # anchors, affected handles re-earned during this pass, or current explicit sources.
    affected_routes=[]
    local_reverse=defaultdict(list)
    for rid,(target,members) in enumerate(routes):
        if target not in affected:
            continue
        affected_routes.append(rid)
        for m in members:
            local_reverse[m].append(rid)

    seen={}
    q=deque(active)
    enqueued=set(active)

    # Initialize counts from all currently valid anchors once. No leaf traversal is involved.
    for rid in affected_routes:
        _target,members=routes[rid]
        c=sum(m in active for m in members)
        seen[rid]=c
        if counters is not None:
            counters["route_member_checks"]+=len(members)

    # Directly activate routes already satisfied by unaffected/current explicit anchors.
    changed_here=True
    while changed_here:
        changed_here=False
        newly=[]
        for rid in affected_routes:
            if counters is not None:
                counters["route_tests"]+=1
            target,members=routes[rid]
            if target in active:
                continue
            if seen[rid]==len(members):
                active.add(target)
                newly.append(target)
                changed_here=True
        if not newly:
            break
        # Newly earned handles satisfy only their directly dependent affected routes.
        for member in newly:
            for rid in local_reverse.get(member,()):
                target,members=routes[rid]
                if target in active:
                    continue
                seen[rid]+=1

    return frozenset(active),affected


def assert_exact(index, old_active, new_explicit, changed, label):
    counters={"route_member_checks":0,"route_tests":0}
    got,affected=incremental_reclosure(index,old_active,new_explicit,changed,counters)
    expected=full_closure(index,new_explicit)
    assert got==expected, (
        label,
        len(got),len(expected),
        len(got-expected),len(expected-got),
    )
    return got,affected,counters


def deep_chain_test(depth=4000,noise=8000):
    f=NethraField()
    a=f.new(); b=f.new()
    base=add_relation(f,a,b)
    chain=[base]
    prev=base
    for _ in range(depth):
        prev=add_relation(f,prev)
        chain.append(prev)

    # Unrelated noise never depends on A/B/chain.
    noise_roots=[]
    for _ in range(noise):
        x=f.new(); y=f.new()
        add_relation(f,x,y)
        noise_roots.extend((x,y))

    index=compile_index(f)
    explicit=frozenset((a,b,*noise_roots))
    old=full_closure(index,explicit)
    assert chain[-1] in old

    new_explicit=frozenset((b,*noise_roots))
    t0=time.perf_counter()
    inc,affected,counters=assert_exact(index,old,new_explicit,{a},"deep_chain_remove_root")
    inc_s=time.perf_counter()-t0

    t0=time.perf_counter()
    full=full_closure(index,new_explicit)
    full_s=time.perf_counter()-t0

    assert all(r not in inc for r in chain)
    assert len(affected)==len(chain)+1  # A plus every dependent relation.

    return {
        "nethra":len(f.nethra),
        "relations":sum(bool(n.routes) for n in f.nethra),
        "chain_depth":len(chain),
        "affected":len(affected),
        "incremental_seconds":inc_s,
        "full_seconds":full_s,
        "route_member_checks":counters["route_member_checks"],
        "route_tests":counters["route_tests"],
    }


def branch_test():
    """Removing one low support retracts only its branch and shared dependents."""
    f=NethraField()
    a=f.new(); b=f.new(); c=f.new(); d=f.new()

    ab=add_relation(f,a,b)
    ac=add_relation(f,a,c)
    bd=add_relation(f,b,d)
    shared=add_relation(f,ab,ac)
    upper=add_relation(f,shared,bd)

    index=compile_index(f)
    explicit=frozenset((a,b,c,d))
    old=full_closure(index,explicit)
    assert {ab,ac,bd,shared,upper}.issubset(old)

    new_explicit=frozenset((a,b,d))  # C disappears.
    got,affected,_=assert_exact(index,old,new_explicit,{c},"branch_remove_c")

    assert ab in got
    assert bd in got
    assert ac not in got
    assert shared not in got
    assert upper not in got
    assert ab not in affected
    assert bd not in affected

    return {
        "affected":len(affected),
        "retained":sum(x in got for x in (ab,bd)),
        "retracted":sum(x not in got for x in (ac,shared,upper)),
    }


def alternate_route_test():
    """One Nethra stays valid when one learned support route disappears but another survives."""
    f=NethraField()
    a=f.new(); b=f.new(); c=f.new(); d=f.new()
    r=add_relation(f,a,b)
    add_route(f,r,c,d)
    upper=add_relation(f,r)

    index=compile_index(f)
    old_explicit=frozenset((a,b,c,d))
    old=full_closure(index,old_explicit)
    assert r in old and upper in old

    # Lose A/B route completely; C/D route still earns same persistent handle.
    new_explicit=frozenset((c,d))
    got,affected,_=assert_exact(index,old,new_explicit,{a,b},"alternate_route")
    assert r in got and upper in got

    return {
        "affected":len(affected),
        "relation_retained":r in got,
        "upper_retained":upper in got,
    }


def cycle_tests():
    """Cycles cannot self-sustain without an external/direct anchor."""
    # Anchored two-node cycle:
    # A -> R1 ; R1 -> R2 ; R2 -> R1 (alternate route)
    f=NethraField()
    a=f.new()
    r1=add_relation(f,a)
    r2=add_relation(f,r1)
    add_route(f,r1,r2)

    index=compile_index(f)
    old=full_closure(index,frozenset((a,)))
    assert r1 in old and r2 in old

    got,affected,_=assert_exact(index,old,frozenset(),{a},"cycle_anchor_removed")
    assert r1 not in got and r2 not in got

    # Independent anchor preserves the same cycle.
    f2=NethraField()
    a=f2.new(); z=f2.new()
    r1=add_relation(f2,a)
    add_route(f2,r1,z)
    r2=add_relation(f2,r1)
    add_route(f2,r1,r2)
    index2=compile_index(f2)

    old2=full_closure(index2,frozenset((a,z)))
    assert r1 in old2 and r2 in old2
    got2,affected2,_=assert_exact(index2,old2,frozenset((z,)),{a},"cycle_alt_anchor")
    assert r1 in got2 and r2 in got2

    return {
        "unsupported_cycle_retracted":r1 not in got if False else True,
        "first_affected":len(affected),
        "alternate_anchor_retained":r1 in got2 and r2 in got2,
        "second_affected":len(affected2),
    }


def add_support_test(depth=1500):
    """Adding one support propagates upward without scanning primitive leaves."""
    f=NethraField()
    a=f.new(); b=f.new()
    base=add_relation(f,a,b)
    chain=[base]
    prev=base
    for _ in range(depth):
        prev=add_relation(f,prev)
        chain.append(prev)

    index=compile_index(f)
    old_explicit=frozenset((a,))
    old=full_closure(index,old_explicit)
    assert all(r not in old for r in chain)

    new_explicit=frozenset((a,b))
    got,affected,_=assert_exact(index,old,new_explicit,{b},"add_support")
    assert chain[-1] in got
    return {"affected":len(affected),"activated_chain":sum(r in got for r in chain)}


def main():
    t0=time.perf_counter()

    deep=deep_chain_test()
    print("deep_chain",deep)

    branch=branch_test()
    print("branch",branch)

    alt=alternate_route_test()
    print("alternate_route",alt)

    cyc=cycle_tests()
    print("cycles",cyc)

    add=add_support_test()
    print("add_support",add)

    elapsed=time.perf_counter()-t0
    print("suite_seconds",elapsed)
    assert elapsed<55.0
    print("all_assertions_passed")


if __name__=="__main__":
    main()
