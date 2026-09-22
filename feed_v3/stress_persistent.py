from __future__ import annotations

import resource
import time

from core import Nethra,NethraMemory
from world import GROUNDED_COUNT

def rss_mb()->float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0

def main():
    mem=NethraMemory(GROUNDED_COUNT)
    target=400000
    created=0
    offset=1
    before=rss_mb()
    t0=time.perf_counter()
    while created<target:
        for a in range(GROUNDED_COUNT):
            if created>=target:
                break
            b=(a+offset)%GROUNDED_COUNT
            if a==b:
                continue
            key=mem.member_key(a,b)
            if key in mem.by_members:
                continue
            n=Nethra(
                nid=mem.next_nid,
                members=key,
                weight=100.0+(created%100),
                last_step=1000,
                confirmations=1,
            )
            mem.next_nid+=1
            mem.nodes[n.nid]=n
            mem.by_members[key]=n.nid
            mem._index(n)
            created+=1
        offset+=1
    build_s=time.perf_counter()-t0
    after=rss_mb()

    active={i:1.0 for i in range(20)}
    t0=time.perf_counter()
    loops=1000
    total_active=0
    for step in range(1001,1001+loops):
        a=mem.field(active,step)
        total_active+=len(a)
    field_s=time.perf_counter()-t0

    print("persistent_relations",created)
    print("rss_before_mb",before)
    print("rss_after_mb",after)
    print("rss_delta_mb",after-before)
    print("build_s",build_s)
    print("field_us_per_step",1e6*field_s/loops)
    print("mean_field_entries",total_active/loops)

    assert created==target
    assert after-before<1024.0
    assert len(mem.nodes)==GROUNDED_COUNT+target
    print("persistent_graph_stress passed")

if __name__=="__main__":
    main()
