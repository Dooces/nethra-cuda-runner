from __future__ import annotations

import resource
import time

from core import Nethra,NethraMemory
from world import GROUNDED_COUNT

def rss_mb()->float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0

def main():
    mem=NethraMemory(GROUNDED_COUNT,leakage=1.0)
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
    graph_rss=rss_mb()

    state={}
    source={i:1.0 for i in range(20)}
    loops=100
    total_entries=0
    max_entries=0
    t0=time.perf_counter()
    for step in range(1001,1001+loops):
        state=mem.field_step(
            state,
            source,
            step,
            dt=.1,
            execution_epsilon=1e-6,
        )
        total_entries+=len(state)
        max_entries=max(max_entries,len(state))
    field_s=time.perf_counter()-t0
    final_rss=rss_mb()

    print("persistent_relations",created)
    print("rss_before_mb",before)
    print("rss_graph_mb",graph_rss)
    print("rss_final_mb",final_rss)
    print("rss_graph_delta_mb",graph_rss-before)
    print("rss_field_delta_mb",final_rss-graph_rss)
    print("build_s",build_s)
    print("field_us_per_step",1e6*field_s/loops)
    print("mean_field_entries",total_entries/loops)
    print("max_field_entries",max_entries)

    assert created==target
    assert graph_rss-before<1024.0
    assert final_rss-before<1024.0
    assert len(mem.nodes)==GROUNDED_COUNT+target
    print("persistent_graph_stress passed")

if __name__=="__main__":
    main()
