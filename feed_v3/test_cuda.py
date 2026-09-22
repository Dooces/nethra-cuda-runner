from __future__ import annotations

import math
import random
import time

from core import Nethra,NethraMemory
from cuda_field import CudaFieldRuntime
from world import GROUNDED_COUNT


def add_relation(mem:NethraMemory,a:int,b:int,weight:float,last_step:int=0)->None:
    key=mem.member_key(a,b)
    if key in mem.by_members:
        return
    n=Nethra(
        nid=mem.next_nid,
        members=key,
        weight=float(weight),
        last_step=int(last_step),
        confirmations=1,
    )
    mem.next_nid+=1
    mem.nodes[n.nid]=n
    mem.by_members[key]=n.nid
    mem._index(n)


def build_graph(relations:int,seed:int=1)->NethraMemory:
    rng=random.Random(seed)
    mem=NethraMemory(GROUNDED_COUNT,half_life=60000.0,tau=100.0,leakage=1.0)
    while len(mem.nodes)-mem.grounded_count<relations:
        a=rng.randrange(GROUNDED_COUNT)
        b=rng.randrange(GROUNDED_COUNT)
        if a==b:
            continue
        add_relation(mem,a,b,20.0+rng.random()*180.0,last_step=rng.randrange(0,1000))
    return mem


def test_cpu_cuda_equivalence():
    mem=build_graph(4000,seed=11)
    gpu=CudaFieldRuntime(mem,execution_epsilon=1e-6,dtype="float32")
    cpu_state={}
    rng=random.Random(99)

    for step in range(1001,1101):
        source={}
        for _ in range(24):
            source[rng.randrange(GROUNDED_COUNT)]=rng.random()
        cpu_state=mem.field_step(cpu_state,source,step,dt=.1,execution_epsilon=1e-6)
        gpu_state=gpu.step(source,step,dt=.1)

        keys=set(cpu_state)|set(gpu_state)
        max_abs=max((abs(cpu_state.get(k,0.0)-gpu_state.get(k,0.0)) for k in keys),default=0.0)
        assert max_abs<2e-5,(step,max_abs,len(cpu_state),len(gpu_state))

    print("cuda_equivalence_max_entries",max(len(cpu_state),len(gpu_state)))
    print("cuda_equivalence_passed")


def benchmark_cuda():
    mem=build_graph(200000,seed=23)
    rng=random.Random(101)
    sources=[]
    for _ in range(1000):
        src={}
        for __ in range(20):
            src[rng.randrange(GROUNDED_COUNT)]=1.0
        sources.append(src)

    cpu_state={}
    t=time.perf_counter()
    for i,src in enumerate(sources[:200],start=1001):
        cpu_state=mem.field_step(cpu_state,src,i,dt=.1,execution_epsilon=1e-6)
    cpu_us=1e6*(time.perf_counter()-t)/200

    gpu=CudaFieldRuntime(mem,execution_epsilon=1e-6,dtype="float32")
    gpu.synchronize()
    t=time.perf_counter()
    max_active=0
    for i,src in enumerate(sources,start=1001):
        state=gpu.step(src,i,dt=.1)
        max_active=max(max_active,len(state))
    gpu.synchronize()
    gpu_us=1e6*(time.perf_counter()-t)/len(sources)

    print("cuda_benchmark_relations",200000)
    print("cpu_us_per_step",cpu_us)
    print("gpu_us_per_step",gpu_us)
    print("gpu_speedup",cpu_us/max(gpu_us,1e-12))
    print("gpu_max_active",max_active)
    assert gpu_us<cpu_us
    print("cuda_benchmark_passed")


if __name__=="__main__":
    test_cpu_cuda_equivalence()
    benchmark_cuda()
