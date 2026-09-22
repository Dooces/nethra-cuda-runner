from __future__ import annotations
import json, math, platform, random, statistics, time
from pathlib import Path
import numpy as np
from scipy import sparse
from fixed_budget_persistence import World, sup

def build_field(relations:int, primitives:int=30744, seed:int=21):
    rng=np.random.default_rng(seed)
    n=primitives+relations
    a=rng.integers(0,primitives,size=relations,dtype=np.int64)
    b=rng.integers(0,primitives,size=relations,dtype=np.int64)
    same=a==b;b[same]=(b[same]+1)%primitives
    rn=primitives+np.arange(relations,dtype=np.int64)
    rows=np.concatenate([np.arange(n,dtype=np.int64),rn,rn,a,b])
    cols=np.concatenate([np.arange(n,dtype=np.int64),a,b,rn,rn])
    vals=np.concatenate([
        np.full(n,-.10,dtype=np.float32),
        np.full(relations,.20,dtype=np.float32),
        np.full(relations,.20,dtype=np.float32),
        np.full(relations,.20,dtype=np.float32),
        np.full(relations,.20,dtype=np.float32),
    ])
    return sparse.csr_matrix((vals,(rows,cols)),shape=(n,n),dtype=np.float32), list(zip(a.tolist(),b.tolist()))

def supports(n=8000,seed=25001):
    w=World();r=random.Random(seed);m=set();out=[]
    for _ in range(n):
        for x in range(12):
            if r.random()<.02:
                if x in m:m.remove(x)
                else:m.add(x)
        out.append(np.fromiter(sup(w.step(m)),dtype=np.int64))
    return out

def cpu_online(M, ss):
    x=np.zeros(M.shape[0],dtype=np.float32);prev=np.empty(0,dtype=np.int64)
    # warm
    for ids in ss[:100]:
        x[prev]=0;x[ids]=1;x=M@x;prev=ids
    times=[]
    x.fill(0);prev=np.empty(0,dtype=np.int64)
    t=time.perf_counter()
    for ids in ss:
        x[prev]=0;x[ids]=1;x=M@x;prev=ids
    return 1e6*(time.perf_counter()-t)/len(ss)

def cold_lookup(pairs,ss):
    d={((a,b) if a<b else (b,a)):1 for a,b in pairs}
    t=time.perf_counter();hits=0
    for ids in ss:
        a=sorted(map(int,ids))
        for i,x in enumerate(a):
            for y in a[i+1:]:
                hits += d.get((x,y) if x<y else (y,x),0)
    return {"us":1e6*(time.perf_counter()-t)/len(ss),"hits":hits}

def gpu_case(M,ss):
    import cupy as cp
    import cupyx.scipy.sparse as cps
    G=cps.csr_matrix(M);x=cp.zeros(M.shape[0],dtype=cp.float32)
    cp.cuda.Device().synchronize()
    # Live CPU->GPU sparse-index transfer and sync every step: strict latency path.
    prev=cp.empty(0,dtype=cp.int64)
    for ids in ss[:100]:
        cur=cp.asarray(ids);x[prev]=0;x[cur]=1;x=G@x;prev=cur
    cp.cuda.Device().synchronize()
    x.fill(0);prev=cp.empty(0,dtype=cp.int64);cp.cuda.Device().synchronize()
    t=time.perf_counter()
    for ids in ss:
        cur=cp.asarray(ids);x[prev]=0;x[cur]=1;x=G@x;cp.cuda.Device().synchronize();prev=cur
    latency=1e6*(time.perf_counter()-t)/len(ss)

    # Same online ordering, source IDs already resident and only one final sync.
    gpu_ids=[cp.asarray(v) for v in ss]
    x=cp.zeros(M.shape[0],dtype=cp.float32);prev=cp.empty(0,dtype=cp.int64);cp.cuda.Device().synchronize()
    t=time.perf_counter()
    for cur in gpu_ids:
        x[prev]=0;x[cur]=1;x=G@x;prev=cur
    cp.cuda.Device().synchronize()
    resident=1e6*(time.perf_counter()-t)/len(ss)
    return {"live_transfer_sync_us":latency,"resident_pipelined_us":resident}

def main():
    ss=supports()
    rows=[]
    cold_pairs=None
    for rel in (21000,56000,200000):
        print("CASE",rel,flush=True)
        M,pairs=build_field(rel,seed=rel)
        if rel==200000:cold_pairs=pairs
        cpu=cpu_online(M,ss)
        try:gpu=gpu_case(M,ss)
        except Exception as e:gpu={"error":repr(e)}
        rows.append({"relations":rel,"cpu_online_us":cpu,"gpu":gpu})
    cold=cold_lookup(cold_pairs,ss)
    out={"host":{"hostname":platform.node(),"platform":platform.platform(),"python":platform.python_version()},
         "steps":len(ss),"mean_active_support":statistics.mean(len(x) for x in ss),"rows":rows,"cold_lookup_200k":cold}
    Path("online_hotfield_gpu.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out,indent=2),flush=True)
if __name__=="__main__":main()
