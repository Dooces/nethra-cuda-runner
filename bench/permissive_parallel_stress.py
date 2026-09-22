from __future__ import annotations
import json, math, os, platform, random, statistics, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from permissive_lazy_decay import LazyStore, run_policy
from fixed_budget_persistence import Cloud

def affinity_cpus():
    try: return len(os.sched_getaffinity(0))
    except Exception: return os.cpu_count() or 1

def mem_gb():
    try:
        pages=os.sysconf("SC_PHYS_PAGES"); size=os.sysconf("SC_PAGE_SIZE")
        return pages*size/(1024**3)
    except Exception: return 0.0

def lineage_task(seed:int):
    r=run_policy(
        f"seed_{seed}", sample_rate=.0625, density_floor=.008,
        half_life=60000.0, windows=8, budget=12000, seed=seed
    )
    return {k:r[k] for k in (
        "name","saved","motor_saved","motor_active","us_per_step",
        "lookup_us_per_step","median_checkpoint_ms","final"
    )}

def cpu_parallel_benchmark():
    cpus=affinity_cpus(); memory=mem_gb()
    # Enough jobs to occupy cores when memory permits. This is scheduler policy only.
    workers=max(1,min(cpus,8 if memory < 32 else 16))
    seeds=list(range(13001,13001+workers))
    t=time.perf_counter()
    serial=[lineage_task(s) for s in seeds[:min(2,len(seeds))]]
    serial_wall=time.perf_counter()-t
    t=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parallel=list(ex.map(lineage_task,seeds))
    parallel_wall=time.perf_counter()-t
    per_job_serial=serial_wall/max(1,len(serial))
    projected_serial=per_job_serial*len(seeds)
    return {
        "logical_cpus":os.cpu_count(),"affinity_cpus":cpus,"memory_gb":memory,
        "workers":workers,"jobs":len(seeds),
        "serial_probe_jobs":len(serial),"serial_probe_wall_s":serial_wall,
        "projected_serial_wall_s":projected_serial,
        "parallel_wall_s":parallel_wall,
        "projected_speedup":projected_serial/max(parallel_wall,1e-12),
        "parallel":parallel,
    }

def intuition_one(seed:int, *, sample_rate=.0625, density_floor=.008, half_life=60000.0, budget=12000):
    A,B,Y=1,2,1000; noise=list(range(3,51)); step=0
    hist=LazyStore(density_floor,half_life); fresh=LazyStore(density_floor,half_life)
    def window(pred, active_prefix, seed2):
        nonlocal step
        c=Cloud(sample_rate,seed ^ 0x6A09E667)
        rr=random.Random(seed2)
        for i in range(budget):
            step += 1
            xs={x for x in noise if rr.random()<.06}
            if i < active_prefix and rr.random()<.08: xs.add(pred)
            ys=set()
            if pred in xs and rr.random()<.90: ys.add(Y)
            elif rr.random()<.02: ys.add(Y)
            c.obs(xs,ys,step)
        return c
    c=window(A,budget,seed+1); hist.checkpoint(c,step); fresh.checkpoint(c,step)
    key=LazyStore.key(A,Y)
    for i in range(20):
        c=window(B,budget,seed+100+i); hist.checkpoint(c,step); fresh.checkpoint(c,step)
    before=hist.r.get(key)
    fresh.r.pop(key,None)
    prefixes=(10,25,50,100,200,400,800)
    rows=[]
    for i,prefix in enumerate(prefixes):
        c=window(A,prefix,seed+1000+i); hist.checkpoint(c,step); fresh.checkpoint(c,step)
        hr=hist.r.get(key); fr=fresh.r.get(key)
        hg=hist.conductance(hr,step) if hr else 0.0
        fg=fresh.conductance(fr,step) if fr else 0.0
        rows.append({"prefix":prefix,"hist_g":hg,"fresh_g":fg,"advantage":hg-fg})
    return {
        "seed":seed,
        "before_return_g":hist.conductance(before,step-7*budget) if before else 0.0,
        "rows":rows,
    }

def intuition_sweep():
    seeds=range(14001,14021)
    rows=[intuition_one(s) for s in seeds]
    prefixes=[x["prefix"] for x in rows[0]["rows"]]
    summary=[]
    for i,p in enumerate(prefixes):
        adv=[r["rows"][i]["advantage"] for r in rows]
        hg=[r["rows"][i]["hist_g"] for r in rows]
        fg=[r["rows"][i]["fresh_g"] for r in rows]
        summary.append({
            "prefix":p,
            "hist_g_mean":statistics.mean(hg),
            "fresh_g_mean":statistics.mean(fg),
            "advantage_mean":statistics.mean(adv),
            "advantage_positive_seeds":sum(x>0 for x in adv),
            "advantage_median":statistics.median(adv),
        })
    return {"seeds":len(rows),"summary":summary,"rows":rows}

def hot_tail_sweep():
    r=run_policy(
        "hot_tail",sample_rate=.0625,density_floor=.003,
        half_life=60000.0,windows=12,budget=12000,seed=15001
    )
    return {
        "saved":r["saved"],"motor_saved":r["motor_saved"],
        "us_per_step":r["us_per_step"],"lookup_us_per_step":r["lookup_us_per_step"],
        "final":r["final"],
    }

def build_sparse(relations:int, primitives:int=30732, seed:int=7):
    import numpy as np
    from scipy import sparse
    rng=np.random.default_rng(seed)
    n=primitives+relations
    a=rng.integers(0,primitives,size=relations,dtype=np.int64)
    b=rng.integers(0,primitives,size=relations,dtype=np.int64)
    same=a==b; b[same]=(b[same]+1)%primitives
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
    return sparse.csr_matrix((vals,(rows,cols)),shape=(n,n),dtype=np.float32)

def median_matvec_cpu(M,loops=300,repeats=5):
    import numpy as np
    x=np.ones(M.shape[0],dtype=np.float32)
    for _ in range(10): x=M@x
    samples=[]
    for _ in range(repeats):
        x.fill(1); t=time.perf_counter()
        for __ in range(loops): x=M@x
        samples.append((time.perf_counter()-t)/loops)
    return 1e6*statistics.median(samples)

def gpu_sparse_benchmark():
    out={"available":False,"cases":[]}
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cps
        out["available"]=True
        out["cupy"]=cp.__version__
        props=cp.cuda.runtime.getDeviceProperties(0)
        name=props.get("name",b"")
        if isinstance(name,bytes): name=name.decode(errors="replace")
        out["device"]=name
        for relations in (9000,50000,200000,400000):
            M=build_sparse(relations,seed=relations+17)
            cpu_us=median_matvec_cpu(M)
            t=time.perf_counter(); G=cps.csr_matrix(M); x=cp.ones(M.shape[0],dtype=cp.float32); cp.cuda.Device().synchronize()
            upload_s=time.perf_counter()-t
            for _ in range(20): x=G@x
            cp.cuda.Device().synchronize()
            loops=1000 if relations<=50000 else 300
            samples=[]
            for _ in range(5):
                x.fill(1); cp.cuda.Device().synchronize(); t=time.perf_counter()
                for __ in range(loops): x=G@x
                cp.cuda.Device().synchronize(); samples.append((time.perf_counter()-t)/loops)
            gpu_us=1e6*statistics.median(samples)
            out["cases"].append({
                "relations":relations,"dimension":M.shape[0],"nnz":int(M.nnz),
                "cpu_us":cpu_us,"gpu_us":gpu_us,
                "gpu_speedup":cpu_us/max(gpu_us,1e-12),"upload_s":upload_s
            })
    except Exception as e:
        out["error"]=repr(e)
    return out

def main():
    report={
        "host":{"hostname":platform.node(),"platform":platform.platform(),"python":platform.python_version()},
        "cpu_parallel":cpu_parallel_benchmark(),
        "intuition":intuition_sweep(),
        "hot_tail":hot_tail_sweep(),
        "gpu_sparse":gpu_sparse_benchmark(),
    }
    Path("permissive_parallel_stress.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "host":report["host"],
        "cpu_parallel":{k:v for k,v in report["cpu_parallel"].items() if k!="parallel"},
        "parallel_jobs":[{k:r[k] for k in ("name","saved","motor_saved","motor_active","us_per_step","lookup_us_per_step")} for r in report["cpu_parallel"]["parallel"]],
        "intuition_summary":report["intuition"]["summary"],
        "hot_tail":report["hot_tail"],
        "gpu_sparse":report["gpu_sparse"],
    },indent=2),flush=True)

if __name__=="__main__":
    main()
