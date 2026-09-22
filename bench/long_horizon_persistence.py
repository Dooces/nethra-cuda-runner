from __future__ import annotations
import json, os, platform, statistics, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from permissive_lazy_decay import run_policy, intuition_case

def cpus():
    try:return len(os.sched_getaffinity(0))
    except Exception:return os.cpu_count() or 1

def long_task(args):
    floor,seed=args
    r=run_policy(
        f"f{floor}_s{seed}",
        sample_rate=.0625,
        density_floor=floor,
        half_life=60000.0,
        windows=32,
        budget=12000,
        seed=seed,
    )
    growth=[x["saved"] for x in r["rows"]]
    return {
        "floor":floor,"seed":seed,
        "saved":r["saved"],"motor_saved":r["motor_saved"],"motor_active":r["motor_active"],
        "us_per_step":r["us_per_step"],"lookup_us_per_step":r["lookup_us_per_step"],
        "median_checkpoint_ms":r["median_checkpoint_ms"],"final":r["final"],
        "growth_every4":[growth[i-1] for i in range(4,len(growth)+1,4)],
        "created_every4":[sum(x["created"] for x in r["rows"][i-4:i]) for i in range(4,len(growth)+1,4)],
    }

def long_sweep():
    floors=(.008,.003,.001)
    tasks=[]
    for fi,f in enumerate(floors):
        for j in range(5):tasks.append((f,16001+100*fi+j))
    workers=min(cpus(),len(tasks))
    t=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows=list(ex.map(long_task,tasks))
    wall=time.perf_counter()-t
    summary=[]
    for f in floors:
        rr=[x for x in rows if x["floor"]==f]
        summary.append({
            "floor":f,"seeds":len(rr),
            "saved_mean":statistics.mean(x["saved"] for x in rr),
            "saved_min":min(x["saved"] for x in rr),"saved_max":max(x["saved"] for x in rr),
            "motor_saved_all":sum(x["motor_saved"]==12 for x in rr),
            "motor_active_all":sum(x["motor_active"]==12 for x in rr),
            "us_mean":statistics.mean(x["us_per_step"] for x in rr),
            "lookup_us_mean":statistics.mean(x["lookup_us_per_step"] for x in rr),
            "checkpoint_ms_median":statistics.median(x["median_checkpoint_ms"] for x in rr),
            "active_001_mean":statistics.mean(x["final"]["active"]["0.01"] for x in rr),
            "active_005_mean":statistics.mean(x["final"]["active"]["0.05"] for x in rr),
            "active_010_mean":statistics.mean(x["final"]["active"]["0.1"] for x in rr),
            "growth_every4_mean":[statistics.mean(x["growth_every4"][i] for x in rr) for i in range(8)],
            "created_every4_mean":[statistics.mean(x["created_every4"][i] for x in rr) for i in range(8)],
        })
    return {"workers":workers,"tasks":len(tasks),"wall_s":wall,"summary":summary,"rows":rows}

def gap_task(args):
    half,seed=args
    # Reuse the exact historical-vs-fresh test with no semantic additions.
    return {"half_life":half,"seed":seed,**intuition_case(
        density_floor=.008,sample_rate=.0625,half_life=half,budget=12000
    )}

def gap_sweep():
    # intuition_case itself uses a long fixed gap; vary only physical decay time.
    halves=(30000.0,60000.0,120000.0,240000.0)
    tasks=[(h,17001+i) for h in halves for i in range(4)]
    workers=min(cpus(),len(tasks))
    t=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows=list(ex.map(gap_task,tasks))
    wall=time.perf_counter()-t
    summary=[]
    for h in halves:
        rr=[x for x in rows if x["half_life"]==h]
        for idx,prefix in enumerate((25,50,100,200,400,800,1600,3200)):
            adv=[x["return"][idx]["hist_g"]-x["return"][idx]["fresh_g"] for x in rr]
            if idx in (0,2,4,7):
                summary.append({
                    "half_life":h,"prefix":prefix,
                    "adv_mean":statistics.mean(adv),
                    "positive":sum(v>0 for v in adv),
                })
    return {"workers":workers,"tasks":len(tasks),"wall_s":wall,"summary":summary,"rows":rows}

def main():
    long=long_sweep()
    gaps=gap_sweep()
    out={"host":{"hostname":platform.node(),"platform":platform.platform(),"python":platform.python_version(),"cpus":cpus()},
         "long":long,"gaps":gaps}
    Path("long_horizon_persistence.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"host":out["host"],"long":{"workers":long["workers"],"tasks":long["tasks"],"wall_s":long["wall_s"],"summary":long["summary"]},
                      "gaps":{"workers":gaps["workers"],"tasks":gaps["tasks"],"wall_s":gaps["wall_s"],"summary":gaps["summary"]}},indent=2),flush=True)

if __name__=="__main__":main()
