from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from core import NethraMemory, ResourceCloud
from world import (
    BASE_COUNT,DYNAMIC_OFFSET,GRID_H,GRID_W,INPUT_COUNT,MOTOR_COUNT,PIXELS,
    HandBallWorld,support,
)

def endpoint_kind(n:int)->str:
    if n < MOTOR_COUNT:return "motor"
    q=n-MOTOR_COUNT
    if q < PIXELS:return "retina"
    if q < INPUT_COUNT:return "dynamic"
    return "other"

def run_epoch(world,mem,active,rng,*,seed,steps,step0,sample_rate,persistence_floor,pixel_origin=None):
    cloud=ResourceCloud(sample_rate,seed)
    prev=None;contacts=0;active_sum=0;t0=time.perf_counter()
    for i in range(int(steps)):
        for m in range(MOTOR_COUNT):
            if rng.random()<.02:
                if m in active:active.remove(m)
                else:active.add(m)
        obs=world.step(active);cur=support(obs);step=step0+i+1
        if prev is not None:cloud.observe(prev,cur,step)
        prev=cur;contacts+=obs.contacts;active_sum+=len(cur)
        if pixel_origin is not None:
            for p in obs.hand_pixels:pixel_origin[p][0]+=1
            for p in obs.ball_pixels:pixel_origin[p][1]+=1
    cp=mem.checkpoint_from_cloud(cloud,step0+steps,persistence_floor)
    return step0+steps,{"cpu_s":time.perf_counter()-t0,"contacts":contacts,"mean_active":active_sum/max(1,steps),**cp}

def motor_afferent_rows(mem,step):
    rows=[]
    for m in range(MOTOR_COUNT):
        y=MOTOR_COUNT+DYNAMIC_OFFSET+m
        r=mem.relation(m,y)
        rows.append({
            "motor":m,"afferent":y,
            "present":r is not None,
            "weight":mem.effective_weight(r,step) if r else 0.0,
            "conductance":mem.conductance(r,step) if r else 0.0,
            "confirmations":r.confirmations if r else 0,
        })
    return rows

def classify_relations(mem,step,pixel_origin):
    counts=Counter();ball_touched=0;hand_touched=0;mixed_touched=0
    top=sorted(mem.by_nid.values(),key=lambda r:mem.effective_weight(r,step),reverse=True)[:100]
    top_rows=[]
    def retinal_origin(n):
        if endpoint_kind(n)!="retina":return None
        p=n-MOTOR_COUNT;h,b=pixel_origin.get(p,(0,0))
        if h and b:return "mixed"
        if b:return "ball"
        if h:return "hand"
        return "unseen"
    for r in mem.by_nid.values():
        ka,kb=endpoint_kind(r.a),endpoint_kind(r.b)
        counts["-".join(sorted((ka,kb)))]+=1
        origins={x for x in (retinal_origin(r.a),retinal_origin(r.b)) if x}
        if "ball" in origins:ball_touched+=1
        if "hand" in origins:hand_touched+=1
        if "mixed" in origins:mixed_touched+=1
    for r in top:
        top_rows.append({
            "nid":r.nid,"a":r.a,"b":r.b,
            "a_kind":endpoint_kind(r.a),"b_kind":endpoint_kind(r.b),
            "a_origin":retinal_origin(r.a),"b_origin":retinal_origin(r.b),
            "weight":mem.effective_weight(r,step),
            "conductance":mem.conductance(r,step),
            "confirmations":r.confirmations,
        })
    return {"category_counts":dict(counts),"ball_retina_relations":ball_touched,
            "hand_retina_relations":hand_touched,"mixed_retina_relations":mixed_touched,
            "top100":top_rows}

def lineage(seed:int,hand_epochs:int,ball_epochs:int,steps:int,sample_rate:float,persistence_floor:float,half_life:float,checkpoint_dir:str|None):
    mem=NethraMemory(BASE_COUNT,half_life=half_life)
    world=HandBallWorld(balls=False,seed=seed)
    rng=random.Random(seed^0xA5A5);active=set();step=0;hand_rows=[]
    for e in range(hand_epochs):
        step,row=run_epoch(world,mem,active,rng,seed=seed*1000+e,steps=steps,step0=step,
                           sample_rate=sample_rate,persistence_floor=persistence_floor)
        hand_rows.append(row)
    hand_eval=motor_afferent_rows(mem,step)
    checkpoint_verified=False
    if checkpoint_dir is not None:
        p=Path(checkpoint_dir)/f"lineage_{seed}_hand.json"
        mem.save(p,step=step,metadata={"phase":"hand","seed":seed})
        loaded,loaded_step,meta=NethraMemory.load(p)
        assert loaded_step==step and len(loaded.by_key)==len(mem.by_key) and meta["phase"]=="hand"
        mem=loaded;checkpoint_verified=True

    world.enable_balls();origin=defaultdict(lambda:[0,0]);ball_rows=[];contacts=0
    for e in range(ball_epochs):
        step,row=run_epoch(world,mem,active,rng,seed=seed*1000+hand_epochs+e,steps=steps,step0=step,
                           sample_rate=sample_rate,persistence_floor=persistence_floor,pixel_origin=origin)
        contacts+=row["contacts"];ball_rows.append(row)
    final_eval=motor_afferent_rows(mem,step)
    classes=classify_relations(mem,step,origin)
    return {
        "seed":seed,"step":step,"hand_persistent":hand_rows[-1]["persistent"],
        "final_persistent":len(mem.by_key),"checkpoint_verified":checkpoint_verified,
        "hand_motor_relations":sum(x["present"] for x in hand_eval),
        "final_motor_relations":sum(x["present"] for x in final_eval),
        "hand_motor":hand_eval,"final_motor":final_eval,
        "ball_contacts":contacts,"classification":classes,
        "hand_epochs":hand_rows,"ball_epochs":ball_rows,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--lineages",type=int,default=16)
    ap.add_argument("--workers",type=int,default=16)
    ap.add_argument("--hand-epochs",type=int,default=8)
    ap.add_argument("--ball-epochs",type=int,default=8)
    ap.add_argument("--steps",type=int,default=12000)
    ap.add_argument("--sample-rate",type=float,default=.0625)
    ap.add_argument("--persistence-floor",type=float,default=.003,
                    help="software checkpoint compression only")
    ap.add_argument("--half-life",type=float,default=60000.0)
    ap.add_argument("--out",default="hand_ball_results.json")
    args=ap.parse_args()
    workers=max(1,min(args.workers,args.lineages,os.cpu_count() or 1))
    with tempfile.TemporaryDirectory(prefix="nethra_hand_cp_") as cpdir:
        jobs=[(18001+i,args.hand_epochs,args.ball_epochs,args.steps,args.sample_rate,args.persistence_floor,args.half_life,cpdir if i==0 else None)
              for i in range(args.lineages)]
        t=time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows=list(ex.map(lambda x: lineage(*x),jobs))
        wall=time.perf_counter()-t
    summary={
        "lineages":len(rows),"workers":workers,"wall_s":wall,
        "hand_motor_relations":[r["hand_motor_relations"] for r in rows],
        "final_motor_relations":[r["final_motor_relations"] for r in rows],
        "hand_persistent_mean":statistics.mean(r["hand_persistent"] for r in rows),
        "final_persistent_mean":statistics.mean(r["final_persistent"] for r in rows),
        "contacts":[r["ball_contacts"] for r in rows],
        "checkpoint_verified":any(r["checkpoint_verified"] for r in rows),
        "ball_retina_relations_mean":statistics.mean(r["classification"]["ball_retina_relations"] for r in rows),
        "category_counts_total":dict(sum((Counter(r["classification"]["category_counts"]) for r in rows),Counter())),
    }
    report={
        "frozen":{"competition":"threshold-free positive predictive-gain resource sharing",
                  "persistence":"permissive checkpoint materialization; floor is software compression only",
                  "decay":"lazy graded persistence; half-life remains experimental",
                  "deletion":"none","semantic_filters":"none"},
        "parameters":vars(args),"summary":summary,"lineages":rows,
    }
    Path(args.out).write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(summary,indent=2,sort_keys=True),flush=True)

if __name__=="__main__":main()
