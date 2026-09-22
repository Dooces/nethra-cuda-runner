from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from core import Nethra, NethraMemory, ResourceCloud
from world import DYNAMIC_OFFSET, MOTOR_COUNT, PRIMITIVE_COUNT, World, support


def active_nethra(mem:NethraMemory, external, step:int, execution_epsilon:float):
    activation=mem.field(external,step)
    active=set(map(int,external))
    active.update(nid for nid,a in activation.items() if a>=execution_epsilon)
    return frozenset(active),activation


def random_toggle(active:set[int],rng:random.Random,p:float=.02):
    for m in range(MOTOR_COUNT):
        if rng.random()<p:
            if m in active:active.remove(m)
            else:active.add(m)


def motor_drive(active:set[int],prev_active,mem:NethraMemory,step:int,rng:random.Random,
                resonance_gain:float,babble_p:float=.02):
    random_toggle(active,rng,babble_p)
    if resonance_gain<=0.0 or prev_active is None:
        return
    activation=mem.field(prev_active,step)
    for m in range(MOTOR_COUNT):
        direct=1.0 if m in prev_active else 0.0
        extra=max(0.0,activation.get(m,0.0)-direct)
        p=1.0-math.exp(-resonance_gain*extra)
        if p>0.0 and rng.random()<p:
            active.add(m)


def development_epoch(world,mem,active,rng,*,seed,steps,step0,sample_rate,persistence_floor,
                      execution_epsilon,field_activity:bool):
    cloud=ResourceCloud(sample_rate,seed)
    prev=None
    for i in range(steps):
        random_toggle(active,rng,.02)
        obs=world.step(active)
        external=support(obs)
        cur,_=active_nethra(mem,external,step0+i+1,execution_epsilon) if field_activity else (external,{})
        if prev is not None:
            cloud.observe(prev,cur,step0+i+1)
        prev=cur
    cp=mem.checkpoint_from_cloud(cloud,step0+steps,persistence_floor)
    return step0+steps,cp


def build_base(path:str,*,sample_rate:float,persistence_floor:float,execution_epsilon:float):
    p=Path(path)
    if p.exists():
        mem,step,meta=NethraMemory.load(p)
        return {"created":False,"step":step,"total_nethra":len(mem.nodes),"constructed":len(mem.nodes)-mem.interface_count,"metadata":meta}
    p.parent.mkdir(parents=True,exist_ok=True)
    mem=NethraMemory(PRIMITIVE_COUNT,half_life=60000.0)
    world=World(balls=False,source=False,energy=False,seed=18001)
    rng=random.Random(18001^0xA5A5);motors=set();step=0
    for e in range(8):
        step,_=development_epoch(world,mem,motors,rng,seed=18001000+e,steps=12000,step0=step,
                                 sample_rate=sample_rate,persistence_floor=persistence_floor,
                                 execution_epsilon=execution_epsilon,field_activity=False)
    world.enable_balls()
    for e in range(8):
        step,_=development_epoch(world,mem,motors,rng,seed=18001008+e,steps=12000,step0=step,
                                 sample_rate=sample_rate,persistence_floor=persistence_floor,
                                 execution_epsilon=execution_epsilon,field_activity=False)
    mem.save(p,step=step,metadata={"phase":"hand_ball","seed":18001})
    # strict same-type receipt
    assert all(type(mem.node(i)) is Nethra for i in range(PRIMITIVE_COUNT))
    assert all(type(n) is Nethra for n in mem.nodes.values())
    return {"created":True,"step":step,"total_nethra":len(mem.nodes),"constructed":len(mem.nodes)-mem.interface_count}


def run_lineage(args):
    (base_path,seed,resonance_gain,feed_rate,steps,checkpoint_every,sample_rate,persistence_floor,
     execution_epsilon)=args
    mem,base_step,_=NethraMemory.load(base_path)
    world=World(balls=True,source=True,energy=True,seed=seed,feed_rate=feed_rate)
    rng=random.Random(seed^0xBADC0DE);motors=set();prev=None;step=base_step
    cloud=ResourceCloud(sample_rate,seed^0xCAFE)
    energy_sum=0.0;min_energy=world.energy;contacts=0;source_contacts=0;low=0;zero=0
    recoveries=0;was_low=False;motor_sum=0;field_motor_mass=0.0
    t0=time.perf_counter()
    for i in range(steps):
        step+=1
        motor_drive(motors,prev,mem,step,rng,resonance_gain)
        obs=world.step(motors)
        external=support(obs)
        cur,activation=active_nethra(mem,external,step,execution_epsilon)
        if prev is not None:
            cloud.observe(prev,cur,step)
        prev=cur
        energy_sum+=obs.energy;min_energy=min(min_energy,obs.energy);motor_sum+=len(motors)
        contacts+=obs.contacts;source_contacts+=int(obs.source_contact)
        if obs.energy<.25:
            low+=1;was_low=True
        if obs.energy<=1e-12:zero+=1
        if was_low and obs.energy>.55:
            recoveries+=1;was_low=False
        for m in range(MOTOR_COUNT):
            field_motor_mass+=max(0.0,activation.get(m,0.0)-(1.0 if m in external else 0.0))
        if (i+1)%checkpoint_every==0:
            mem.checkpoint_from_cloud(cloud,step,persistence_floor)
            cloud=ResourceCloud(sample_rate,(seed^0xCAFE)+(i+1))
    if cloud.n:
        mem.checkpoint_from_cloud(cloud,step,persistence_floor)
    elapsed=time.perf_counter()-t0
    return {
        "seed":seed,"resonance_gain":resonance_gain,"feed_rate":feed_rate,
        "final_energy":world.energy,"mean_energy":energy_sum/steps,"min_energy":min_energy,
        "source_contact_fraction":source_contacts/steps,"ball_contacts":contacts,
        "low_fraction":low/steps,"zero_fraction":zero/steps,"recoveries":recoveries,
        "mean_active_motors":motor_sum/steps,"mean_field_motor_mass":field_motor_mass/(steps*MOTOR_COUNT),
        "total_nethra":len(mem.nodes),"constructed":len(mem.nodes)-mem.interface_count,
        "us_per_step":1e6*elapsed/steps,
    }


def summarize(rows):
    return {
        "n":len(rows),
        "final_energy_mean":statistics.mean(r["final_energy"] for r in rows),
        "mean_energy_mean":statistics.mean(r["mean_energy"] for r in rows),
        "contact_fraction_mean":statistics.mean(r["source_contact_fraction"] for r in rows),
        "low_fraction_mean":statistics.mean(r["low_fraction"] for r in rows),
        "zero_fraction_mean":statistics.mean(r["zero_fraction"] for r in rows),
        "recoveries_total":sum(r["recoveries"] for r in rows),
        "survived_above_025":sum(r["final_energy"]>.25 for r in rows),
        "ended_above_055":sum(r["final_energy"]>.55 for r in rows),
        "mean_active_motors":statistics.mean(r["mean_active_motors"] for r in rows),
        "mean_field_motor_mass":statistics.mean(r["mean_field_motor_mass"] for r in rows),
        "us_per_step_mean":statistics.mean(r["us_per_step"] for r in rows),
        "constructed_mean":statistics.mean(r["constructed"] for r in rows),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",required=True)
    ap.add_argument("--lineages",type=int,default=12)
    ap.add_argument("--workers",type=int,default=12)
    ap.add_argument("--steps",type=int,default=120000)
    ap.add_argument("--checkpoint-every",type=int,default=12000)
    ap.add_argument("--sample-rate",type=float,default=.0625)
    ap.add_argument("--persistence-floor",type=float,default=.003)
    ap.add_argument("--execution-epsilon",type=float,default=.01)
    ap.add_argument("--out",default="feeding_results.json")
    args=ap.parse_args()
    base=build_base(args.base,sample_rate=args.sample_rate,persistence_floor=args.persistence_floor,
                    execution_epsilon=args.execution_epsilon)
    conditions=[
        ("random_feed",0.0,.0045),
        ("resonant_feed",1.0,.0045),
        ("resonant_no_energy",1.0,0.0),
    ]
    jobs=[]
    for ci,(_,gain,feed) in enumerate(conditions):
        for i in range(args.lineages):
            jobs.append((args.base,22001+i,gain,feed,args.steps,args.checkpoint_every,args.sample_rate,
                         args.persistence_floor,args.execution_epsilon))
    workers=max(1,min(args.workers,len(jobs),os.cpu_count() or 1))
    t0=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows=list(ex.map(run_lineage,jobs))
    wall=time.perf_counter()-t0
    grouped={}
    k=0
    for name,_,_ in conditions:
        rr=rows[k:k+args.lineages];k+=args.lineages;grouped[name]=summarize(rr)
    report={
        "base":base,"parameters":vars(args),"conditions":grouped,"wall_s":wall,"rows":rows,
        "learner_receipt":{
            "single_node_type":"Nethra",
            "reward":None,"energy_input":False,"food_label":False,"contact_input":False,
            "action_selector":False,"object_labels":False,
            "output_drive":"same generic Nethra field activation for all 12 actuator Nethra plus ongoing random babbling",
        }
    }
    Path(args.out).write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"base":base,"wall_s":wall,"conditions":grouped},indent=2),flush=True)

if __name__=="__main__":main()
