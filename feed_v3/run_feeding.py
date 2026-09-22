from __future__ import annotations

import argparse
import gc
import json
import random
import resource
import time
from pathlib import Path

from core import NethraMemory,ResourceCloud
from world import GROUNDED_COUNT,MOTOR_COUNT,World,sensory_current

def rss_mb()->float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0

def toggle_babble(active:set[int],rng:random.Random,p:float=.02)->None:
    for m in range(MOTOR_COUNT):
        if rng.random()<p:
            if m in active:
                active.remove(m)
            else:
                active.add(m)

def make_cloud(*,sample_rate:float,proposal_budget:int,candidate_capacity:int,seed:int)->ResourceCloud:
    return ResourceCloud(
        sample_rate=sample_rate,
        proposal_budget=proposal_budget,
        candidate_capacity=candidate_capacity,
        seed=seed,
    )

def source_current(obs,babble:set[int])->dict[int,float]:
    current=sensory_current(obs)
    for m in babble:
        current[m]=current.get(m,0.0)+1.0
    return current

def motor_currents_from_field(field:dict[int,float])->dict[int,float]:
    return {
        m:max(0.0,min(1.0,float(field.get(m,0.0))))
        for m in range(MOTOR_COUNT)
        if float(field.get(m,0.0))>0.0
    }

def developmental_epoch(
    world:World,
    mem:NethraMemory,
    babble:set[int],
    rng:random.Random,
    field:dict[int,float],
    *,
    seed:int,
    steps:int,
    step0:int,
    sample_rate:float,
    proposal_budget:int,
    candidate_capacity:int,
    persistence_floor:float,
    field_dt:float,
)->tuple[int,dict[int,float],dict]:
    cloud=make_cloud(
        sample_rate=sample_rate,
        proposal_budget=proposal_budget,
        candidate_capacity=candidate_capacity,
        seed=seed,
    )

    for i in range(steps):
        step=step0+i+1
        motors=motor_currents_from_field(field)
        obs=world.step(motors)
        toggle_babble(babble,rng)
        drive=source_current(obs,babble)
        next_field=mem.field_step(field,drive,step,dt=field_dt)
        if field:
            cloud.observe(field,next_field,step)
        field=next_field

    receipt=mem.checkpoint_from_cloud(cloud,step0+steps,persistence_floor)
    return step0+steps,field,receipt

def build_base(
    path:str,
    *,
    sample_rate:float,
    proposal_budget:int,
    candidate_capacity:int,
    persistence_floor:float,
    field_dt:float,
)->dict:
    p=Path(path)
    if p.exists():
        mem,step,meta=NethraMemory.load(p)
        return {
            "created":False,
            "step":step,
            "constructed":len(mem.nodes)-mem.grounded_count,
            "total_nethra":len(mem.nodes),
            "metadata":meta,
        }

    mem=NethraMemory(GROUNDED_COUNT,half_life=60000.0,leakage=1.0)
    world=World(balls=False,source=False,energy=False)
    rng=random.Random(18001^0xA5A5)
    babble:set[int]=set()
    field:dict[int,float]={}
    step=0
    receipts=[]

    for epoch in range(8):
        step,field,receipt=developmental_epoch(
            world,mem,babble,rng,field,
            seed=18001000+epoch,
            steps=12000,
            step0=step,
            sample_rate=sample_rate,
            proposal_budget=proposal_budget,
            candidate_capacity=candidate_capacity,
            persistence_floor=persistence_floor,
            field_dt=field_dt,
        )
        receipts.append(receipt)

    world.enable_balls()

    for epoch in range(8):
        step,field,receipt=developmental_epoch(
            world,mem,babble,rng,field,
            seed=18001008+epoch,
            steps=12000,
            step0=step,
            sample_rate=sample_rate,
            proposal_budget=proposal_budget,
            candidate_capacity=candidate_capacity,
            persistence_floor=persistence_floor,
            field_dt=field_dt,
        )
        receipts.append(receipt)

    mem.save(
        p,
        step=step,
        metadata={
            "phase":"hand_ball",
            "seed":18001,
            "sample_rate":sample_rate,
            "proposal_budget":proposal_budget,
            "candidate_capacity":candidate_capacity,
            "persistence_floor":persistence_floor,
            "field_dt":field_dt,
        },
    )
    return {
        "created":True,
        "step":step,
        "constructed":len(mem.nodes)-mem.grounded_count,
        "total_nethra":len(mem.nodes),
        "max_epoch_candidates":max(r["candidates"] for r in receipts),
        "max_epoch_dropped":max(r["dropped"] for r in receipts),
        "rss_mb":rss_mb(),
    }

def run_condition(
    base_path:str,
    *,
    seed:int,
    use_field_outputs:bool,
    feed_rate:float,
    steps:int,
    checkpoint_every:int,
    sample_rate:float,
    proposal_budget:int,
    candidate_capacity:int,
    persistence_floor:float,
    field_dt:float,
)->dict:
    mem,base_step,_=NethraMemory.load(base_path)
    world=World(
        balls=True,
        source=True,
        energy=True,
        feed_rate=feed_rate,
        initial_energy=.60,
    )
    rng=random.Random(seed^0xBADC0DE)
    babble:set[int]=set()
    field:dict[int,float]={}
    cloud=make_cloud(
        sample_rate=sample_rate,
        proposal_budget=proposal_budget,
        candidate_capacity=candidate_capacity,
        seed=seed^0xCAFE,
    )

    step=base_step
    energy_sum=0.0
    min_energy=world.energy
    source_contacts=0
    ball_contacts=0
    low_steps=0
    zero_steps=0
    recoveries=0
    was_low=False
    motor_current_sum=0.0
    checkpoint_receipts=[]
    max_field_size=0
    max_output_activation=0.0
    start_rss=rss_mb()
    t0=time.perf_counter()

    for i in range(steps):
        step+=1
        if use_field_outputs:
            motors=motor_currents_from_field(field)
        else:
            motors={m:1.0 for m in babble}

        obs=world.step(motors)
        toggle_babble(babble,rng)
        drive=source_current(obs,babble)
        next_field=mem.field_step(field,drive,step,dt=field_dt)

        if field:
            cloud.observe(field,next_field,step)
        field=next_field

        max_field_size=max(max_field_size,len(field))
        max_output_activation=max(
            max_output_activation,
            max((field.get(m,0.0) for m in range(MOTOR_COUNT)),default=0.0),
        )
        motor_current_sum+=sum(motors.values())
        energy_sum+=obs.energy
        min_energy=min(min_energy,obs.energy)
        source_contacts+=int(obs.source_contact)
        ball_contacts+=obs.contacts

        if obs.energy<.25:
            low_steps+=1
            was_low=True
        if obs.energy<=1e-12:
            zero_steps+=1
        if was_low and obs.energy>.55:
            recoveries+=1
            was_low=False

        if (i+1)%checkpoint_every==0:
            checkpoint_receipts.append(
                mem.checkpoint_from_cloud(cloud,step,persistence_floor)
            )
            cloud=make_cloud(
                sample_rate=sample_rate,
                proposal_budget=proposal_budget,
                candidate_capacity=candidate_capacity,
                seed=(seed^0xCAFE)+(i+1),
            )

    if cloud.n:
        checkpoint_receipts.append(
            mem.checkpoint_from_cloud(cloud,step,persistence_floor)
        )

    elapsed=time.perf_counter()-t0
    return {
        "seed":seed,
        "use_field_outputs":use_field_outputs,
        "feed_rate":feed_rate,
        "steps":steps,
        "final_energy":world.energy,
        "mean_energy":energy_sum/steps,
        "min_energy":min_energy,
        "source_contact_fraction":source_contacts/steps,
        "ball_contacts":ball_contacts,
        "low_fraction":low_steps/steps,
        "zero_fraction":zero_steps/steps,
        "recoveries":recoveries,
        "mean_motor_current":motor_current_sum/(steps*MOTOR_COUNT),
        "max_output_activation":max_output_activation,
        "constructed":len(mem.nodes)-mem.grounded_count,
        "total_nethra":len(mem.nodes),
        "max_field_size":max_field_size,
        "max_checkpoint_candidates":max((r["candidates"] for r in checkpoint_receipts),default=0),
        "max_checkpoint_dropped":max((r["dropped"] for r in checkpoint_receipts),default=0),
        "candidate_capacity":candidate_capacity,
        "proposal_budget":proposal_budget,
        "rss_start_mb":start_rss,
        "rss_max_mb":rss_mb(),
        "wall_s":elapsed,
        "us_per_step":1e6*elapsed/steps,
        "checkpoints":checkpoint_receipts,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",required=True)
    ap.add_argument("--steps",type=int,default=12000)
    ap.add_argument("--checkpoint-every",type=int,default=12000)
    ap.add_argument("--sample-rate",type=float,default=.0625)
    ap.add_argument("--proposal-budget",type=int,default=64)
    ap.add_argument("--candidate-capacity",type=int,default=65536)
    ap.add_argument("--persistence-floor",type=float,default=.003)
    ap.add_argument("--field-dt",type=float,default=.1)
    ap.add_argument("--seed",type=int,default=22001)
    ap.add_argument("--out",default="feeding_results.json")
    args=ap.parse_args()

    base=build_base(
        args.base,
        sample_rate=args.sample_rate,
        proposal_budget=args.proposal_budget,
        candidate_capacity=args.candidate_capacity,
        persistence_floor=args.persistence_floor,
        field_dt=args.field_dt,
    )

    conditions=[
        ("random_feed",False,.0045),
        ("field_feed",True,.0045),
        ("field_no_energy",True,0.0),
    ]

    rows={}
    for name,use_field,feed_rate in conditions:
        rows[name]=run_condition(
            args.base,
            seed=args.seed,
            use_field_outputs=use_field,
            feed_rate=feed_rate,
            steps=args.steps,
            checkpoint_every=args.checkpoint_every,
            sample_rate=args.sample_rate,
            proposal_budget=args.proposal_budget,
            candidate_capacity=args.candidate_capacity,
            persistence_floor=args.persistence_floor,
            field_dt=args.field_dt,
        )
        gc.collect()

    report={
        "base":base,
        "parameters":vars(args),
        "conditions":rows,
        "learner_receipt":{
            "node_type":"Nethra",
            "reward":None,
            "energy_input":False,
            "source_label":False,
            "contact_input":False,
            "action_selector":False,
            "candidate_pair_cartesian_product":False,
            "candidate_memory_bound":args.candidate_capacity,
            "new_pair_proposals_per_sampled_interval":args.proposal_budget,
            "field_law":"source current + symmetric conductance + leakage",
            "output_transduction":"output Nethra field activation directly drives actuator current",
        },
    }
    Path(args.out).write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "base":base,
        "conditions":{
            name:{
                k:value
                for k,value in row.items()
                if k!="checkpoints"
            }
            for name,row in rows.items()
        },
        "process_max_rss_mb":rss_mb(),
    },indent=2),flush=True)

if __name__=="__main__":
    main()
