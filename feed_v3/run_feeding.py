from __future__ import annotations

import argparse
import gc
import json
import random
import resource
import statistics
import time
from pathlib import Path

from core import NethraMemory,ResourceCloud
from world import GROUNDED_COUNT,MOTOR_COUNT,World,grounded_activation

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

def direct_motor_currents(
    babble:set[int],
    previous_grounded:dict[int,float]|None,
    previous_field:dict[int,float]|None,
    resonance_gain:float,
)->dict[int,float]:
    currents={m:1.0 for m in babble}
    if resonance_gain<=0.0 or previous_field is None:
        return currents
    previous_grounded=previous_grounded or {}
    for m in range(MOTOR_COUNT):
        direct=float(previous_grounded.get(m,0.0))
        resonant=max(0.0,float(previous_field.get(m,0.0))-direct)
        if resonant>0.0:
            currents[m]=min(1.0,currents.get(m,0.0)+resonance_gain*resonant)
    return currents

def developmental_epoch(
    world:World,
    mem:NethraMemory,
    babble:set[int],
    rng:random.Random,
    *,
    seed:int,
    steps:int,
    step0:int,
    sample_rate:float,
    proposal_budget:int,
    candidate_capacity:int,
    persistence_floor:float,
)->tuple[int,dict]:
    cloud=make_cloud(
        sample_rate=sample_rate,
        proposal_budget=proposal_budget,
        candidate_capacity=candidate_capacity,
        seed=seed,
    )
    prev_field=None
    for i in range(steps):
        step=step0+i+1
        toggle_babble(babble,rng)
        currents={m:1.0 for m in babble}
        obs=world.step(currents)
        grounded=grounded_activation(obs)
        field=mem.field(grounded,step)
        if prev_field is not None:
            cloud.observe(prev_field,field,step)
        prev_field=field
    receipt=mem.checkpoint_from_cloud(cloud,step0+steps,persistence_floor)
    return step0+steps,receipt

def build_base(
    path:str,
    *,
    sample_rate:float,
    proposal_budget:int,
    candidate_capacity:int,
    persistence_floor:float,
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

    mem=NethraMemory(GROUNDED_COUNT,half_life=60000.0)
    world=World(balls=False,source=False,energy=False)
    rng=random.Random(18001^0xA5A5)
    babble:set[int]=set()
    step=0
    receipts=[]

    for epoch in range(8):
        step,receipt=developmental_epoch(
            world,mem,babble,rng,
            seed=18001000+epoch,
            steps=12000,
            step0=step,
            sample_rate=sample_rate,
            proposal_budget=proposal_budget,
            candidate_capacity=candidate_capacity,
            persistence_floor=persistence_floor,
        )
        receipts.append(receipt)

    world.enable_balls()

    for epoch in range(8):
        step,receipt=developmental_epoch(
            world,mem,babble,rng,
            seed=18001008+epoch,
            steps=12000,
            step0=step,
            sample_rate=sample_rate,
            proposal_budget=proposal_budget,
            candidate_capacity=candidate_capacity,
            persistence_floor=persistence_floor,
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
    resonance_gain:float,
    feed_rate:float,
    steps:int,
    checkpoint_every:int,
    sample_rate:float,
    proposal_budget:int,
    candidate_capacity:int,
    persistence_floor:float,
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
    previous_grounded=None
    previous_field=None
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
    resonance_current_sum=0.0
    checkpoint_receipts=[]
    max_field_size=0
    start_rss=rss_mb()
    t0=time.perf_counter()

    for i in range(steps):
        step+=1
        toggle_babble(babble,rng)
        currents=direct_motor_currents(
            babble,
            previous_grounded,
            previous_field,
            resonance_gain,
        )
        for m,current in currents.items():
            direct=1.0 if m in babble else 0.0
            resonance_current_sum+=max(0.0,float(current)-direct)
        motor_current_sum+=sum(currents.values())

        obs=world.step(currents)
        grounded=grounded_activation(obs)
        field=mem.field(grounded,step)
        max_field_size=max(max_field_size,len(field))

        if previous_field is not None:
            cloud.observe(previous_field,field,step)

        previous_grounded=grounded
        previous_field=field

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
        "resonance_gain":resonance_gain,
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
        "mean_resonance_motor_current":resonance_current_sum/(steps*MOTOR_COUNT),
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
    ap.add_argument("--steps",type=int,default=60000)
    ap.add_argument("--checkpoint-every",type=int,default=12000)
    ap.add_argument("--sample-rate",type=float,default=.0625)
    ap.add_argument("--proposal-budget",type=int,default=64)
    ap.add_argument("--candidate-capacity",type=int,default=65536)
    ap.add_argument("--persistence-floor",type=float,default=.003)
    ap.add_argument("--seed",type=int,default=22001)
    ap.add_argument("--out",default="feeding_results.json")
    args=ap.parse_args()

    base=build_base(
        args.base,
        sample_rate=args.sample_rate,
        proposal_budget=args.proposal_budget,
        candidate_capacity=args.candidate_capacity,
        persistence_floor=args.persistence_floor,
    )

    conditions=[
        ("random_feed",0.0,.0045),
        ("resonant_feed",1.0,.0045),
        ("resonant_no_energy",1.0,0.0),
    ]

    rows={}
    for name,gain,feed_rate in conditions:
        rows[name]=run_condition(
            args.base,
            seed=args.seed,
            resonance_gain=gain,
            feed_rate=feed_rate,
            steps=args.steps,
            checkpoint_every=args.checkpoint_every,
            sample_rate=args.sample_rate,
            proposal_budget=args.proposal_budget,
            candidate_capacity=args.candidate_capacity,
            persistence_floor=args.persistence_floor,
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
            "output_transduction":"Nethra activation maps directly to continuous actuator current",
        },
    }
    Path(args.out).write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "base":base,
        "conditions":{
            name:{
                k:value
                for k,value in row.items()
                if k not in ("checkpoints",)
            }
            for name,row in rows.items()
        },
        "process_max_rss_mb":rss_mb(),
    },indent=2),flush=True)

if __name__=="__main__":
    main()
