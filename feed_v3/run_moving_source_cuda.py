from __future__ import annotations

import argparse
import json
import random
import resource
import time
from pathlib import Path

from core import NethraMemory
from cuda_field import CudaFieldRuntime
from world import World, MOTOR_COUNT
from run_feeding_cuda import (
    make_cloud, field_motor_currents, toggle_babble, grounded_prime,
    observe_if_sampled, rebuild_runtime,
)

POSITIONS=((1.02,0.06),(1.18,0.06),(1.14,-0.14))

def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0

def run(base_path, *, seed, steps, segment_every, checkpoint_every, out):
    mem,base_step,_=NethraMemory.load(base_path)
    runtime=CudaFieldRuntime(mem,execution_epsilon=1e-6,dtype="float32")
    world=World(
        balls=True,source=True,energy=True,
        feed_rate=.0045,basal_cost=.00021,motor_cost=.0000275,
        initial_energy=.60,source_x=POSITIONS[0][0],source_y=POSITIONS[0][1],
    )
    rng=random.Random(seed^0xBADC0DE)
    babble=set()
    cloud=make_cloud(sample_rate=.0625,proposal_budget=64,candidate_capacity=65536,seed=seed^0xCAFE)
    previous_for_sample=None
    step=base_step
    segments=[]
    checkpoints=[]
    seg_contact=seg_energy=seg_motor=0.0
    seg_low=seg_zero=0
    seg_start_energy=world.energy
    t0=time.perf_counter()

    for i in range(steps):
        pos_index=(i//segment_every)%len(POSITIONS)
        sx,sy=POSITIONS[pos_index]
        world.source_x=sx
        world.source_y=sy

        step+=1
        motors=field_motor_currents(runtime)
        obs=world.step(motors)
        toggle_babble(babble,rng)
        runtime.prime(grounded_prime(obs,babble))
        runtime.step({},step,dt=.1,return_active=False)
        previous_for_sample=observe_if_sampled(cloud,runtime,step,previous_for_sample)

        seg_contact+=int(obs.source_contact)
        seg_energy+=obs.energy
        seg_motor+=sum(motors.values())
        seg_low+=int(obs.energy<.25)
        seg_zero+=int(obs.energy<=1e-12)

        if (i+1)%segment_every==0:
            segments.append({
                "segment":len(segments)+1,
                "position_index":pos_index,
                "source":[sx,sy],
                "start_energy":seg_start_energy,
                "final_energy":world.energy,
                "mean_energy":seg_energy/segment_every,
                "contact_fraction":seg_contact/segment_every,
                "low_fraction":seg_low/segment_every,
                "zero_fraction":seg_zero/segment_every,
                "mean_motor_current":seg_motor/(segment_every*MOTOR_COUNT),
            })
            seg_contact=seg_energy=seg_motor=0.0
            seg_low=seg_zero=0
            seg_start_energy=world.energy

        if (i+1)%checkpoint_every==0:
            state=runtime.snapshot()
            checkpoints.append(mem.checkpoint_from_cloud(cloud,step,.003))
            runtime=rebuild_runtime(mem,runtime,state)
            cloud=make_cloud(
                sample_rate=.0625,proposal_budget=64,candidate_capacity=65536,
                seed=(seed^0xCAFE)+(i+1),
            )
            previous_for_sample=runtime.snapshot() if cloud.will_sample(step+1) else None

    if cloud.n:
        state=runtime.snapshot()
        checkpoints.append(mem.checkpoint_from_cloud(cloud,step,.003))
        runtime=rebuild_runtime(mem,runtime,state)

    runtime.synchronize()
    report={
        "seed":seed,
        "steps":steps,
        "segment_every":segment_every,
        "checkpoint_every":checkpoint_every,
        "positions":[list(x) for x in POSITIONS],
        "segments":segments,
        "checkpoints":checkpoints,
        "final_energy":world.energy,
        "constructed":len(mem.nodes)-mem.grounded_count,
        "total_nethra":len(mem.nodes),
        "wall_s":time.perf_counter()-t0,
        "rss_max_mb":rss_mb(),
        "learner_receipt":{
            "reward":None,
            "energy_input":False,
            "source_label":False,
            "contact_input":False,
            "source_position_schedule_visible_only_as_grounded_pixels":True,
        },
    }
    Path(out).write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",required=True)
    ap.add_argument("--seed",type=int,default=26001)
    ap.add_argument("--steps",type=int,default=60000)
    ap.add_argument("--segment-every",type=int,default=6000)
    ap.add_argument("--checkpoint-every",type=int,required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    run(a.base,seed=a.seed,steps=a.steps,segment_every=a.segment_every,checkpoint_every=a.checkpoint_every,out=a.out)

if __name__=="__main__":
    main()
