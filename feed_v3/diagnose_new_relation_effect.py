from __future__ import annotations
import copy, json, math, random
from pathlib import Path
import numpy as np

from core import NethraMemory
from cuda_field import CudaFieldRuntime
from world import World, MOTOR_COUNT
from run_feeding_cuda import (
    make_cloud, field_motor_currents, toggle_babble, grounded_prime,
    observe_if_sampled,
)

BASE="../benchmark-output/base_hand_ball_cuda.json"
SOURCE=(1.02,0.06)
PRE=12000
POST=6000
SEED=27001

def q(v):
    if not v: return {}
    a=np.asarray(v,dtype=float)
    return {str(p):float(np.quantile(a,p)) for p in (0,.1,.25,.5,.75,.9,.99,1)}

mem,base_step,_=NethraMemory.load(BASE)
runtime=CudaFieldRuntime(mem,execution_epsilon=1e-6,dtype="float32")
world=World(
    balls=True,source=True,energy=True,feed_rate=.0045,
    basal_cost=.00021,motor_cost=.0000275,initial_energy=.60,
    source_x=SOURCE[0],source_y=SOURCE[1],
)
rng=random.Random(SEED^0xBADC0DE)
babble=set()
cloud=make_cloud(sample_rate=.0625,proposal_budget=64,candidate_capacity=65536,seed=SEED^0xCAFE)
previous=None
step=base_step

for i in range(PRE):
    step+=1
    motors=field_motor_currents(runtime)
    obs=world.step(motors)
    toggle_babble(babble,rng)
    runtime.prime(grounded_prime(obs,babble))
    runtime.step({},step,dt=.1,return_active=False)
    previous=observe_if_sampled(cloud,runtime,step,previous)

state=runtime.snapshot()
world_at_fork=copy.deepcopy(world)
rng_at_fork=copy.deepcopy(rng)
babble_at_fork=set(babble)
frozen_mem=copy.deepcopy(mem)
before_ids=set(mem.nodes)
receipt=mem.checkpoint_from_cloud(cloud,step,.003)
new=[n for nid,n in mem.nodes.items() if nid not in before_ids]

new_g=[mem.conductance(n,step) for n in new]
new_w=[n.weight for n in new]
old_rel=[n for n in frozen_mem.nodes.values() if n.members]
old_g=[frozen_mem.conductance(n,step) for n in old_rel]

cats={"motor_member":0,"sensory_member":0,"constructed_member":0,"touches_source_pixel":0}
source_nid=MOTOR_COUNT+World._pixel(*SOURCE)
for n in new:
    for m in n.members:
        if m<MOTOR_COUNT: cats["motor_member"]+=1
        elif m<frozen_mem.grounded_count: cats["sensory_member"]+=1
        else: cats["constructed_member"]+=1
        if m==source_nid: cats["touches_source_pixel"]+=1

learned=CudaFieldRuntime(mem,execution_epsilon=1e-6,dtype="float32")
frozen=CudaFieldRuntime(frozen_mem,execution_epsilon=1e-6,dtype="float32")
learned.set_state(state); frozen.set_state(state)
wl=copy.deepcopy(world_at_fork); wf=copy.deepcopy(world_at_fork)
rng2=copy.deepcopy(rng_at_fork); babble2=set(babble_at_fork)

sum_abs=0.0; max_abs=0.0; first_gt=None; contact_mismatch=0; max_energy_delta=0.0
samples=[]
for i in range(POST):
    ml=field_motor_currents(learned)
    mf=field_motor_currents(frozen)
    delta=np.abs(ml-mf)
    md=float(delta.max()) if delta.size else 0.0
    max_abs=max(max_abs,md); sum_abs+=float(delta.sum())
    if first_gt is None and md>1e-9: first_gt=i+1
    ol=wl.step(ml); of=wf.step(mf)
    contact_mismatch+=int(ol.source_contact!=of.source_contact)
    max_energy_delta=max(max_energy_delta,abs(ol.energy-of.energy))
    toggle_babble(babble2,rng2)
    learned.prime(grounded_prime(ol,babble2))
    frozen.prime(grounded_prime(of,babble2))
    learned.step({},step+i+1,dt=.1,return_active=False)
    frozen.step({},step+i+1,dt=.1,return_active=False)
    if (i+1) in (1,10,100,1000,3000,6000):
        samples.append({
            "post_step":i+1,
            "max_motor_abs_delta":md,
            "energy_delta":ol.energy-of.energy,
            "learned_contact":ol.source_contact,
            "frozen_contact":of.source_contact,
        })

out={
    "fork_step":step,
    "receipt":receipt,
    "new_relation_count":len(new),
    "new_weight_quantiles":q(new_w),
    "new_conductance_quantiles":q(new_g),
    "old_conductance_quantiles":q(old_g),
    "new_member_category_touches":cats,
    "post_steps":POST,
    "first_motor_delta_gt_1e-9":first_gt,
    "max_motor_abs_delta":max_abs,
    "mean_abs_motor_delta_per_motor_per_step":sum_abs/(POST*MOTOR_COUNT),
    "contact_mismatch_steps":contact_mismatch,
    "max_energy_delta":max_energy_delta,
    "samples":samples,
}
Path("../benchmark-output/new-relation-effect.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print(json.dumps(out,indent=2,sort_keys=True))
