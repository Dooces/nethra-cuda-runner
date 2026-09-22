from __future__ import annotations
import json, random
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

from core import NethraMemory
from cuda_field import CudaFieldRuntime
from world import World, MOTOR_COUNT
from run_feeding_cuda import make_cloud,field_motor_currents,toggle_babble,grounded_prime,observe_if_sampled

BASE="../benchmark-output/base_hand_ball_cuda.json"
SOURCE=(1.02,0.06)
SEED=28001
STEPS=12000
FLOOR=.003

mem,base_step,_=NethraMemory.load(BASE)
runtime=CudaFieldRuntime(mem,execution_epsilon=1e-6,dtype="float32")
world=World(balls=True,source=True,energy=True,feed_rate=.0045,basal_cost=.00021,motor_cost=.0000275,initial_energy=.60,source_x=SOURCE[0],source_y=SOURCE[1])
rng=random.Random(SEED^0xBADC0DE)
babble=set()
cloud=make_cloud(sample_rate=.0625,proposal_budget=64,candidate_capacity=65536,seed=SEED^0xCAFE)
previous=None
step=base_step
for i in range(STEPS):
    step+=1
    motors=field_motor_currents(runtime)
    obs=world.step(motors)
    toggle_babble(babble,rng)
    runtime.prime(grounded_prime(obs,babble))
    runtime.step({},step,dt=.1,return_active=False)
    previous=observe_if_sampled(cloud,runtime,step,previous)

source_nid=MOTOR_COUNT+World._pixel(*SOURCE)

def role(nid):
    if nid<MOTOR_COUNT:return "motor"
    if nid<mem.grounded_count:return "sensory"
    return "constructed"

def ptype(a,b):
    return "+".join(sorted((role(a),role(b))))

cand=Counter(); positive=Counter(); pass_floor=Counter()
motor_ev=[]; source_ev=[]
for c in cloud.candidates.values():
    t=ptype(c.source,c.consequence)
    cand[t]+=1
    if c.evidence>0:
        positive[t]+=1
    density=1000.0*c.evidence/max(1,cloud.n)
    if density>=FLOOR:
        pass_floor[t]+=1
    if c.source<MOTOR_COUNT or c.consequence<MOTOR_COUNT:
        motor_ev.append((c.evidence,density))
    if c.source==source_nid or c.consequence==source_nid:
        source_ev.append((c.evidence,density))

best={}
for c in cloud.candidates.values():
    if c.source==c.consequence:continue
    key=mem.member_key(c.source,c.consequence)
    if c.evidence>best.get(key,(-1,None))[0]:
        best[key]=(c.evidence,c)
selected=[]
for key,(ev,c) in best.items():
    density=1000.0*ev/max(1,cloud.n)
    if key not in mem.by_members and density>=FLOOR:
        selected.append((key,ev,density,c))
selected_types=Counter(ptype(*key) for key,_,_,_ in selected)

def quant(v):
    if not v:return {}
    a=np.asarray(v,float)
    return {str(p):float(np.quantile(a,p)) for p in (0,.5,.9,.99,1)}

out={
  "sampled_intervals":cloud.n,
  "candidate_count":len(cloud.candidates),
  "candidate_pair_types":dict(cand),
  "positive_evidence_pair_types":dict(positive),
  "above_floor_pair_types":dict(pass_floor),
  "selected_new_pair_types":dict(selected_types),
  "selected_new_count":len(selected),
  "selected_motor_count":sum(1 for key,_,_,_ in selected if key[0]<MOTOR_COUNT or key[1]<MOTOR_COUNT),
  "selected_source_pixel_count":sum(1 for key,_,_,_ in selected if source_nid in key),
  "motor_candidate_count":len(motor_ev),
  "motor_positive_count":sum(1 for e,d in motor_ev if e>0),
  "motor_above_floor_count":sum(1 for e,d in motor_ev if d>=FLOOR),
  "motor_evidence_quantiles_positive":quant([e for e,d in motor_ev if e>0]),
  "motor_density_quantiles_positive":quant([d for e,d in motor_ev if e>0]),
  "source_pixel_candidate_count":len(source_ev),
  "source_pixel_positive_count":sum(1 for e,d in source_ev if e>0),
  "source_pixel_above_floor_count":sum(1 for e,d in source_ev if d>=FLOOR),
  "source_pixel_evidence_quantiles_positive":quant([e for e,d in source_ev if e>0]),
}
Path("../benchmark-output/candidate-allocation.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print(json.dumps(out,indent=2,sort_keys=True))
