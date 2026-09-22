from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np

from core import NethraMemory
from cuda_field import CudaFieldRuntime
from world import World, MOTOR_COUNT
from run_feeding_cuda import field_motor_currents,toggle_babble,grounded_prime

BASE="../benchmark-output/base_hand_ball_cuda.json"
SOURCE=(1.02,0.06)
SEED=29001
STEPS=12000

mem,step,_=NethraMemory.load(BASE)
runtime=CudaFieldRuntime(mem,execution_epsilon=1e-6,dtype="float32")
world=World(
    balls=True,source=True,energy=True,feed_rate=.0045,
    basal_cost=.00021,motor_cost=.0000275,initial_energy=.60,
    source_x=SOURCE[0],source_y=SOURCE[1],
)
rng=random.Random(SEED^0xBADC0DE)
babble=set()
values=np.zeros((STEPS,MOTOR_COUNT),dtype=np.float64)
bab=np.zeros((STEPS,MOTOR_COUNT),dtype=np.uint8)

for i in range(STEPS):
    currents=field_motor_currents(runtime)
    for m,v in currents.items():
        values[i,m]=v
    obs=world.step(currents)
    toggle_babble(babble,rng)
    for m in babble:
        bab[i,m]=1
    runtime.prime(grounded_prime(obs,babble))
    step+=1
    runtime.step({},step,dt=.1,return_active=False)

thresholds=(0.0,1e-6,1e-4,1e-3,1e-2,.05,.1,.25,.5)
per_motor=[]
for m in range(MOTOR_COUNT):
    v=values[:,m]
    per_motor.append({
        "motor":m,
        "babble_on_fraction":float(bab[:,m].mean()),
        "mean":float(v.mean()),
        "std":float(v.std()),
        "min":float(v.min()),
        "max":float(v.max()),
        "presence":{str(t):float((v>t).mean()) for t in thresholds},
        "quantiles":{str(q):float(np.quantile(v,q)) for q in (0,.01,.1,.25,.5,.75,.9,.99,1)},
    })

flat=values.reshape(-1)
out={
    "steps":STEPS,
    "all_motor_value_quantiles":{str(q):float(np.quantile(flat,q)) for q in (0,.01,.1,.25,.5,.75,.9,.99,1)},
    "all_motor_presence":{str(t):float((flat>t).mean()) for t in thresholds},
    "per_motor":per_motor,
}
Path("../benchmark-output/motor-signal.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print(json.dumps(out,indent=2,sort_keys=True))
