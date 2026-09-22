from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Tuple

MOTOR_COUNT = 12
JOINT_COUNT = 6
RANGES = ((0.0,1.65),(-0.9,0.9),(0.0,1.45),(0.0,1.45),(0.0,1.25),(0.0,1.35))
REST = (.75,0.0,.10,.10,.20,.10)
X_MIN,X_MAX=.28,1.24
Y_MIN,Y_MAX=-.24,.72
GRID_W,GRID_H=192,160
PIXELS=GRID_W*GRID_H
DYNAMIC_OFFSET=PIXELS
INPUT_COUNT=PIXELS+2*JOINT_COUNT

@dataclass(frozen=True)
class Obs:
    motors: FrozenSet[int]
    inputs: FrozenSet[int]

class World:
    def __init__(self):
        self.angle=list(REST)
        self.vel=[0.0]*JOINT_COUNT

    def _joint_step(self,j,pos,neg):
        a=self.angle[j];v=self.vel[j];lo,hi=RANGES[j]
        accel=2.1*(float(pos)-float(neg))-1.9*v-.35*a
        v+=.10*accel;a+=.10*v
        if a<lo:a=lo;v=0.0
        elif a>hi:a=hi;v=0.0
        self.angle[j]=a;self.vel[j]=v

    def geometry(self):
        elbow_a,wrist_a,index_a,middle_a,thumb_o,thumb_f=self.angle
        def rot(t,x,y=0.0):
            c=math.cos(t);s=math.sin(t);return c*x-s*y,s*x+c*y
        def add(a,b):return a[0]+b[0],a[1]+b[1]
        elbow=(.55,0.0);wrist=add(elbow,rot(elbow_a,.42));o=elbow_a+wrist_a
        palm=add(wrist,rot(o,.06));ib=add(wrist,rot(o,.12,.025));mb=add(wrist,rot(o,.12,0));tb=add(wrist,rot(o,.036,-.025))
        index=add(ib,rot(o-.95*index_a,.10));middle=add(mb,rot(o-.95*middle_a,.11));thumb=add(tb,rot(o-.90+thumb_o-.55*thumb_f,.08))
        return palm,index,middle,thumb

    @staticmethod
    def _pixel(x,y):
        ix=max(0,min(GRID_W-1,int(round((x-X_MIN)/(X_MAX-X_MIN)*(GRID_W-1)))))
        iy=max(0,min(GRID_H-1,int(round((y-Y_MIN)/(Y_MAX-Y_MIN)*(GRID_H-1)))))
        return iy*GRID_W+ix

    def step(self,motors:Iterable[int])->Obs:
        motors=frozenset(map(int,motors))
        for j in range(JOINT_COUNT):self._joint_step(j,2*j in motors,2*j+1 in motors)
        inputs={self._pixel(x,y) for x,y in self.geometry()}
        for j,v in enumerate(self.vel):
            if v>.020:inputs.add(DYNAMIC_OFFSET+2*j)
            elif v<-.020:inputs.add(DYNAMIC_OFFSET+2*j+1)
        return Obs(motors,frozenset(inputs))

def support(row: Obs) -> FrozenSet[int]:
    return frozenset(row.motors) | frozenset(MOTOR_COUNT+i for i in row.inputs)

class LocalResourceCompetition:
    def __init__(self, sample_rate:float=1.0, sample_seed:int=0):
        if not (0.0 < sample_rate <= 1.0): raise ValueError('sample_rate')
        self.sample_rate=float(sample_rate);self.sampler=random.Random(int(sample_seed))
        self.n=0;self.seen_intervals=0
        self.source_count:Dict[int,int]={};self.target_count:Dict[int,int]={}
        self.joint:Dict[int,Dict[int,int]]={};self.evidence:Dict[int,Dict[int,float]]={}

    def observe(self,sources:Iterable[int],consequences:Iterable[int])->None:
        self.seen_intervals += 1
        if self.sample_rate < 1.0 and self.sampler.random() >= self.sample_rate:
            return
        xs=frozenset(map(int,sources));ys=frozenset(map(int,consequences));self.n+=1
        for x in xs:self.source_count[x]=self.source_count.get(x,0)+1
        for y in ys:self.target_count[y]=self.target_count.get(y,0)+1
        for y in ys:
            joint=self.joint.setdefault(y,{});ev=self.evidence.setdefault(y,{})
            baseline=self.target_count[y]/self.n
            gains=[]
            for x in xs:
                joint[x]=joint.get(x,0)+1
                conditional=joint[x]/self.source_count[x]
                gain=math.log(conditional/baseline) if conditional>0 and baseline>0 else 0.0
                gain=max(0.0,gain);gains.append((x,gain));ev.setdefault(x,0.0)
            total=sum(g for _,g in gains)
            if total>0:
                for x,g in gains:
                    if g>0:ev[x]+=g/total

    def pair_count(self)->int:return sum(len(v) for v in self.joint.values())

@dataclass
class PersistedRelation:
    weight: float
    confirmations: int
    last_checkpoint: int

class PersistentNethraStore:
    def __init__(self, create_floor:float, decay:float, field_epsilon:float):
        self.create_floor=float(create_floor);self.decay=float(decay);self.field_epsilon=float(field_epsilon)
        self.relations:Dict[Tuple[int,int],PersistedRelation]={};self.checkpoint_index=0

    @staticmethod
    def route(x:int,y:int)->Tuple[int,int]:
        a,b=int(x),int(y);return (a,b) if a<b else (b,a)

    def checkpoint(self,c:LocalResourceCompetition)->dict:
        self.checkpoint_index+=1
        for r in self.relations.values():r.weight*=self.decay
        observed:Dict[Tuple[int,int],float]={}
        for y,pool in c.evidence.items():
            for x,ev in pool.items():
                if x==y:continue
                route=self.route(x,y)
                if ev>observed.get(route,0.0):observed[route]=float(ev)
        created=strengthened=0
        for route,ev in observed.items():
            old=self.relations.get(route)
            if old is None:
                if ev>=self.create_floor:
                    self.relations[route]=PersistedRelation(ev,1,self.checkpoint_index);created+=1
            else:
                old.weight+=ev;old.confirmations+=1;old.last_checkpoint=self.checkpoint_index;strengthened+=1
        weights=[r.weight for r in self.relations.values()]
        active=sum(w>=self.field_epsilon for w in weights)
        return {'created':created,'strengthened':strengthened,'relations':len(weights),'active_estimate':active,'dormant_estimate':len(weights)-active,'weight_quantiles':quantiles(weights)}

    def lookup_active_pairs(self,active_support:FrozenSet[int])->int:
        xs=sorted(active_support);hits=0
        for i,a in enumerate(xs):
            for b in xs[i+1:]:
                r=self.relations.get((a,b) if a<b else (b,a))
                if r is not None and r.weight>=self.field_epsilon:hits+=1
        return hits

def quantiles(xs:List[float])->dict:
    if not xs:return {'min':0,'p10':0,'p50':0,'p90':0,'max':0}
    s=sorted(xs)
    def q(p):return s[min(len(s)-1,max(0,int(round(p*(len(s)-1)))))]
    return {'min':s[0],'p10':q(.1),'p50':q(.5),'p90':q(.9),'max':s[-1]}

def motor_routes()->List[Tuple[int,int]]:
    out=[]
    for j in range(JOINT_COUNT):
        for sign in (0,1):
            x=2*j+sign;y=MOTOR_COUNT+DYNAMIC_OFFSET+2*j+sign
            out.append(PersistentNethraStore.route(x,y))
    return out

def run_hand_epoch(*,seed:int,steps:int,sample_rate:float,store:PersistentNethraStore)->dict:
    rng=random.Random(int(seed));w=World();motors=set();prev=None
    c=LocalResourceCompetition(sample_rate=sample_rate,sample_seed=seed^0x5A17)
    lookup_hits=0;lookup_s=0.0;t0=time.perf_counter();active_sum=0
    for _ in range(int(steps)):
        for m in range(MOTOR_COUNT):
            if rng.random()<.02:
                if m in motors:motors.remove(m)
                else:motors.add(m)
        row=w.step(motors);cur=support(row);active_sum+=len(cur)
        if prev is not None:c.observe(prev,cur)
        if store.relations:
            a=time.perf_counter();lookup_hits+=store.lookup_active_pairs(cur);lookup_s+=time.perf_counter()-a
        prev=cur
    elapsed=time.perf_counter()-t0
    cp=store.checkpoint(c)
    after={r:store.relations.get(r).weight if r in store.relations else 0.0 for r in motor_routes()}
    return {'seed':seed,'steps':steps,'sample_rate':sample_rate,'processed_intervals':c.n,'observed_pairs':c.pair_count(),'elapsed_s':elapsed,'us_per_step':1e6*elapsed/steps,'lookup_us_per_step':1e6*lookup_s/steps,'mean_active_support':active_sum/steps,'lookup_hits':lookup_hits,'motor_materialized_after':sum(v>0 for v in after.values()),'motor_weights_after':[after[r] for r in motor_routes()],'checkpoint':cp}

def run_policy(name:str,*,sample_rate:float,create_floor:float,decay:float,field_epsilon:float,epochs:int,steps:int,start_seed:int)->dict:
    store=PersistentNethraStore(create_floor,decay,field_epsilon);rows=[]
    t0=time.perf_counter()
    for e in range(epochs):rows.append(run_hand_epoch(seed=start_seed+e,steps=steps,sample_rate=sample_rate,store=store))
    total=time.perf_counter()-t0
    mweights=[store.relations[r].weight if r in store.relations else 0.0 for r in motor_routes()]
    return {'name':name,'sample_rate':sample_rate,'create_floor':create_floor,'decay':decay,'field_epsilon':field_epsilon,'epochs':epochs,'steps_per_epoch':steps,'total_s':total,'rows':rows,'final_relations':len(store.relations),'final_active_estimate':sum(r.weight>=field_epsilon for r in store.relations.values()),'final_motor_materialized':sum(w>0 for w in mweights),'final_motor_weights':mweights,'final_weight_quantiles':quantiles([r.weight for r in store.relations.values()])}

def synthetic_intuition(*,sample_rate:float,create_floor:float,decay:float,block:int=2000)->dict:
    hist=PersistentNethraStore(create_floor,decay,field_epsilon=1.0)
    fresh=PersistentNethraStore(create_floor,decay,field_epsilon=1.0)
    Y=1000;A=1;B=2;noise=list(range(3,35))
    def one_block(pred:int,seed:int):
        rr=random.Random(seed);c=LocalResourceCompetition(sample_rate=sample_rate,sample_seed=seed^71)
        for _ in range(block):
            xs={x for x in noise if rr.random()<.06}
            if rr.random()<.08:xs.add(pred)
            ys=set()
            if pred in xs and rr.random()<.90:ys.add(Y)
            elif rr.random()<.02:ys.add(Y)
            c.observe(xs,ys)
        return c
    hist.checkpoint(one_block(A,1));fresh.checkpoint(one_block(A,1))
    for i in range(1,6):
        c=one_block(B,10+i);hist.checkpoint(c);fresh.checkpoint(c)
    routeA=hist.route(A,Y);hist_before=hist.relations.get(routeA).weight if routeA in hist.relations else 0.0
    fresh.relations.pop(routeA,None)
    return_blocks=[]
    for i in range(1,9):
        c=one_block(A,100+i);hist.checkpoint(c);fresh.checkpoint(c)
        hw=hist.relations.get(routeA).weight if routeA in hist.relations else 0.0
        fw=fresh.relations.get(routeA).weight if routeA in fresh.relations else 0.0
        return_blocks.append({'block':i,'history_weight':hw,'fresh_weight':fw,'history_exists':routeA in hist.relations,'fresh_exists':routeA in fresh.relations})
    return {'sample_rate':sample_rate,'create_floor':create_floor,'decay':decay,'block':block,'history_weight_before_return':hist_before,'return':return_blocks}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='emergent_resource_decay_results.json');ap.add_argument('--epochs',type=int,default=6);ap.add_argument('--steps',type=int,default=12000);args=ap.parse_args()
    policies=[('strict_full',1.0,500.0,1.0,25.0),('sample50_decay90',.50,80.0,.90,20.0),('sample25_decay85',.25,35.0,.85,15.0),('sample25_permissive_decay70',.25,10.0,.70,10.0),('sample125_permissive_decay80',.125,8.0,.80,8.0)]
    report={'host':{},'policies':[],'intuition':[]}
    import platform,os
    report['host']={'platform':platform.platform(),'python':platform.python_version(),'hostname':platform.node(),'pid':os.getpid()}
    for name,sr,floor,decay,eps in policies:
        print('RUN',name,flush=True)
        report['policies'].append(run_policy(name,sample_rate=sr,create_floor=floor,decay=decay,field_epsilon=eps,epochs=args.epochs,steps=args.steps,start_seed=4100))
        report['intuition'].append({'name':name,'result':synthetic_intuition(sample_rate=sr,create_floor=floor,decay=decay,block=2000)})
    Path(args.out).write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    summary=[]
    for p in report['policies']:
        us=[r['us_per_step'] for r in p['rows']];lookup=[r['lookup_us_per_step'] for r in p['rows']]
        summary.append({'name':p['name'],'final_relations':p['final_relations'],'final_active_estimate':p['final_active_estimate'],'motor':p['final_motor_materialized'],'median_us_per_step':statistics.median(us),'median_lookup_us_per_step':statistics.median(lookup),'weight_p50':p['final_weight_quantiles']['p50'],'weight_p90':p['final_weight_quantiles']['p90']})
    print(json.dumps({'host':report['host'],'summary':summary,'intuition':[{'name':x['name'],'before':x['result']['history_weight_before_return'],'return':[(r['block'],round(r['history_weight'],2),round(r['fresh_weight'],2)) for r in x['result']['return']]} for x in report['intuition']]},indent=2),flush=True)

if __name__=='__main__':main()
