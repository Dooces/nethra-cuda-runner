from __future__ import annotations
import argparse,json,math,random,time,platform
from dataclasses import dataclass
from pathlib import Path
from typing import Dict,FrozenSet,Tuple

MOTOR_COUNT=12;JOINT_COUNT=6
RANGES=((0.,1.65),(-.9,.9),(0.,1.45),(0.,1.45),(0.,1.25),(0.,1.35));REST=(.75,0.,.10,.10,.20,.10)
X_MIN,X_MAX=.28,1.24;Y_MIN,Y_MAX=-.24,.72;GRID_W,GRID_H=192,160;PIXELS=GRID_W*GRID_H;DYNAMIC_OFFSET=PIXELS
MASK=(1<<64)-1

@dataclass(frozen=True)
class Obs: motors:FrozenSet[int]; inputs:FrozenSet[int]
class World:
    def __init__(self):self.angle=list(REST);self.vel=[0.]*JOINT_COUNT
    def _joint(self,j,p,n):
        a=self.angle[j];v=self.vel[j];lo,hi=RANGES[j];acc=2.1*(float(p)-float(n))-1.9*v-.35*a;v+=.1*acc;a+=.1*v
        if a<lo:a=lo;v=0.
        elif a>hi:a=hi;v=0.
        self.angle[j]=a;self.vel[j]=v
    def geometry(self):
        ea,wa,ia,ma,to,tf=self.angle
        def rot(t,x,y=0.):c=math.cos(t);s=math.sin(t);return c*x-s*y,s*x+c*y
        def add(a,b):return a[0]+b[0],a[1]+b[1]
        e=(.55,0.);w=add(e,rot(ea,.42));o=ea+wa;p=add(w,rot(o,.06));ib=add(w,rot(o,.12,.025));mb=add(w,rot(o,.12,0));tb=add(w,rot(o,.036,-.025))
        return p,add(ib,rot(o-.95*ia,.10)),add(mb,rot(o-.95*ma,.11)),add(tb,rot(o-.90+to-.55*tf,.08))
    def pixel(self,x,y):
        ix=max(0,min(GRID_W-1,int(round((x-X_MIN)/(X_MAX-X_MIN)*(GRID_W-1)))));iy=max(0,min(GRID_H-1,int(round((y-Y_MIN)/(Y_MAX-Y_MIN)*(GRID_H-1)))));return iy*GRID_W+ix
    def step(self,motors):
        motors=frozenset(motors)
        for j in range(JOINT_COUNT):self._joint(j,2*j in motors,2*j+1 in motors)
        inp={self.pixel(x,y) for x,y in self.geometry()}
        for j,v in enumerate(self.vel):
            if v>.020:inp.add(DYNAMIC_OFFSET+2*j)
            elif v<-.020:inp.add(DYNAMIC_OFFSET+2*j+1)
        return Obs(motors,frozenset(inp))
def support(r):return frozenset(r.motors)|frozenset(MOTOR_COUNT+i for i in r.inputs)

def u01(token:int,seed:int)->float:
    z=(int(token)+0x9E3779B97F4A7C15*(int(seed)+1))&MASK;z=((z^(z>>30))*0xBF58476D1CE4E5B9)&MASK;z=((z^(z>>27))*0x94D049BB133111EB)&MASK;z^=z>>31
    return ((z>>11)&((1<<53)-1))/float(1<<53)

class Competition:
    def __init__(self,rate=.25,seed=77):self.rate=float(rate);self.seed=int(seed);self.n=0;self.seen=0;self.sc={};self.tc={};self.j={};self.ev={}
    def observe(self,xs,ys,token):
        self.seen+=1
        if u01(token,self.seed)>=self.rate:return
        xs=frozenset(xs);ys=frozenset(ys);self.n+=1
        for x in xs:self.sc[x]=self.sc.get(x,0)+1
        for y in ys:self.tc[y]=self.tc.get(y,0)+1
        for y in ys:
            jj=self.j.setdefault(y,{});ee=self.ev.setdefault(y,{});b=self.tc[y]/self.n;g=[]
            for x in xs:
                jj[x]=jj.get(x,0)+1;c=jj[x]/self.sc[x];gg=max(0.,math.log(c/b)) if c>0 and b>0 else 0.;g.append((x,gg));ee.setdefault(x,0.)
            tot=sum(v for _,v in g)
            if tot>0:
                for x,v in g:
                    if v>0:ee[x]+=v/tot

@dataclass
class Rel: weight:float; confirms:int; last_step:int
class Store:
    def __init__(self,rate_floor=.8,half_life=60000.,field_tau=100.):self.rate_floor=float(rate_floor);self.half=float(half_life);self.tau=float(field_tau);self.r:Dict[Tuple[int,int],Rel]={};self.last_step=0
    @staticmethod
    def route(x,y):return (x,y) if x<y else (y,x)
    def checkpoint(self,c:Competition,global_step:int):
        elapsed=max(0,int(global_step)-self.last_step);self.last_step=int(global_step);factor=2.**(-elapsed/self.half) if self.half>0 else 0.
        for rr in self.r.values():rr.weight*=factor
        obs={}
        for y,p in c.ev.items():
            for x,ev in p.items():
                if x==y:continue
                k=self.route(int(x),int(y));obs[k]=max(obs.get(k,0.),float(ev))
        new=strong=0
        processed=max(1,c.n)
        for k,ev in obs.items():
            density=1000.*ev/processed
            scaled_ev=ev/max(c.rate,1e-12)
            rr=self.r.get(k)
            if rr is None:
                if density>=self.rate_floor:self.r[k]=Rel(scaled_ev,1,global_step);new+=1
            else:rr.weight+=scaled_ev;rr.confirms+=1;rr.last_step=global_step;strong+=1
        return {'created':new,'strengthened':strong,'relations':len(self.r),'weight_q':q([v.weight for v in self.r.values()]),'conductance':field_stats(self.r,self.tau)}
    def lookup(self,s):
        xs=sorted(s);hits=0
        for i,a in enumerate(xs):
            for b in xs[i+1:]:
                if self.route(a,b) in self.r:hits+=1
        return hits

def q(v):
    if not v:return {'p10':0,'p50':0,'p90':0,'max':0}
    a=sorted(v)
    def z(p):return a[min(len(a)-1,int(round(p*(len(a)-1))))]
    return {'p10':z(.1),'p50':z(.5),'p90':z(.9),'max':a[-1]}
def field_stats(r,tau):
    ws=[v.weight for v in r.values()]
    gs=[1.5*(1-math.exp(-max(0.,w)/tau)) for w in ws]
    return {'current_gmin02_edges':len(ws)*2,'zero_floor_g_gt_001':2*sum(g>.01 for g in gs),'zero_floor_g_gt_005':2*sum(g>.05 for g in gs),'zero_floor_g_gt_010':2*sum(g>.10 for g in gs),'zero_floor_mass':2*sum(gs),'current_floor_min_mass':.4*len(ws)}
def motor_routes():
    return [Store.route(2*j+s,MOTOR_COUNT+DYNAMIC_OFFSET+2*j+s) for j in range(JOINT_COUNT) for s in (0,1)]

def lineage(check_every,total_steps,rate_floor,half_life,sample_rate=.25,seed=5501):
    w=World();rng=random.Random(seed);mot=set();prev=None;store=Store(rate_floor,half_life);comp=Competition(sample_rate,seed^123);checks=[];lookup_s=0.;t0=time.perf_counter()
    for step in range(1,total_steps+1):
        for m in range(MOTOR_COUNT):
            if rng.random()<.02:
                if m in mot:mot.remove(m)
                else:mot.add(m)
        cur=support(w.step(mot))
        if prev is not None:comp.observe(prev,cur,step)
        if store.r:
            a=time.perf_counter();store.lookup(cur);lookup_s+=time.perf_counter()-a
        prev=cur
        if step%check_every==0 or step==total_steps:
            cp=store.checkpoint(comp,step);checks.append({'step':step,'processed':comp.n,**cp});comp=Competition(sample_rate,seed^123)
    elapsed=time.perf_counter()-t0;mr=motor_routes()
    keys=set(store.r);mw=[store.r[k].weight if k in store.r else 0. for k in mr]
    return {'check_every':check_every,'total_steps':total_steps,'sample_rate':sample_rate,'rate_floor':rate_floor,'half_life':half_life,'elapsed_s':elapsed,'us_per_step':1e6*elapsed/total_steps,'lookup_us_per_step':1e6*lookup_s/total_steps,'relations':len(keys),'motor':sum(x>0 for x in mw),'motor_weights':mw,'keys':[f'{a}:{b}' for a,b in sorted(keys)],'checks':checks,'final_field':field_stats(store.r,store.tau)}

def compare(a,b):
    A=set(a['keys']);B=set(b['keys']);return {'jaccard':len(A&B)/max(1,len(A|B)),'intersection':len(A&B),'union':len(A|B),'count_a':len(A),'count_b':len(B)}

def intuition(rate_floor=.8,half_life=60000.,sample_rate=.25):
    A=1;B=2;Y=1000;noise=list(range(3,40));hist=Store(rate_floor,half_life);fresh=Store(rate_floor,half_life);step=0
    def block(pred,n,seed):
        nonlocal step
        c=Competition(sample_rate,999);rr=random.Random(seed)
        for _ in range(n):
            step+=1;xs={x for x in noise if rr.random()<.06}
            if rr.random()<.08:xs.add(pred)
            ys=set()
            if pred in xs and rr.random()<.9:ys.add(Y)
            elif rr.random()<.02:ys.add(Y)
            c.observe(xs,ys,step)
        return c
    c=block(A,5000,1);hist.checkpoint(c,step);fresh.checkpoint(c,step)
    for i in range(6):c=block(B,5000,10+i);hist.checkpoint(c,step);fresh.checkpoint(c,step)
    route=Store.route(A,Y);before=hist.r.get(route).weight if route in hist.r else 0.;fresh.r.pop(route,None)
    ret=[]
    for i in range(8):
        c=block(A,500,100+i);hist.checkpoint(c,step);fresh.checkpoint(c,step);ret.append({'i':i+1,'hist':hist.r.get(route).weight if route in hist.r else 0.,'fresh':fresh.r.get(route).weight if route in fresh.r else 0.})
    return {'before':before,'return':ret}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='cadence_normalized_decay.json');ap.add_argument('--total',type=int,default=120000);args=ap.parse_args()
    schedules=[]
    for rate_floor in (.8,.25):
        for ce in (7500,15000,30000):
            print('RUN',rate_floor,ce,flush=True);schedules.append(lineage(ce,args.total,rate_floor,60000.,.25,5501))
    comps=[]
    for rf in (.8,.25):
        rows=[x for x in schedules if x['rate_floor']==rf]
        comps.append({'rate_floor':rf,'7500_vs_15000':compare(rows[0],rows[1]),'15000_vs_30000':compare(rows[1],rows[2])})
    report={'host':{'platform':platform.platform(),'python':platform.python_version(),'hostname':platform.node()},'schedules':schedules,'comparisons':comps,'intuition':{'rf08':intuition(.8,60000.,.25),'rf025':intuition(.25,60000.,.25)}}
    Path(args.out).write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'host':report['host'],'summary':[{'rf':x['rate_floor'],'cadence':x['check_every'],'relations':x['relations'],'motor':x['motor'],'us':round(x['us_per_step'],2),'lookup_us':round(x['lookup_us_per_step'],2),'field':x['final_field']} for x in schedules],'comparisons':comps,'intuition':report['intuition']},indent=2),flush=True)
if __name__=='__main__':main()
