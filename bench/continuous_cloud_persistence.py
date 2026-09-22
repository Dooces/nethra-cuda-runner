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
    def __init__(self,density_floor=.25,half_life=60000.,tau=100.,field_epsilon=.05):
        self.floor=float(density_floor);self.half=float(half_life);self.tau=float(tau);self.eps=float(field_epsilon);self.r:Dict[Tuple[int,int],Rel]={};self.last_step=0;self.last_cloud_ev={}
    @staticmethod
    def route(x,y):return (x,y) if x<y else (y,x)
    def cloud_routes(self,c:Competition):
        obs={}
        for y,p in c.ev.items():
            for x,ev in p.items():
                if x==y:continue
                k=self.route(int(x),int(y));obs[k]=max(obs.get(k,0.),float(ev))
        return obs
    def conductance(self,w):return 1.5*(1-math.exp(-max(0.,w)/self.tau))
    def active(self,rr):return self.conductance(rr.weight)>=self.eps
    def checkpoint(self,c:Competition,step:int):
        elapsed=max(0,int(step)-self.last_step);self.last_step=int(step);factor=2.**(-elapsed/self.half) if self.half>0 else 0.
        for rr in self.r.values():rr.weight*=factor
        obs=self.cloud_routes(c);created=strengthened=0;processed=max(1,c.n)
        for k,total_ev in obs.items():
            density=1000.*total_ev/processed
            rr=self.r.get(k)
            if rr is None:
                if density>=self.floor:
                    rr=Rel(total_ev/max(c.rate,1e-12),1,step);self.r[k]=rr;created+=1
            else:
                prev=self.last_cloud_ev.get(k,0.);delta=max(0.,total_ev-prev)
                if delta>0:rr.weight+=delta/max(c.rate,1e-12);rr.confirms+=1;rr.last_step=step;strengthened+=1
        self.last_cloud_ev=obs
        active_keys={k for k,v in self.r.items() if self.active(v)}
        return {'created':created,'strengthened':strengthened,'relations':len(self.r),'active':len(active_keys),'dormant':len(self.r)-len(active_keys),'active_keys':[f'{a}:{b}' for a,b in sorted(active_keys)]}
    def lookup(self,s):
        xs=sorted(s);hits=0
        for i,a in enumerate(xs):
            for b in xs[i+1:]:
                rr=self.r.get(self.route(a,b))
                if rr is not None and self.active(rr):hits+=1
        return hits

def motor_routes():return [Store.route(2*j+s,MOTOR_COUNT+DYNAMIC_OFFSET+2*j+s) for j in range(JOINT_COUNT) for s in (0,1)]
def compare(A,B):
    a=set(A);b=set(B);return {'jaccard':len(a&b)/max(1,len(a|b)),'intersection':len(a&b),'union':len(a|b),'a':len(a),'b':len(b)}

def lineage(*,check_every,total_steps,density_floor,sample_rate=.25,half_life=60000.,seed=6601):
    w=World();rng=random.Random(seed);mot=set();prev=None;cloud=Competition(sample_rate,seed^321);store=Store(density_floor,half_life);rows=[];lookup_s=0.;t0=time.perf_counter()
    for step in range(1,total_steps+1):
        for m in range(MOTOR_COUNT):
            if rng.random()<.02:
                if m in mot:mot.remove(m)
                else:mot.add(m)
        cur=support(w.step(mot))
        if prev is not None:cloud.observe(prev,cur,step)
        if store.r:
            a=time.perf_counter();store.lookup(cur);lookup_s+=time.perf_counter()-a
        prev=cur
        if step%check_every==0 or step==total_steps:rows.append({'step':step,**store.checkpoint(cloud,step)})
    elapsed=time.perf_counter()-t0;active=rows[-1]['active_keys'];mr=motor_routes();active_set={Store.route(*map(int,x.split(':'))) for x in active}
    return {'cadence':check_every,'sample_rate':sample_rate,'floor':density_floor,'relations':len(store.r),'active':len(active),'motor_saved':sum(k in store.r for k in mr),'motor_active':sum(k in active_set for k in mr),'us_per_step':1e6*elapsed/total_steps,'lookup_us_per_step':1e6*lookup_s/total_steps,'active_keys':active,'rows':rows}

def intuition(density_floor=.8,sample_rate=.25,half_life=15000.,field_epsilon=.05):
    A=1;B=2;Y=1000;noise=list(range(3,40));hist=Store(density_floor,half_life,100.,field_epsilon);fresh=Store(density_floor,half_life,100.,field_epsilon);cloud_h=Competition(sample_rate,991);cloud_f=Competition(sample_rate,991);step=0
    def feed(pred,n,seed):
        nonlocal step
        rr=random.Random(seed)
        for _ in range(n):
            step+=1;xs={x for x in noise if rr.random()<.06}
            if rr.random()<.08:xs.add(pred)
            ys=set()
            if pred in xs and rr.random()<.90:ys.add(Y)
            elif rr.random()<.02:ys.add(Y)
            cloud_h.observe(xs,ys,step);cloud_f.observe(xs,ys,step)
    feed(A,5000,1);hist.checkpoint(cloud_h,step);fresh.checkpoint(cloud_f,step)
    for i in range(8):feed(B,5000,10+i);hist.checkpoint(cloud_h,step);fresh.checkpoint(cloud_f,step)
    route=Store.route(A,Y);fresh.r.pop(route,None);fresh.last_cloud_ev.pop(route,None)
    before={'weight':hist.r.get(route).weight if route in hist.r else 0.,'active':route in hist.r and hist.active(hist.r[route])}
    ret=[]
    for i in range(8):
        feed(A,500,100+i);hist.checkpoint(cloud_h,step);fresh.checkpoint(cloud_f,step)
        hr=hist.r.get(route);fr=fresh.r.get(route)
        ret.append({'block':i+1,'hist_weight':hr.weight if hr else 0.,'hist_active':bool(hr and hist.active(hr)),'fresh_weight':fr.weight if fr else 0.,'fresh_active':bool(fr and fresh.active(fr))})
    return {'before_return':before,'return':ret}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--total',type=int,default=120000);ap.add_argument('--out',default='continuous_cloud_persistence.json');args=ap.parse_args()
    runs=[]
    for floor in (.8,.25):
        for cadence in (7500,15000,30000):
            print('RUN cadence',floor,cadence,flush=True);runs.append(lineage(check_every=cadence,total_steps=args.total,density_floor=floor,sample_rate=.25))
    throttles=[]
    for sr in (.25,.125,.0625):
        print('RUN throttle',sr,flush=True);throttles.append(lineage(check_every=15000,total_steps=args.total,density_floor=.25,sample_rate=sr,seed=7701))
    comps=[]
    for floor in (.8,.25):
        x=[r for r in runs if r['floor']==floor];comps.append({'floor':floor,'active_7500_15000':compare(x[0]['active_keys'],x[1]['active_keys']),'active_15000_30000':compare(x[1]['active_keys'],x[2]['active_keys'])})
    report={'host':{'hostname':platform.node(),'platform':platform.platform(),'python':platform.python_version()},'cadence_runs':runs,'comparisons':comps,'throttle_runs':throttles,'intuition':intuition()}
    Path(args.out).write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'host':report['host'],'cadence':[{'floor':r['floor'],'cadence':r['cadence'],'saved':r['relations'],'active':r['active'],'motor_saved':r['motor_saved'],'motor_active':r['motor_active'],'us':round(r['us_per_step'],2),'lookup_us':round(r['lookup_us_per_step'],2)} for r in runs],'comparisons':comps,'throttle':[{'sr':r['sample_rate'],'saved':r['relations'],'active':r['active'],'motor_saved':r['motor_saved'],'motor_active':r['motor_active'],'us':round(r['us_per_step'],2),'lookup_us':round(r['lookup_us_per_step'],2)} for r in throttles],'intuition':report['intuition']},indent=2),flush=True)
if __name__=='__main__':main()
