from __future__ import annotations
import argparse,json,math,random,time,platform
from dataclasses import dataclass
from pathlib import Path
from typing import Dict,FrozenSet,Tuple
MOTOR_COUNT=12;JOINT_COUNT=6
RANGES=((0.,1.65),(-.9,.9),(0.,1.45),(0.,1.45),(0.,1.25),(0.,1.35));REST=(.75,0.,.10,.10,.20,.10)
X_MIN,X_MAX=.28,1.24;Y_MIN,Y_MAX=-.24,.72;W,H=192,160;PIX=W*H;DYN=PIX;MASK=(1<<64)-1
@dataclass(frozen=True)
class Obs: motors:FrozenSet[int]; inputs:FrozenSet[int]
class World:
    def __init__(self):self.a=list(REST);self.v=[0.]*6
    def _j(self,j,p,n):
        a=self.a[j];v=self.v[j];lo,hi=RANGES[j];v+=.1*(2.1*(float(p)-float(n))-1.9*v-.35*a);a+=.1*v
        if a<lo:a=lo;v=0.
        elif a>hi:a=hi;v=0.
        self.a[j]=a;self.v[j]=v
    def geom(self):
        ea,wa,ia,ma,to,tf=self.a
        def r(t,x,y=0.):c=math.cos(t);s=math.sin(t);return c*x-s*y,s*x+c*y
        def q(a,b):return a[0]+b[0],a[1]+b[1]
        e=(.55,0.);w=q(e,r(ea,.42));o=ea+wa;p=q(w,r(o,.06));ib=q(w,r(o,.12,.025));mb=q(w,r(o,.12,0));tb=q(w,r(o,.036,-.025))
        return p,q(ib,r(o-.95*ia,.10)),q(mb,r(o-.95*ma,.11)),q(tb,r(o-.90+to-.55*tf,.08))
    def pix(self,x,y):
        ix=max(0,min(W-1,int(round((x-X_MIN)/(X_MAX-X_MIN)*(W-1)))));iy=max(0,min(H-1,int(round((y-Y_MIN)/(Y_MAX-Y_MIN)*(H-1)))));return iy*W+ix
    def step(self,m):
        m=frozenset(m)
        for j in range(6):self._j(j,2*j in m,2*j+1 in m)
        x={self.pix(a,b) for a,b in self.geom()}
        for j,v in enumerate(self.v):
            if v>.020:x.add(DYN+2*j)
            elif v<-.020:x.add(DYN+2*j+1)
        return Obs(m,frozenset(x))
def sup(o):return frozenset(o.motors)|frozenset(12+i for i in o.inputs)
def u01(t,s):
    z=(int(t)+0x9E3779B97F4A7C15*(int(s)+1))&MASK;z=((z^(z>>30))*0xBF58476D1CE4E5B9)&MASK;z=((z^(z>>27))*0x94D049BB133111EB)&MASK;z^=z>>31;return ((z>>11)&((1<<53)-1))/float(1<<53)
class Cloud:
    def __init__(self,sr=.25,seed=1):self.sr=sr;self.seed=seed;self.n=0;self.sc={};self.tc={};self.j={};self.ev={}
    def obs(self,xs,ys,t):
        if u01(t,self.seed)>=self.sr:return
        xs=frozenset(xs);ys=frozenset(ys);self.n+=1
        for x in xs:self.sc[x]=self.sc.get(x,0)+1
        for y in ys:self.tc[y]=self.tc.get(y,0)+1
        for y in ys:
            jj=self.j.setdefault(y,{});ee=self.ev.setdefault(y,{});b=self.tc[y]/self.n;gg=[]
            for x in xs:
                jj[x]=jj.get(x,0)+1;c=jj[x]/self.sc[x];g=max(0.,math.log(c/b)) if c>0 and b>0 else 0.;gg.append((x,g));ee.setdefault(x,0.)
            z=sum(g for _,g in gg)
            if z>0:
                for x,g in gg:
                    if g>0:ee[x]+=g/z
@dataclass
class Rel: w:float; confirms:int
class Store:
    def __init__(self,floor=1.,half=30000.,tau=100.):self.floor=floor;self.half=half;self.tau=tau;self.r:Dict[Tuple[int,int],Rel]={}
    @staticmethod
    def key(x,y):return (x,y) if x<y else (y,x)
    def g(self,w):return 1.5*(1-math.exp(-max(0.,w)/self.tau))
    def cp(self,c:Cloud,elapsed:int):
        f=2**(-elapsed/self.half) if self.half>0 else 0.
        for r in self.r.values():r.w*=f
        seen={}
        for y,p in c.ev.items():
            for x,e in p.items():
                if x==y:continue
                k=self.key(x,y);seen[k]=max(seen.get(k,0.),e)
        new=rein=0
        for k,e in seen.items():
            r=self.r.get(k)
            if r is None:
                if e>=self.floor:self.r[k]=Rel(e/max(c.sr,1e-12),1);new+=1
            else:r.w+=e/max(c.sr,1e-12);r.confirms+=1;rein+=1
        return {'created':new,'reinforced':rein,'saved':len(self.r),'g01':sum(self.g(r.w)>=.01 for r in self.r.values()),'g05':sum(self.g(r.w)>=.05 for r in self.r.values()),'g10':sum(self.g(r.w)>=.10 for r in self.r.values())}
    def lookup_all(self,s):
        xs=sorted(s);return sum(self.key(a,b) in self.r for i,a in enumerate(xs) for b in xs[i+1:])
def motors():return [Store.key(2*j+s,12+DYN+2*j+s) for j in range(6) for s in (0,1)]
def run(name,floor,half,sr=.25,windows=8,budget=15000,seed=8801):
    st=Store(floor,half);w=World();rng=random.Random(seed);m=set();prev=None;rows=[];lookup=0.;t0=time.perf_counter();step=0
    for win in range(windows):
        c=Cloud(sr,seed^555)
        for _ in range(budget):
            step+=1
            for x in range(12):
                if rng.random()<.02:
                    if x in m:m.remove(x)
                    else:m.add(x)
            cur=sup(w.step(m))
            if prev is not None:c.obs(prev,cur,step)
            if st.r:
                a=time.perf_counter();st.lookup_all(cur);lookup+=time.perf_counter()-a
            prev=cur
        rows.append({'window':win+1,**st.cp(c,budget)})
    elapsed=time.perf_counter()-t0;mr=motors()
    return {'name':name,'floor':floor,'half':half,'sr':sr,'windows':windows,'budget':budget,'saved':len(st.r),'motor_saved':sum(k in st.r for k in mr),'g01':sum(st.g(r.w)>=.01 for r in st.r.values()),'g05':sum(st.g(r.w)>=.05 for r in st.r.values()),'g10':sum(st.g(r.w)>=.10 for r in st.r.values()),'us':1e6*elapsed/(windows*budget),'lookup_us':1e6*lookup/(windows*budget),'rows':rows}
def intuition(floor=1.,half=15000.,sr=.25,budget=15000):
    A,B,Y=1,2,1000;noise=list(range(3,40));h=Store(floor,half);fresh=Store(floor,half);step=0
    def window(pred,strong,seed):
        nonlocal step
        c=Cloud(sr,313);rr=random.Random(seed)
        for i in range(budget):
            step+=1;xs={x for x in noise if rr.random()<.06}
            if rr.random()<(0.08 if i<strong else 0.):xs.add(pred)
            ys=set()
            if pred in xs and rr.random()<.90:ys.add(Y)
            elif rr.random()<.02:ys.add(Y)
            c.obs(xs,ys,step)
        return c
    c=window(A,budget,1);h.cp(c,budget);fresh.cp(c,budget)
    for i in range(5):c=window(B,budget,10+i);h.cp(c,budget);fresh.cp(c,budget)
    k=Store.key(A,Y);fresh.r.pop(k,None)
    before={'hist_w':h.r.get(k).w if k in h.r else 0.,'hist_g':h.g(h.r[k].w) if k in h.r else 0.}
    ret=[]
    for i,strong in enumerate((250,250,500,500,1000,1000,2000,4000),1):
        c=window(A,strong,100+i);h.cp(c,budget);fresh.cp(c,budget);hr=h.r.get(k);fr=fresh.r.get(k)
        ret.append({'window':i,'a_exposure':strong,'hist_w':hr.w if hr else 0.,'hist_g':h.g(hr.w) if hr else 0.,'fresh_w':fr.w if fr else 0.,'fresh_g':fresh.g(fr.w) if fr else 0.})
    return {'before':before,'return':ret}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='fixed_budget_persistence.json');a=ap.parse_args()
    ps=[('f1_h15',1.,15000.,.25),('f03_h15',.3,15000.,.25),('f01_h15',.1,15000.,.25),('f03_h30',.3,30000.,.25),('f03_h60',.3,60000.,.25),('f03_h30_sr125',.3,30000.,.125)]
    runs=[]
    for p in ps:print('RUN',p[0],flush=True);runs.append(run(*p))
    report={'host':{'hostname':platform.node(),'platform':platform.platform(),'python':platform.python_version()},'runs':runs,'intuition':intuition()};Path(a.out).write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'host':report['host'],'runs':[{k:r[k] for k in ('name','floor','half','sr','saved','motor_saved','g01','g05','g10','us','lookup_us')} for r in runs],'intuition':report['intuition']},indent=2),flush=True)
if __name__=='__main__':main()
