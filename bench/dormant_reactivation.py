from __future__ import annotations
import json,math,random,platform
from dataclasses import dataclass
from pathlib import Path
MASK=(1<<64)-1

def u01(t,s):
    z=(int(t)+0x9E3779B97F4A7C15*(int(s)+1))&MASK;z=((z^(z>>30))*0xBF58476D1CE4E5B9)&MASK;z=((z^(z>>27))*0x94D049BB133111EB)&MASK;z^=z>>31;return ((z>>11)&((1<<53)-1))/float(1<<53)
class Cloud:
    def __init__(self,sr=.25,seed=313):self.sr=sr;self.seed=seed;self.n=0;self.sc={};self.tc={};self.j={};self.ev={}
    def obs(self,xs,ys,t):
        if u01(t,self.seed)>=self.sr:return
        self.n+=1
        for x in xs:self.sc[x]=self.sc.get(x,0)+1
        for y in ys:self.tc[y]=self.tc.get(y,0)+1
        for y in ys:
            jj=self.j.setdefault(y,{});ee=self.ev.setdefault(y,{});base=self.tc[y]/self.n;gs=[]
            for x in xs:
                jj[x]=jj.get(x,0)+1;c=jj[x]/self.sc[x];g=max(0.,math.log(c/base)) if c>0 and base>0 else 0.;gs.append((x,g));ee.setdefault(x,0.)
            z=sum(g for _,g in gs)
            if z:
                for x,g in gs:
                    if g:ee[x]+=g/z
@dataclass
class Rel:w:float
class Store:
    def __init__(self,floor=1.,half=15000.,eps=.05,tau=100.):self.floor=floor;self.half=half;self.eps=eps;self.tau=tau;self.r={}
    @staticmethod
    def key(a,b):return (a,b) if a<b else (b,a)
    def g(self,w):return 1.5*(1-math.exp(-w/self.tau))
    def cp(self,c,elapsed):
        f=2**(-elapsed/self.half)
        for r in self.r.values():r.w*=f
        obs={}
        for y,p in c.ev.items():
            for x,e in p.items():
                if x!=y:obs[self.key(x,y)]=max(obs.get(self.key(x,y),0.),e)
        for k,e in obs.items():
            if k in self.r:self.r[k].w+=e/c.sr
            elif e>=self.floor:self.r[k]=Rel(e/c.sr)

def main():
    A,B,Y=1,2,1000;noise=list(range(3,40));budget=15000;sr=.25;h=Store();fresh=Store();step=0
    def win(pred,active_prefix,seed):
        nonlocal step
        c=Cloud(sr);rr=random.Random(seed)
        for i in range(budget):
            step+=1;xs={x for x in noise if rr.random()<.06}
            if i<active_prefix and rr.random()<.08:xs.add(pred)
            ys=set()
            if pred in xs and rr.random()<.90:ys.add(Y)
            elif rr.random()<.02:ys.add(Y)
            c.obs(xs,ys,step)
        return c
    c=win(A,budget,1);h.cp(c,budget);fresh.cp(c,budget);k=Store.key(A,Y)
    gap=[]
    for i in range(10):
        c=win(B,budget,10+i);h.cp(c,budget);fresh.cp(c,budget);r=h.r.get(k);gap.append({'gap':i+1,'w':r.w if r else 0.,'g':h.g(r.w) if r else 0.,'active':bool(r and h.g(r.w)>=h.eps)})
    fresh.r.pop(k,None)
    ret=[]
    for i,prefix in enumerate((50,100,200,400,800,1600,3200,6400),1):
        c=win(A,prefix,100+i);h.cp(c,budget);fresh.cp(c,budget);hr=h.r.get(k);fr=fresh.r.get(k)
        ret.append({'return':i,'prefix':prefix,'hist_w':hr.w if hr else 0.,'hist_g':h.g(hr.w) if hr else 0.,'hist_active':bool(hr and h.g(hr.w)>=h.eps),'fresh_w':fr.w if fr else 0.,'fresh_g':fresh.g(fr.w) if fr else 0.,'fresh_active':bool(fr and fresh.g(fr.w)>=fresh.eps)})
    out={'host':{'hostname':platform.node(),'platform':platform.platform(),'python':platform.python_version()},'gap':gap,'return':ret};Path('dormant_reactivation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
