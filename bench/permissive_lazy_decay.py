from __future__ import annotations
import json, math, random, statistics, time, platform
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from fixed_budget_persistence import World, sup, Cloud, motors

@dataclass
class LazyRel:
    weight: float
    last_step: int
    confirms: int = 1

class LazyStore:
    """Persistence is permissive; decay is lazy and has no per-checkpoint O(N) sweep."""
    def __init__(self, density_floor: float, half_life: float = 60000.0, tau: float = 100.0):
        self.floor = float(density_floor)
        self.half = float(half_life)
        self.tau = float(tau)
        self.r: Dict[Tuple[int,int], LazyRel] = {}

    @staticmethod
    def key(a: int, b: int) -> Tuple[int,int]:
        return (a,b) if a < b else (b,a)

    def effective_weight(self, rr: LazyRel, step: int) -> float:
        if self.half <= 0:
            return 0.0
        return rr.weight * (2.0 ** (-(max(0, int(step)-rr.last_step) / self.half)))

    def conductance(self, rr: LazyRel, step: int) -> float:
        w = self.effective_weight(rr, step)
        return 1.5 * (1.0 - math.exp(-max(0.0,w)/self.tau))

    def checkpoint(self, c: Cloud, step: int) -> dict:
        t0 = time.perf_counter()
        observed: Dict[Tuple[int,int], float] = {}
        for y,pool in c.ev.items():
            for x,ev in pool.items():
                if x == y:
                    continue
                k = self.key(int(x), int(y))
                if ev > observed.get(k,0.0):
                    observed[k] = float(ev)
        created = reinforced = 0
        processed = max(1,c.n)
        for k,ev in observed.items():
            density = 1000.0 * ev / processed
            rr = self.r.get(k)
            if rr is None:
                if density >= self.floor:
                    self.r[k] = LazyRel(ev/max(c.sr,1e-12), int(step), 1)
                    created += 1
            else:
                rr.weight = self.effective_weight(rr, step) + ev/max(c.sr,1e-12)
                rr.last_step = int(step)
                rr.confirms += 1
                reinforced += 1
        return {
            "created": created,
            "reinforced": reinforced,
            "saved": len(self.r),
            "checkpoint_ms": 1000.0*(time.perf_counter()-t0),
        }

    def lookup(self, active_support, step: int, epsilon: float = 0.01) -> int:
        xs = sorted(active_support)
        hits = 0
        for i,a in enumerate(xs):
            for b in xs[i+1:]:
                rr = self.r.get(self.key(a,b))
                if rr is not None and self.conductance(rr, step) >= epsilon:
                    hits += 1
        return hits

    def final_scan(self, step: int, epsilons=(.001,.01,.05,.10)) -> dict:
        t0=time.perf_counter()
        weights=[self.effective_weight(rr,step) for rr in self.r.values()]
        gs=[1.5*(1.0-math.exp(-max(0.0,w)/self.tau)) for w in weights]
        return {
            "scan_ms":1000.0*(time.perf_counter()-t0),
            "saved":len(weights),
            "active":{str(e):sum(g>=e for g in gs) for e in epsilons},
            "weight_sum":sum(weights),
        }

def run_policy(name: str, *, sample_rate: float, density_floor: float, half_life: float,
               windows: int = 8, budget: int = 15000, seed: int = 12001) -> dict:
    st=LazyStore(density_floor,half_life)
    w=World(); rng=random.Random(seed); mot=set(); prev=None
    step=0; rows=[]; lookup_s=0.0; t0=time.perf_counter()
    for win in range(windows):
        c=Cloud(sample_rate,seed^0x5151)
        for _ in range(budget):
            step += 1
            for m in range(12):
                if rng.random() < .02:
                    if m in mot: mot.remove(m)
                    else: mot.add(m)
            cur=sup(w.step(mot))
            if prev is not None: c.obs(prev,cur,step)
            if st.r:
                a=time.perf_counter(); st.lookup(cur,step,.01); lookup_s += time.perf_counter()-a
            prev=cur
        rows.append({"window":win+1, **st.checkpoint(c,step)})
    elapsed=time.perf_counter()-t0
    mr=motors()
    m_saved=sum(k in st.r for k in mr)
    m_active=sum(k in st.r and st.conductance(st.r[k],step)>=.01 for k in mr)
    return {
        "name":name,"sample_rate":sample_rate,"density_floor":density_floor,"half_life":half_life,
        "windows":windows,"budget":budget,"saved":len(st.r),"motor_saved":m_saved,"motor_active":m_active,
        "us_per_step":1e6*elapsed/(windows*budget),
        "lookup_us_per_step":1e6*lookup_s/(windows*budget),
        "median_checkpoint_ms":statistics.median(r["checkpoint_ms"] for r in rows),
        "final":st.final_scan(step),"rows":rows,
    }

def intuition_case(*, density_floor=.025, sample_rate=.0625, half_life=30000.0, budget=15000):
    A,B,Y=1,2,1000; noise=list(range(3,40)); step=0
    hist=LazyStore(density_floor,half_life); fresh=LazyStore(density_floor,half_life)
    def window(pred, active_prefix, seed):
        nonlocal step
        c=Cloud(sample_rate,777)
        rr=random.Random(seed)
        for i in range(budget):
            step += 1
            xs={x for x in noise if rr.random()<.06}
            if i < active_prefix and rr.random()<.08: xs.add(pred)
            ys=set()
            if pred in xs and rr.random()<.90: ys.add(Y)
            elif rr.random()<.02: ys.add(Y)
            c.obs(xs,ys,step)
        return c
    c=window(A,budget,1); hist.checkpoint(c,step); fresh.checkpoint(c,step)
    key=LazyStore.key(A,Y)
    gap=[]
    for i in range(16):
        c=window(B,budget,10+i); hist.checkpoint(c,step); fresh.checkpoint(c,step)
        hr=hist.r.get(key)
        gap.append({"gap":i+1,
                    "hist_w":hist.effective_weight(hr,step) if hr else 0.0,
                    "hist_g":hist.conductance(hr,step) if hr else 0.0})
    fresh.r.pop(key,None)
    returns=[]
    prefixes=(25,50,100,200,400,800,1600,3200)
    for i,prefix in enumerate(prefixes,1):
        c=window(A,prefix,100+i); hist.checkpoint(c,step); fresh.checkpoint(c,step)
        hr=hist.r.get(key); fr=fresh.r.get(key)
        returns.append({
            "return":i,"prefix":prefix,
            "hist_w":hist.effective_weight(hr,step) if hr else 0.0,
            "hist_g":hist.conductance(hr,step) if hr else 0.0,
            "fresh_w":fresh.effective_weight(fr,step) if fr else 0.0,
            "fresh_g":fresh.conductance(fr,step) if fr else 0.0,
        })
    return {"gap":gap,"return":returns}

def main():
    policies=[
        ("strict_sr25_f08",.25,.8,60000.0),
        ("perm_sr25_f008",.25,.08,60000.0),
        ("throttle125_f008",.125,.08,60000.0),
        ("throttle0625_f0025",.0625,.025,60000.0),
        ("throttle0625_f0008",.0625,.008,30000.0),
    ]
    runs=[]
    for name,sr,floor,half in policies:
        print("RUN",name,flush=True)
        runs.append(run_policy(name,sample_rate=sr,density_floor=floor,half_life=half))
    intuition=intuition_case()
    report={"host":{"hostname":platform.node(),"platform":platform.platform(),"python":platform.python_version()},
            "runs":runs,"intuition":intuition}
    Path("permissive_lazy_decay.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "host":report["host"],
        "runs":[{k:r[k] for k in ("name","sample_rate","density_floor","saved","motor_saved","motor_active","us_per_step","lookup_us_per_step","median_checkpoint_ms","final")} for r in runs],
        "intuition":intuition,
    },indent=2),flush=True)

if __name__=="__main__":
    main()
