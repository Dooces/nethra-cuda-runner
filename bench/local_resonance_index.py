from __future__ import annotations
import json,random,statistics,time,platform
from pathlib import Path
from fixed_budget_persistence import Cloud,Store,World,sup

def train():
    floor=.3;half=15000.;sr=1.;budget=15000;windows=8;seed=9901
    st=Store(floor,half);w=World();rng=random.Random(seed);m=set();prev=None;step=0
    for _ in range(windows):
        c=Cloud(sr,seed^555)
        for __ in range(budget):
            step+=1
            for x in range(12):
                if rng.random()<.02:
                    if x in m:m.remove(x)
                    else:m.add(x)
            cur=sup(w.step(m))
            if prev is not None:c.obs(prev,cur,step)
            prev=cur
        st.cp(c,budget)
    return st

def main():
    st=train();adj={}
    for i,(a,b) in enumerate(st.r):
        adj.setdefault(a,[]).append(i);adj.setdefault(b,[]).append(i)
    deg=[len(v) for v in adj.values()]
    w=World();rng=random.Random(12345);m=set();sizes=[];lookup=[]
    for step in range(15000):
        for x in range(12):
            if rng.random()<.02:
                if x in m:m.remove(x)
                else:m.add(x)
        cur=sup(w.step(m));t=time.perf_counter();touched=set()
        for x in cur:touched.update(adj.get(x,()))
        lookup.append(time.perf_counter()-t);sizes.append(len(touched))
    s=sorted(sizes);d=sorted(deg)
    out={'host':{'hostname':platform.node(),'platform':platform.platform(),'python':platform.python_version()},'saved_relations':len(st.r),'indexed_members':len(adj),'degree':{'mean':statistics.mean(deg),'median':statistics.median(deg),'p90':d[int(.9*(len(d)-1))],'p99':d[int(.99*(len(d)-1))],'max':max(deg)},'touched_per_interval':{'mean':statistics.mean(sizes),'median':statistics.median(sizes),'p90':s[int(.9*(len(s)-1))],'p99':s[int(.99*(len(s)-1))],'max':max(sizes)},'lookup_us':{'mean':1e6*statistics.mean(lookup),'median':1e6*statistics.median(lookup),'p90':1e6*sorted(lookup)[int(.9*(len(lookup)-1))]}}
    Path('local_resonance_index.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
