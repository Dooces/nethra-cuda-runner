"""How long in a direction: object bounces on 0..L-1 at speed 1 with displacement Nethra (d-, d0, d+).
Last rightward run: live activation of d+ at each step of the run, and for each constructed Nethra
the number of run intervals it is refound in (closure under current topology)."""
import os
os.environ.setdefault("ND","3"); os.environ.setdefault("VMAX","1")
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"delta_input.py")).read().split("LASTN=")[0])
runs=[]; cur=[]
for t,x in enumerate(xs):
    for i,w in tents(x,NC,0,L-1).items(): pc[i].push(w)
    if DELTA and t>0:
        for i,w in tents(x-xs[t-1],ND,-VMAX,VMAX).items(): dc[i].push(w)
    f.step(1.0)
    if t>0 and x>xs[t-1]:
        clo=f.closure(f.previous_explicit,f.current_source_event)
        cur.append((dc[-1].activation, frozenset(n for n in clo if n not in pc and n not in dc)))
    elif cur: runs.append(cur); cur=[]
if cur: runs.append(cur)
last=runs[-1]
print(f"leak={LEAK} run length {len(last)}: d+ activation by step", " ".join(f"{a:.3f}" for a,_ in last[:8]))
from collections import Counter
cnt=Counter(n for _,s in last for n in s)
print("  constructed Nethra refound in how many of the run's intervals:", sorted(Counter(cnt.values()).items()), "(count: nethra)")
