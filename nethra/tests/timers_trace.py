"""Topology and per-relation flow into O at each k of the last episode (timers.py stream). Arg: COND."""
import os,sys
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"timers.py")).read().split("t0=time.perf_counter()")[0])
names={}
for p,(hi,lo) in T.items(): names[hi]=f"T{p}h"; names[lo]=f"T{p}l"
names[E]="E"; names[O]="O"
nm=lambda n: names.get(n, f"N{f.nethra.index(n)-len(names)+1}")
WARM=32 if COND!="none" else 0
for _ in range(WARM): tick()
for ep in range(EP):
    if COND=="reset": clock[0]=0
    last= ep==EP-1
    for k in range(1,D+1):
        tick((E,))
        if last:
            A=f.current_interval_integral; parts=[]
            for (rel,mem),row in f._physical_incidences(f.current_event).items():
                if mem is O:
                    d=A.get(rel,0)-A.get(O,0); parts.append(f"{nm(rel)}:g{row['g']:.2f} A{A.get(rel,0):.3f} q{max(0,row['g']*d):.4f}")
            cl=[nm(n) for n in f.closure(f.previous_explicit,f.current_source_event) if n not in names]
            print(f"k={k} closure {cl}\n   into O: "+"  ".join(parts))
    tick((O,))
    gap=10 if COND=="locked" else rng.randint(7,13)
    for _ in range(gap): tick()
print("topology:")
for n in f.nethra:
    if n in names: continue
    print(" ",nm(n)," | ".join("{"+",".join(sorted(map(nm,r)))+"}" for r in n.routes))
