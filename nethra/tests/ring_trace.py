import os,sys
os.environ.setdefault("SHOW","0")
src=open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"ring_graded.py")).read()
exec(src.split("f=core.NethraField")[0])
f=core.NethraField(top_only_conduction=TOP, leakage=LEAK)
cells=[f.new() for _ in range(NC)]
nm=lambda n: f"c{cells.index(n)}" if n in cells else f"N{f.nethra.index(n)-NC+1}"
X=int(os.environ.get("X",5)); positions=[(STEP*t)%L for t in range(L)]
for lap in range(LAPS):
    for t,x in enumerate(positions):
        for i,w in weights(x).items(): cells[i].push(w)
        f.step(1.0)
        if lap==LAPS-1 and x==X: break
    else: continue
    break
print("routes:")
for n in f.nethra[NC:]:
    print(" ",nm(n)," | ".join("{"+",".join(sorted(map(nm,r)))+"}" for r in n.routes))
A=f.current_interval_integral
print("A (interval with x=%d):"%X, " ".join(f"{nm(n)}:{A.get(n,0):.3f}" for n in f.nethra))
print("conducting incidences with flow toward a cell:")
for (rel,mem),row in f._physical_incidences(f.current_event).items():
    if row["g"]>0:
        q=row["g"]*(A.get(rel,0)-A.get(mem,0))
        print(f"  {nm(rel)}-{nm(mem)} g {row['g']:.3f} q(toward member) {q:+.4f}")
