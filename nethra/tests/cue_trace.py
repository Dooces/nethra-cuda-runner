import os, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import nethra as core, conduction_variants as cond; cond.install(core)
NM=["A","B","C","c1","c2","Z"]
f=core.NethraField(); L=[f.new() for _ in range(6)]
def show(xs):
    for x in xs: L[x].push(1.0)
    f.step(1.0)
for _ in range(40): show([0,3]); show([1,3]); show([5])
n1=len(f.nethra)
for _ in range(40): show([0,4]); show([2,4]); show([5])
idx={n:i for i,n in enumerate(f.nethra)}; nm=lambda n: NM[idx[n]] if idx[n]<6 else f"N{idx[n]}"
ph=f._physical_incidences(f.current_event)
print("phase1 built", n1-6, "phase2 built", len(f.nethra)-n1)
for n in f.nethra[6:]:
    cond_=sorted(f"{nm(m)}:{row['g']:.2f}" for (r,m),row in ph.items() if r is n and row["g"]>0)
    print(f"  {nm(n)}", " | ".join("{"+",".join(sorted(map(nm,R)))+"}" for R in n.routes), " conducts", ",".join(cond_))
g=core.NethraField.from_checkpoint_dict(f.checkpoint_dict()); g.topology_and_evidence_change=False
M=g.nethra
for _ in range(4): M[5].push(1.0); g.step(1.0)
M[0].push(1.0); M[4].push(1.0); g.step(1.0)
print(" A+c2: B", round(M[1].activation,4), "C", round(M[2].activation,4), " constructed A:", {nm(n):round(n.activation,4) for n in M[6:]})
