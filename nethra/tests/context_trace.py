import os, sys, random
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import nethra as core
NAMES=["C1","C2","X","Y","Z","F1","F2"]; C1,C2,X,Y,Z,F1,F2=range(7)

rng=random.Random(0); f=core.NethraField(conduction=os.environ.get("COND","top_and_leaves")); L=[f.new() for _ in range(7)]
def show(xs):
    for x in xs: L[x].push(1.0)
    f.step(1.0)
order=[]
for _ in range(20):
    for c in rng.sample([C1,C2],2):
        for _ in range(3): show([c])
        show([F1])
    for s in rng.sample([Y,Z],2):
        order.append(NAMES[s]); show([X]); show([s]); show([F2])
n_small=len(f.nethra)
for _ in range(30):
    for c,s in rng.sample([(C1,Y),(C2,Z)],2):
        show([c,X]); show([c,s]); show([c,F1]); show([F2])
idx={n:i for i,n in enumerate(f.nethra)}; nm=lambda n: NAMES[idx[n]] if idx[n]<7 else f"N{idx[n]}"
print("first X->? order:", order[:2])
ph=f._physical_incidences(f.current_event)
for who in (X,Y,Z,C1,C2):
    nb=sorted(((nm(r),round(row["g"],3)) for (r,m),row in ph.items() if m is L[who] and row["g"]>0), key=lambda t:-t[1])
    print(f"{NAMES[who]} conducts to:", nb)
for n in f.nethra[7:n_small]+f.nethra[n_small:]:
    cond=[nm(m) for (r,m),row in ph.items() if r is n and row["g"]>0]
    print(f"  {nm(n)} routes", " | ".join("{"+",".join(sorted(map(nm,R)))+"}" for R in n.routes), " conducting:", ",".join(sorted(cond)))
