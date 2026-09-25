import os,sys,random,itertools
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import nethra as core, direction_variants; direction_variants.install(core)
E=os.environ.get; P=int(E("P",4)); k=int(E("K",6)); DIR=E("DIR","shared"); BLK=int(E("BLK",40))
pairs=list(itertools.combinations(range(P),2)); rng=random.Random(7); rng.shuffle(pairs); ctx=pairs[:k]
A,B=P,P+1
f=core.NethraField(direction=DIR); L=[f.new() for _ in range(P+2+k)]
names=[f"f{i}" for i in range(P)]+["A","B"]+[f"O{j}" for j in range(k)]
def nm(n):
    i=f.nethra.index(n); return names[i] if i<len(names) else f"N{i}"
def show(xs):
    for x in xs: L[x].push(1.0)
    f.step(1.0)
wrong=[]; last=None
for blk in range(BLK*k):
    c=rng.randrange(k)
    for s in (A,B,P+2+c):
        show(list(ctx[c])+[s])
        if s==B and blk>=(BLK*3//4)*k:
            acts=[L[P+2+j].activation for j in range(k)]
            if acts[c]!=max(acts): wrong.append((blk,c,acts.index(max(acts))))
print(f"DIR={DIR} contexts {[tuple(names[i] for i in p) for p in ctx]} wrong reads (block, right, read top): {wrong}")
# Nethra leading to each outcome: relations having O_j in a route
ph=f._physical_incidences(f.current_event)
for j in range(k):
    O=L[P+2+j]
    rels=[r for r in f.nethra[P+2+k:] if any(O in R for R in r.routes)]
    print(f"O{j} ctx {tuple(names[i] for i in ctx[j])}: {len(rels)} Nethra hold it")
    for r in rels[:3]:
        rs=" | ".join("{"+",".join(sorted(map(nm,R)))+"}" for R in r.routes)
        gs=" ".join(f"{nm(m)}:{row['g']:.2f}/{row.get('g_in',row['g']):.2f}" for (rr,m),row in ph.items() if rr is r and (row['g']>0 or row.get('g_in',0)>0))
        print(f"   {nm(r)} {rs}\n      g rel->mem/mem->rel: {gs}")
