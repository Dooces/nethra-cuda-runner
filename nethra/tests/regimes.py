"""cue_capacity part 2 stream (contexts = pairs of P features, pushed in all 3 intervals of a block:
ctx+A, ctx+B, ctx+O_c).  Read after the B interval of the last quarter: O_c activation max?  env P K DIR BLK"""
import os,sys,random,itertools,time
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import nethra as core, direction_variants; direction_variants.install(core)
E=os.environ.get; P=int(E("P",6)); k=int(E("K",8)); DIR=E("DIR","shared"); BLK=int(E("BLK",40))
pairs=list(itertools.combinations(range(P),2)); rng=random.Random(7); rng.shuffle(pairs); ctx=pairs[:k]
A,B=P,P+1
f=core.NethraField(direction=DIR); L=[f.new() for _ in range(P+2+k)]; ok=[]; t0=time.perf_counter()
def show(xs):
    for x in xs: L[x].push(1.0)
    f.step(1.0)
for blk in range(BLK*k):
    c=rng.randrange(k)
    for s in (A,B,P+2+c):
        show(list(ctx[c])+[s])
        if s==B and blk>=(BLK*3//4)*k:
            acts=[L[P+2+j].activation for j in range(k)]; ok.append(acts[c]==max(acts))
print(f"P={P} K={k} DIR={DIR} BLK={BLK}: correct top {sum(ok)/len(ok):.2f} (n={len(ok)}) Nethra {len(f.nethra)} {time.perf_counter()-t0:.0f}s")
