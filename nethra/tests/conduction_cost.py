"""Cost by conduction variant (env COND, see conduction_variants.py): two-object binocular stream
(gpu_bench.py stream, tol 0.01, 0.8), ms/interval, frontier and Nethra built per 80 together.
usage: COND=topleaves conduction_cost.py 0 400 unused"""
import os, sys, json, math, random
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
T=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.dirname(T)); sys.path.insert(0,T)
TOGETHER, MEASURE = int(sys.argv[1]), int(sys.argv[2]); OUT=sys.argv[3]
sys.argv=[sys.argv[0],"16"]
import binocular3d as b, nethra as core
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import conduction_variants as cond; cond.install(core)
TH,J,AL,PER2=0.8,10,3,37
def loop(c,rx,ry,rz,per,fy):
    return [(c[0]+rx*math.sin(2*math.pi*k/per), c[1]+ry*math.sin(fy*2*math.pi*k/per+.5), c[2]+rz*math.cos(2*math.pi*k/per)) for k in range(per)]
LOOPS=[loop((250,130,250),200,80,200,40,2), loop((250,370,250),180,80,200,PER2,1)]
rng=random.Random(1); counters=[0,0]
def where(i):
    base=LOOPS[i][counters[i]%len(LOOPS[i])]; counters[i]+=1
    return tuple(min(499,max(0,round(base[a]+rng.randint(-J,J)))) for a in range(3))
stream=[]
for i,L in enumerate(LOOPS):
    for eyes,n in (((0,),AL*len(L)),((1,),AL*len(L)),((0,1),2*AL*len(L))): stream+=[(eyes,[where(i)]) for _ in range(n)]
stream+=[((0,1),[where(0),where(1)]) for _ in range(TOGETHER+MEASURE)]
# convert to push lists
def pushes(eyes,ps):
    class R:  # record pushes
        def __init__(s): s.nethra=[P(i) for i in range(2*b.R*b.R)]; s.out={}
    class P:
        def __init__(s,i): s.i=i
        def push(s,v): rec.out[s.i]=rec.out.get(s.i,0.0)+float(v)
    global rec; rec=R(); b.push(rec,eyes,ps); return sorted(rec.out.items())
plist=[pushes(e,p) for e,p in stream]
import time
BUILD=len(plist)-MEASURE
f=core.NethraField(frontier_tolerance=0.01, source_similarity_threshold=TH)
for _ in range(2*b.R*b.R): f.new()
for pl in plist[:BUILD]:
    for i,v in pl: f.nethra[i].push(v)
    f.step(1.0)
out=[]
for c in range(0,MEASURE,80):
    t0=time.perf_counter(); n0=len(f.frontier_sizes); k0=len(f.nethra)
    for pl in plist[BUILD+c:BUILD+c+80]:
        for i,v in pl: f.nethra[i].push(v)
        f.step(1.0)
    fr=f.frontier_sizes[n0:]
    out.append(f"{(time.perf_counter()-t0)/len(fr)*1000:.0f}ms/fr{sum(fr)/len(fr):.0f}/+{len(f.nethra)-k0}")
cond_inc=sum(1 for k,v in f.incidence_evidence.items() if k not in f.covered_incidences) if f.top_only_conduction else len(f.incidence_evidence)
print(os.environ.get("COND"), "together 0-79 ... :", " ".join(out), "| conducting incidences", cond_inc, "of", len(f.incidence_evidence))
