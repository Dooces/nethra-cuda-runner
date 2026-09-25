"""Two-object binocular stream of gpu_bench.py (current core, top-only default, tol 0.01, 0.8).
Builds up to TOGETHER intervals together, saves checkpoint + remaining stream for MEASURE intervals."""
import os, sys, json, math, random
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
T=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0,T)
TOGETHER, MEASURE = int(sys.argv[1]), int(sys.argv[2]); OUT=sys.argv[3]
sys.argv=[sys.argv[0],"16"]
import binocular3d as b, nethra as core
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
BUILD=len(plist)-MEASURE
f=core.NethraField(frontier_tolerance=0.01, source_similarity_threshold=TH, top_only_conduction=os.environ.get("TOP","1")=="1")
for _ in range(2*b.R*b.R): f.new()
for pl in plist[:BUILD]:
    for i,v in pl: f.nethra[i].push(v)
    f.step(1.0)
json.dump({"checkpoint":f.checkpoint_dict(),"measure":plist[BUILD:]}, open(OUT,"w"))
print("nethra",len(f.nethra),"incidences",len(f.incidence_evidence),"covered",len(f.covered_incidences))
