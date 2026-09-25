"""Timers (square waves, periods PER, default 2,4,8) + process E pushed D intervals (D may be a list: drawn per episode), then finish O, then a gap of timers only.
COND: none | locked | random | reset.  Read: P toward O after each E interval k=1..D, last EPREAD episodes."""
import os,sys,math,random,time
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import nethra as core
COND=sys.argv[1]; DS=[int(x) for x in os.environ.get("D","5").split(",")]; D=max(DS); EP=int(os.environ.get("EP",20)); EPREAD=int(os.environ.get("EPREAD",8))
PER=[int(x) for x in os.environ.get("PER","2,4,8").split(",")]; SEED=int(os.environ.get("SEED",1))
DIR=os.environ.get("DIR","shared")
def P_toward(f):
    A=f.current_interval_integral; out={}
    for (rel,mem),row in f._physical_incidences(f.current_event).items():
        d=A.get(rel,0.0)-A.get(mem,0.0)
        g=row["g"] if (d>0 or f.direction=="shared") else row["g_in"]
        q=g*d
        if q>0: out[mem]=out.get(mem,0.0)+q
    return out
f=core.NethraField(direction=DIR); rng=random.Random(SEED)
T={p:(f.new(),f.new()) for p in PER}; E=f.new(); O=f.new()
clock=[0]
def push_timers():
    if COND=="none": return
    c=clock[0]
    for p,(hi,lo) in T.items(): (hi if (c%p)<p//2 else lo).push(1.0)
def tick(extra=()):
    push_timers()
    for n in extra: n.push(1.0)
    f.step(1.0); clock[0]+=1
t0=time.perf_counter()
WARM=32 if COND!="none" else 0
for _ in range(WARM): tick()
built_warm=len(f.nethra)
rows=[]; onsets=[]; allrows=[]
for ep in range(EP):
    if COND=="reset": clock[0]=0
    onsets.append(clock[0]%max(PER) if COND!="none" else -1)
    pk=[]; d=rng.choice(DS)
    for k in range(1,d+1):
        tick((E,)); P=P_toward(f)
        cl=f.closure(f.previous_explicit,f.current_source_event)
        nO=sum(1 for n in cl if any(O in r for r in n.routes))
        pk.append((P.get(O,0.0),O.activation,nO,k==d))
    tick((O,))
    gap=10 if COND=="locked" else rng.randint(7,13)
    for _ in range(gap): tick()
    if ep>=EP-EPREAD: rows.append(pk)
    allrows.append(pk)
el=time.perf_counter()-t0
print(f"COND={COND} DIR={DIR} PER={PER} D={D} built warm {built_warm-len(PER)*2-2 if COND!='none' else 0} total constructed {len(f.nethra)-len(PER)*2-2}  onset phases {onsets}  {el:.1f}s")
rr=[r for r in allrows[EP-EPREAD:]]
print(" k | n eps | mean P->O | refound Nethra with O route (mean) | share of eps where O came next")
for k in range(1,D+1):
    xs=[r[k-1] for r in rr if len(r)>=k]
    if not xs: continue
    print(f" {k} | {len(xs)} | {sum(x[0] for x in xs)/len(xs):.4f} | {sum(x[2] for x in xs)/len(xs):.2f} | {sum(x[3] for x in xs)/len(xs):.2f}")
