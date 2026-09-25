"""Delta input, field reads only (env COND as in conduction_variants.py, default top; topleaves = core default).  Object bounces on positions 0..L-1 (speeds cycled per pass).
Position: NC tent cells.  MODE: none | delta (displacement tent cells, pushed with its shares) |
shuffle (same displacement values, randomly permuted in time).  Frontier 0.
Reads over the last LASTN intervals, P = sum max(0, g(A_rel - A_mem)) of the completed interval:
  share of P over position cells at the true next position's cells (weights of next pattern),
  at the previous position's cells, P centroid |err| to next vs staying put;
  ablation on a frozen copy: same interval pushed without the displacement cells -> P next share."""
import os, sys, math, random, importlib.util
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import nethra as core, conduction_variants as cond; cond.install(core)
E=os.environ.get
L=int(E("L",8)); NC=int(E("NC",L)); MODE=E("MODE","delta"); ND=int(E("ND",3)); VMAX=float(E("VMAX",1))
SPEEDS=[int(s) for s in E("SPEEDS","1").split(",")]; PASSES=int(E("PASSES",8))
ABL=E("ABL","1")=="1"
def tents(v,n,lo,hi):
    s=(hi-lo)/(n-1); return {i:1-abs(v-(lo+i*s))/s for i in range(n) if 1-abs(v-(lo+i*s))/s>1e-9}
xs=[0]
for k in range(PASSES):
    v=SPEEDS[k%len(SPEEDS)]*(1 if k%2==0 else -1); tgt=L-1 if v>0 else 0; x=xs[-1]
    while x!=tgt: x=max(0,min(L-1,x+v)); xs.append(x)
ds=[0]+[xs[t]-xs[t-1] for t in range(1,len(xs))]
if MODE=="shuffle":
    rng=random.Random(3); tail=ds[1:]; rng.shuffle(tail); ds=[0]+tail
f=core.NethraField()
pc=[f.new() for _ in range(NC)]; dc=[f.new() for _ in range(ND)] if MODE!="none" else []
cpos=[i*(L-1)/(NC-1) for i in range(NC)]
def P_toward(g):
    A=g.current_interval_integral; out={}
    for (r,m),row in g._physical_incidences(g.current_event).items():
        q=row["g"]*(A.get(r,0.0)-A.get(m,0.0))
        if q>0: out[m]=out.get(m,0.0)+q
    return out
def push(g,t,with_d=True):
    for i,w in tents(xs[t],NC,0,L-1).items(): g.nethra[i].push(w)
    if with_d and dc and t>0:
        for i,w in tents(ds[t],ND,-VMAX,VMAX).items(): g.nethra[NC+i].push(w)
def reads(g,t):
    P=P_toward(g); pv=[P.get(g.nethra[i],0.0) for i in range(NC)]; tot=sum(pv)
    nxt=tents(xs[t+1],NC,0,L-1); prv=tents(xs[t-1],NC,0,L-1) if t>0 else {}
    sh_n=sum(pv[i]*w for i,w in nxt.items())/tot if tot>0 else 0.0
    sh_p=sum(pv[i]*w for i,w in prv.items())/tot if tot>0 else 0.0
    cen=sum(p*c for p,c in zip(pv,cpos))/tot if tot>0 else None
    dirn=1 if xs[t+1]>xs[t] else -1
    ah=sum(p for p,c in zip(pv,cpos) if (c-xs[t])*dirn>0.5)/tot if tot>0 else 0.0
    return sh_n, sh_p, cen, tot, ah
LASTN=2*(L-1); R=[]; Ab=[]
for t,x in enumerate(xs[:-1]):
    last = t>=len(xs)-1-LASTN
    if last and ABL:
        g=core.NethraField.from_checkpoint_dict(f.checkpoint_dict()); g.topology_and_evidence_change=False
        push(g,t,with_d=False); g.step(1.0); Ab.append(reads(g,t))
    push(f,t); f.step(1.0)
    if last: R.append(reads(f,t)+(abs(x-xs[t+1]),))
def m(v): v=[a for a in v if a is not None]; return sum(v)/len(v) if v else float("nan")
cerr=[abs(r[2]-xs[len(xs)-1-LASTN+k+0]- (xs[len(xs)-LASTN+k]-xs[len(xs)-1-LASTN+k])) if r[2] is not None else None for k,r in enumerate(R)]
# simpler: recompute centroid error explicitly
ts=list(range(len(xs)-1-LASTN,len(xs)-1)); cerr=[abs(r[2]-xs[t+1]) for r,t in zip(R,ts) if r[2] is not None]
print(f"L={L} NC={NC} mode={MODE} speeds={SPEEDS} cond={E('COND','top')}: nethra {len(f.nethra)-NC-len(dc)} | P share at next cells {m([r[0] for r in R]):.3f}"
      f" at previous {m([r[1] for r in R]):.3f} | P centroid |err| {m(cerr):.2f} staying {m([r[5] for r in R]):.2f} | P ahead {m([r[4] for r in R]):.2f} | P total {m([r[3] for r in R]):.4f}"
      + (f" | ablation (same field, d not pushed): next {m([a[0] for a in Ab]):.3f} prev {m([a[1] for a in Ab]):.3f} ahead {m([a[4] for a in Ab]):.2f} total {m([a[3] for a in Ab]):.4f}" if Ab and dc else ""))
