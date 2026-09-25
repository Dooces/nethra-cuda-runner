"""1D object, graded tent cells.  Stream: RING (step STEP around a ring) or BOUNCE (bounces between
walls at speeds SPEEDS).  Direction mode DIR (shared/split).  Read on the last READN intervals: the
field's own next step, no input: frozen copy, nothing pushed, one step; gain = activation - leakage-only
decay of the current activation (what conduction added; exactly 0 without constructed Nethra).
  gain share ahead / behind the object along the true next motion (cells > 0.5 spacing away),
  centroid of positive gain vs true next position vs staying put.
Also the same reads at reversal intervals only (BOUNCE)."""
import os,sys,math,random,cmath
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import nethra as core
E=os.environ.get
L=int(E("L",12)); NC=int(E("NC",8)); STREAM=E("STREAM","RING"); STEP=int(E("STEP",1)); LAPS=int(E("LAPS",6))
SPEEDS=[int(s) for s in E("SPEEDS","1").split(",")]; PASSES=int(E("PASSES",10)); DIR=E("DIR","shared")
SHUF=E("SHUF","0")=="1"; READN=int(E("READN",0))
ring = STREAM=="RING"
if ring:
    sp=L/NC; cpos=[i*sp for i in range(NC)]
    xs=[(STEP*t)%L for t in range(L*LAPS+1)]
    def dist(a,b): return ((a-b)+L/2)%L-L/2        # signed a-b on ring
else:
    sp=(L-1)/(NC-1); cpos=[i*sp for i in range(NC)]
    xs=[0]
    for k in range(PASSES):
        v=SPEEDS[k%len(SPEEDS)]*(1 if k%2==0 else -1); tgt=L-1 if v>0 else 0; x=xs[-1]
        while x!=tgt: x=max(0,min(L-1,x+v)); xs.append(x)
    def dist(a,b): return a-b
def weights(x):
    out={}
    for i,c in enumerate(cpos):
        w=1-abs(dist(x,c))/sp
        if w>1e-9: out[i]=w
    return out
if SHUF:   # same positions, random time order: no motion to carry
    rng=random.Random(5); body=xs[:-1]; rng.shuffle(body); xs=body+[xs[-1]]
f=core.NethraField(direction=DIR); cells=[f.new() for _ in range(NC)]
READN=READN or (2*L if ring else 2*(L-1))
res=[]
T=len(xs)-1
for t in range(T):
    for i,w in weights(xs[t]).items(): cells[i].push(w)
    f.step(1.0)
    if t>=T-READN:
        g=core.NethraField.from_checkpoint_dict(f.checkpoint_dict()); g.topology_and_evidence_change=False
        now=[c.activation for c in g.nethra[:NC]]; g.step(1.0)
        gain=[g.nethra[i].activation-now[i]*math.exp(-f.leakage) for i in range(NC)]
        after=[g.nethra[i].activation for i in range(NC)]
        def centre(vals):
            vals={i:max(0.0,v) for i,v in enumerate(vals)}
            if ring:
                z=sum(v*cmath.exp(2j*math.pi*cpos[i]/L) for i,v in vals.items()); return (cmath.phase(z)/(2*math.pi)*L)%L
            return sum(v*cpos[i] for i,v in vals.items())/max(1e-300,sum(vals.values()))
        x,xn=xs[t],xs[t+1]; step=dist(xn,x); sgn=1 if step>0 else -1
        ah=bh=0.0
        for i,v in enumerate(gain):
            if v<=0: continue
            o=dist(cpos[i],x)*sgn
            if o>0.5*sp: ah+=v
            elif o<-0.5*sp: bh+=v
        pos={i:v for i,v in enumerate(gain) if v>0}
        if pos:
            if ring:
                z=sum(v*cmath.exp(2j*math.pi*cpos[i]/L) for i,v in pos.items()); cen=(cmath.phase(z)/(2*math.pi)*L)%L
            else:
                cen=sum(v*cpos[i] for i,v in pos.items())/sum(pos.values())
            err=abs(dist(cen,xn))
        else: err=None
        rev = (not ring) and t>0 and (xs[t]-xs[t-1])*(xn-x)<0
        prof=[0.0]*9
        for i,v in enumerate(gain):
            o=int(round(dist(cpos[i],x)*sgn/sp)); 
            if -4<=o<=4: prof[o+4]+=v
        shift=dist(centre(after),centre(now))*sgn
        res.append((ah,bh,err,abs(step),rev,sum(pos.values()),prof,shift))
def rep(rs,label):
    A=sum(r[0] for r in rs); B=sum(r[1] for r in rs); e=[r[2] for r in rs if r[2] is not None]
    pr=[sum(r[6][k] for r in rs)/len(rs) for k in range(9)]
    print("   gain by cell offset along motion (spacings) -4..4: "+" ".join(f"{v:+.4f}" for v in pr))
    print(f"   live activation centre shift along true motion, free step: {sum(r[7] for r in rs)/len(rs):+.3f} (true step {sum(r[3] for r in rs)/len(rs):.2f}); share of intervals shifting forward {sum(1 for r in rs if r[7]>0)/len(rs):.2f}")
    print(f"  {label}: n={len(rs)} gain ahead {A/(A+B+1e-300):.2f} behind {B/(A+B+1e-300):.2f} | gain point |to next| {sum(e)/max(1,len(e)):.2f} staying put {sum(r[3] for r in rs)/len(rs):.2f} | total gain {sum(r[5] for r in rs)/len(rs):.4f}")
print(f"{STREAM} L={L} NC={NC} {'STEP='+str(STEP) if ring else 'SPEEDS='+str(SPEEDS)} DIR={DIR} SHUF={SHUF}: Nethra built {len(f.nethra)-NC}")
rep(res,"all")
if not ring:
    rv=[r for r in res if r[4]]; 
    if rv: rep(rv,"reversal intervals")
