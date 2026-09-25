"""Closed sensorimotor loop.  World: object bounces on 0..L-1 at speed 1.  Eye at e.  Each interval:
push retina Nethra of offset o = x - e (one per offset, |o| <= R), efference Nethra of the eye's
movement this interval (EFF), eye-position Nethra (PROP); step.  Device (fixed, identical in every
condition): reads live activation of the retina Nethra, c = sum a_o o / sum a_o (positive a only);
next movement = sign(c) if |c| > DEAD else 0.  The field is never told the target, the movement
or the offset.  Expose EXPOSE intervals closed-loop, then TEST intervals from the same world state,
evidence change off, under conditions:
  exposed       - the exposed field
  lesion        - exposed field, every constructed Nethra's incidence evidence set to 0
  noconstruct   - a field that ran the same loop with construction and evidence change off
  shuffled  - a field exposed closed-loop while the object jumped to random positions
Read: mean |o| over the test (idealized: reflex on current offset ~2.5, perfect one-step expectation 1.0)."""
import os,sys,math,random,time
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import nethra as core
E=os.environ.get
L=int(E("L",10)); R=int(E("R",4)); DIR=E("DIR","split"); EFF=E("EFF","1")=="1"; PROP=E("PROP","1")=="1"
EXPOSE=int(E("EXPOSE",300)); TEST=int(E("TEST",100)); DEAD=float(E("DEAD",0.02)); SEED=int(E("SEED",1))
class World:
    def __init__(s, mode="bounce", seed=0): s.x=0; s.v=1; s.e=0; s.m=0; s.mode=mode; s.rng=random.Random(seed)
    def advance(s):
        s.e+=s.m; s.e=max(0,min(L-1,s.e))
        if s.mode=="bounce":
            x=s.x+s.v
            if x<0 or x>L-1: s.v=-s.v; x=s.x+s.v
            s.x=x
        else: s.x=s.rng.randrange(L)
def make_field(evid=True):
    f=core.NethraField(direction=DIR, topology_and_evidence_change=evid)
    ret={o:f.new() for o in range(-R,R+1)}; mot={-1:f.new(),1:f.new()}; pos=[f.new() for _ in range(L)]
    return f,ret,mot,pos
def handles(f):
    n=f.nethra; ret={o:n[i] for i,o in enumerate(range(-R,R+1))}; k=2*R+1
    return ret,{-1:n[k],1:n[k+1]},n[k+2:k+2+L]
def interval(f,w,ret,mot,pos):
    o=w.x-w.e
    if abs(o)<=R: ret[o].push(1.0)
    if EFF and w.m: mot[w.m].push(1.0)
    if PROP: pos[w.e].push(1.0)
    f.step(1.0)
    num=den=0.0
    for oo,n in ret.items():
        a=n.activation
        if a>0: num+=a*oo; den+=a
    c=num/den if den>0 else 0.0
    w.m = (1 if c>DEAD else -1 if c<-DEAD else 0)
    w.advance()
    return abs(o)
def expose(mode, evid=True, seed=0):
    f,ret,mot,pos=make_field(evid); w=World(mode,seed); errs=[]
    for t in range(EXPOSE): errs.append(interval(f,w,ret,mot,pos))
    return f,w,errs
def test(f,w,lesion=False):
    g=core.NethraField.from_checkpoint_dict(f.checkpoint_dict()); g.topology_and_evidence_change=False
    if lesion:
        for store in (g.incidence_evidence, g.incidence_evidence_in):
            for key,b in store.items():
                for s in b: b[s]=0.0
        for n in g.nethra:
            for r,b in n.routes.items():
                for s in b: b[s]=0.0
        g._incidence_plan={}
    ret,mot,pos=handles(g)
    ww=World(); ww.__dict__.update({k:v for k,v in w.__dict__.items() if k!="rng"}); ww.mode="bounce"
    errs=[interval(g,ww,ret,mot,pos) for _ in range(TEST)]
    return sum(errs)/len(errs), errs
t0=time.perf_counter()
f,w,tr=expose("bounce")
fn,wn,_=expose("bounce",evid=False)
fs,ws,_=expose("random",seed=SEED)
print(f"L={L} R={R} DIR={DIR} EFF={EFF} PROP={PROP} EXPOSE={EXPOSE} TEST={TEST}: Nethra built {len(f.nethra)-(2*R+3+L)}; exposure |o| first/last 50: {sum(tr[:50])/50:.2f} / {sum(tr[-50:])/50:.2f}")
for name,(ff,ww,les) in {"exposed":(f,w,False),"lesion":(f,w,True),"noconstruct":(fn,w,False),"shuffled":(fs,w,False)}.items():
    m,errs=test(ff,ww,les)
    print(f"  {name:12s} test mean |o| {m:.2f}   first 30: {''.join(str(min(9,e)) for e in errs[:30])}")
print(f"  ({time.perf_counter()-t0:.0f}s)")
