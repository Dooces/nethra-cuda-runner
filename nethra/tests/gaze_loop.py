"""Closed sensorimotor loop, reflex-taught, then field-driven.
World: object bounces on 0..L-1 at speed 1.  Eye at e (clamped to 0..L-1).
Nethra: retina (one per offset o = x - e, |o| <= R), motor command mL/mR, eye position (one per e).
Interval t: the eye moves by m_t during the interval.  Pushed: retina o_t, eye position e_t, and (only
while the reflex drives) the motor command m_t.  Then e += m_t, object moves.
Learning phase (TRAIN intervals): an innate reflex drives: m_t = sign(o_(t-LAT)).
Test phase (TEST intervals, reflex off): nothing pushes the motor Nethra; the actuator reads their live
activation after interval t-1:  m_t = sign(a_R - a_L) if |a_R - a_L| > DEAD else 0.
Conditions at test (same world state, same actuator):
  trained      - the field after the learning phase
  lesion       - same field, every constructed Nethra's incidence evidence 0
  shuffletrain - learning phase with the object at random positions (reflex still drives)
  noconstruct  - learning phase with construction and evidence change off
Reads: test mean |o|; teacher (reflex) mean |o| on the same world; staying put; share of test intervals the
eye moves; reversal timing (interval of eye reversal minus interval of object reversal)."""
import os,sys,math,random,time
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import nethra as core
E=os.environ.get
L=int(E("L",10)); R=int(E("R",4)); DIR=E("DIR","split"); LAT=int(E("LAT",2))
TRAIN=int(E("TRAIN",300)); TEST=int(E("TEST",80)); DEAD=float(E("DEAD",0.002)); LEARN=E("LEARN","0")=="1"; PROP=E("PROP","1")=="1"
CONDS=E("CONDS","trained,lesion,shuffletrain,noconstruct").split(",")
class World:
    def __init__(s, mode="bounce", seed=0): s.x=0; s.v=1; s.e=0; s.mode=mode; s.rng=random.Random(seed); s.hist=[]
    def advance(s, m):
        s.e=max(0,min(L-1,s.e+m))
        if s.mode=="bounce":
            x=s.x+s.v
            if x<0 or x>L-1: s.v=-s.v; x=s.x+s.v
            s.x=x
        else: s.x=s.rng.randrange(L)
def make_field(evid=True):
    f=core.NethraField(direction=DIR, topology_and_evidence_change=evid)
    for _ in range(2*R+1+2+L): f.new()
    return f
def handles(f):
    n=f.nethra; k=2*R+1
    return {o:n[i] for i,o in enumerate(range(-R,R+1))},{-1:n[k],1:n[k+1]},n[k+2:k+2+L]
def sgn(v,dead=0.0): return 1 if v>dead else -1 if v<-dead else 0
def run(f, w, n, driver, push_motor):
    ret,mot,pos=handles(f); log=[]
    for _ in range(n):
        o=w.x-w.e
        if driver=="reflex":
            past=w.hist[-LAT] if len(w.hist)>=LAT else 0
            m=sgn(past)
        else:
            d=mot[1].activation-mot[-1].activation
            m=sgn(d,DEAD)
        if abs(o)<=R: ret[o].push(1.0)
        if PROP: pos[w.e].push(1.0)
        if push_motor and m: mot[m].push(1.0)
        f.step(1.0)
        w.hist.append(o); log.append((w.x,w.e,o,m,w.v))
        w.advance(m)
    return log
def lesion(g):
    for store in (g.incidence_evidence, g.incidence_evidence_in):
        for b in store.values():
            for s in b: b[s]=0.0
    for nn in g.nethra:
        for b in nn.routes.values():
            for s in b: b[s]=0.0
    g._incidence_plan={}
def clone_world(w):
    c=World(); c.x,c.v,c.e=w.x,w.v,w.e; c.hist=list(w.hist); return c
def summary(log):
    mo=sum(abs(r[2]) for r in log)/len(log); mv=sum(1 for r in log if r[3])/len(log)
    # reversal timing: for each object reversal, first later interval the eye moves in the new direction
    lags=[]
    for i in range(1,len(log)):
        if log[i][4]!=log[i-1][4]:
            for j in range(i,len(log)):
                if log[j][3]==log[i][4]: lags.append(j-i); break
    near=sum(1 for r in log if abs(r[2])<=1)/len(log)
    mv_=[r for r in log if r[3]]; match=sum(1 for r in mv_ if r[3]==r[4])/len(mv_) if mv_ else float("nan")
    return mo,mv,lags,near,match
t0=time.perf_counter()
trained={}
f=make_field(); w=World(); run(f,w,TRAIN,"reflex",True); trained["trained"]=(f,w)
if "shuffletrain" in CONDS:
    fs=make_field(); ws=World("random",7); run(fs,ws,TRAIN,"reflex",True); trained["shuffletrain"]=(fs,None)
if "noconstruct" in CONDS:
    fn=make_field(False); wn=World(); run(fn,wn,TRAIN,"reflex",True); trained["noconstruct"]=(fn,None)
print(f"L={L} R={R} DIR={DIR} LAT={LAT} TRAIN={TRAIN} TEST={TEST} DEAD={DEAD} LEARN={LEARN}: Nethra built {len(f.nethra)-(2*R+3+L)}")
tl=run(core.NethraField.from_checkpoint_dict(f.checkpoint_dict()),clone_world(w),TEST,"reflex",True)
mo,mv,lags,near,match=summary(tl); print(f"  {'teacher':12s} mean |o| {mo:.2f} |o|<=1 {near:.2f} moving {mv:.2f} moves with object {match:.2f} eye reversal lag {lags}")
xs=[r[0] for r in tl]; best=min(sum(abs(x-e) for x in xs)/len(xs) for e in range(L))
print(f"  {'best fixed eye':12s} mean |o| {best:.2f} |o|<=1 {max(sum(1 for x in xs if abs(x-e)<=1)/len(xs) for e in range(L)):.2f}")
for c in CONDS:
    base,_=trained["trained"] if c=="lesion" else trained[c]
    g=core.NethraField.from_checkpoint_dict(base.checkpoint_dict()); g.topology_and_evidence_change=LEARN
    if c=="lesion": lesion(g)
    log=run(g,clone_world(w),TEST,"field",False)
    mo,mv,lags,near,match=summary(log)
    print(f"  {c:12s} mean |o| {mo:.2f} |o|<=1 {near:.2f} moving {mv:.2f} moves with object {match:.2f} eye reversal lag {lags}  o: {''.join('+' if r[2]>0 else '-' if r[2]<0 else '0' for r in log[:30])}")
print(f"  ({time.perf_counter()-t0:.0f}s)")
