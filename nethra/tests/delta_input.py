"""Delta input test.  One object bouncing on a line of positions 0..L-1 with speed from SPEEDS
(cycled per run end-to-end).  Position: NC tent cells over [0, L-1].  Displacement (DELTA=1): ND
tent cells over [-VMAX, VMAX], pushed with their shares of (x_t - x_{t-1}) in the same interval.
Reads over the last LASTN intervals: built per pass, constructed Nethra refound per interval, structural
next (cells of after routes of Nethra refound by their first route) as a position vs the true next,
P split behind / at / ahead along the motion, P share toward displacement cells, ms/interval."""
import os, sys, math, time, importlib.util
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
spec=importlib.util.spec_from_file_location("nethra",os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"nethra.py"))
core=importlib.util.module_from_spec(spec); sys.modules["nethra"]=core; spec.loader.exec_module(core)
E=os.environ.get
L=int(E("L",8)); NC=int(E("NC",L)); DELTA=E("DELTA","1")=="1"; ND=int(E("ND",5)); VMAX=float(E("VMAX",2))
SPEEDS=[int(s) for s in E("SPEEDS","1").split(",")]; PASSES=int(E("PASSES",8)); TOP=E("TOP","1")=="1"
LEAK=float(E("LEAK",1)); TH=float(E("TH",0.999)); TOL=float(E("TOL",0)); DUMP=E("DUMP","0")=="1"
sp=(L-1)/(NC-1)
def tents(v, n, lo, hi):
    s=(hi-lo)/(n-1); out={}
    for i in range(n):
        w=1-abs(v-(lo+i*s))/s
        if w>1e-9: out[i]=w
    return out
# stream: bounce, speed changes each pass
xs=[0]; p=0
for k in range(PASSES):
    v=SPEEDS[k%len(SPEEDS)]*(1 if k%2==0 else -1)
    target=L-1 if v>0 else 0
    x=xs[-1]
    while x!=target:
        x=max(0,min(L-1,x+v)); xs.append(x)
f=core.NethraField(top_only_conduction=TOP, leakage=LEAK, source_similarity_threshold=TH, frontier_tolerance=TOL)
pc=[f.new() for _ in range(NC)]; dc=[f.new() for _ in range(ND)] if DELTA else []
base=NC+len(dc)
def nm(n):
    if n in pc: return f"x{pc.index(n)}"
    if n in dc: return f"d{dc.index(n)}"
    return f"N{f.nethra.index(n)-base+1}"
def P_toward(f):
    A=f.current_interval_integral; out={}
    for (rel,mem),row in f._physical_incidences(f.current_event).items():
        q=row["g"]*(A.get(rel,0.0)-A.get(mem,0.0))
        if q>0: out[mem]=out.get(mem,0.0)+q
    return out
LASTN=int(E("LASTN", 2*(L-1)))
built=[]; b0=0; stats=[]; t0=time.perf_counter(); tlast=None
for t,x in enumerate(xs):
    for i,w in tents(x,NC,0,L-1).items(): pc[i].push(w)
    if DELTA and t>0:
        for i,w in tents(x-xs[t-1],ND,-VMAX,VMAX).items(): dc[i].push(w)
    f.step(1.0)
    if t>0 and (x in (0,L-1)):
        built.append(len(f.nethra)-base-b0); b0=len(f.nethra)-base
    if t>=len(xs)-1-LASTN and t<len(xs)-1:
        if tlast is None: tlast=time.perf_counter(); tl0=t
        xn=xs[t+1]; direction=1 if xn>x else -1
        clo=f.closure(f.previous_explicit,f.current_source_event)
        cons=[n for n in clo if n not in pc and n not in dc]
        nxt=set()
        for n in cons:
            rs=list(n.routes)
            if rs and rs[0]<=clo:
                for r in rs[1:]: nxt|={pc.index(m) for m in r if m in pc}
        sn=sum(i*sp for i in nxt)/len(nxt) if nxt else None
        P=P_toward(f)
        split={"behind":0.0,"at":0.0,"ahead":0.0}
        for i,c in enumerate(pc):
            off=(i*sp-x)*direction
            k="at" if abs(off)<sp*0.999 else ("ahead" if off>0 else "behind")
            split[k]+=P.get(c,0.0)
        pd=sum(P.get(c,0.0) for c in dc); pall=sum(P.values()) or 1.0
        stats.append((len(cons), None if sn is None else abs(sn-xn), abs(x-xn), split, pd/pall, len(nxt)))
tl=(time.perf_counter()-tlast)/max(1,len(xs)-1-tl0)*1000 if tlast else 0
S={k:sum(s[3][k] for s in stats) for k in ("behind","at","ahead")}; ST=sum(S.values()) or 1
se=[s[1] for s in stats if s[1] is not None]
print(f"L={L} NC={NC} delta={DELTA} ND={ND} speeds={SPEEDS} top={TOP} leak={LEAK} th={TH}: nethra {len(f.nethra)-base} built per pass {built}")
print(f"  last {len(stats)} intervals: refound constructed/interval {sum(s[0] for s in stats)/len(stats):.1f}; structural next |err| {sum(se)/max(1,len(se)):.2f} ({len(se)}/{len(stats)}, cells in set {sum(s[5] for s in stats)/len(stats):.1f}); staying put {sum(s[2] for s in stats)/len(stats):.2f}")
print(f"  P toward position cells behind {S['behind']/ST:.2f} at {S['at']/ST:.2f} ahead {S['ahead']/ST:.2f}; share of all P toward displacement cells {sum(s[4] for s in stats)/len(stats):.2f}; ms/interval {tl:.1f}")
if DUMP:
    for n in f.nethra[base:]:
        print("   ",nm(n)," | ".join("{"+",".join(sorted(map(nm,r)))+"}" for r in n.routes))
