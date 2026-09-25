"""Tiny exact loop, graded: one object on a ring of L integer positions, NC tent cells (spacing
L/NC, width = spacing).  Reads per interval in the last 2 laps: closure (constructed only), the
cells in the after routes of Nethra refound by their first route (structural next), and P toward
cells as a position (circular centroid) vs the true next position and staying put."""
import os, sys, math, importlib.util, cmath
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
spec=importlib.util.spec_from_file_location("nethra",os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"nethra.py"))
core=importlib.util.module_from_spec(spec); sys.modules["nethra"]=core; spec.loader.exec_module(core)
L=int(os.environ.get("L",12)); NC=int(os.environ.get("NC",8)); LAPS=int(os.environ.get("LAPS",5))
STEP=int(os.environ.get("STEP",1)); TOP=os.environ.get("TOP","1")=="1"; LEAK=float(os.environ.get("LEAK",1))
SHOW=int(os.environ.get("SHOW",L))
sp=L/NC
def weights(x):
    out={}
    for i in range(NC):
        d=abs(((x-i*sp)+L/2)%L-L/2)
        w=1-d/sp
        if w>1e-9: out[i]=w
    return out
def P_toward(f):
    A=f.current_interval_integral; out={}
    for (rel,mem),row in f._physical_incidences(f.current_event).items():
        q=row["g"]*(A.get(rel,0.0)-A.get(mem,0.0))
        if q>0: out[mem]=out.get(mem,0.0)+q
    return out
def circ(ws):  # circular centroid position of a cell->weight map
    z=sum(w*cmath.exp(2j*math.pi*(i*sp)/L) for i,w in ws.items())
    return (cmath.phase(z)/(2*math.pi)*L)%L if abs(z)>0 else None
def cd(a,b): return abs(((a-b)+L/2)%L-L/2)
f=core.NethraField(top_only_conduction=TOP, leakage=LEAK)
cells=[f.new() for _ in range(NC)]
nm=lambda n: f"c{cells.index(n)}" if n in cells else f"N{f.nethra.index(n)-NC+1}"
built=[]; rows=[]; errs=[]
positions=[(STEP*t)%L for t in range(L)]
for lap in range(LAPS):
    b0=len(f.nethra)
    for t,x in enumerate(positions):
        for i,w in weights(x).items(): cells[i].push(w)
        f.step(1.0)
        if lap>=LAPS-2:
            xn=positions[(t+1)%L]
            P=P_toward(f); Pc={cells.index(n):v for n,v in P.items() if n in cells}
            clo=f.closure(f.previous_explicit,f.current_source_event)
            nxt=set()
            for n in clo:
                if n in cells or not n.routes: continue
                rs=list(n.routes)
                if rs[0]<=clo:
                    for r in rs[1:]: nxt|={cells.index(m) for m in r if m in cells}
            pc=circ(Pc); sn=circ({i:1.0 for i in nxt}) if nxt else None
            errs.append((cd(pc,xn) if pc is not None else None, cd(sn,xn) if sn is not None else None, cd(x,xn)))
            if lap==LAPS-1 and t<SHOW:
                rows.append(f"x={x:2d} next={xn:2d} cells {sorted(weights(x))} | closure {len([n for n in clo if n not in cells])} constructed | struct next cells {sorted(nxt)} | P cells "+" ".join(f"c{i}:{v:.4f}" for i,v in sorted(Pc.items()))+f" | P point {pc if pc is None else round(pc,2)}")
    built.append(len(f.nethra)-b0)
print(f"L={L} NC={NC} STEP={STEP} top={TOP} leak={LEAK} built per lap {built}")
for r in rows: print(r)
ok=[e for e in errs if e[0] is not None]; ok2=[e for e in errs if e[1] is not None]
print(f"mean |P point - next| {sum(e[0] for e in ok)/max(1,len(ok)):.2f} ({len(ok)} of {len(errs)}), structural next {sum(e[1] for e in ok2)/max(1,len(ok2)):.2f} ({len(ok2)}), staying put {sum(e[2] for e in errs)/len(errs):.2f}")
