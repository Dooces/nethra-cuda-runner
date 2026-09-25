"""Tiny exact loop: one object on a ring of K positions, one Nethra per position (symbolic) or
tent cells (graded).  Reads per interval after settling: closure, structural next, P toward the
position cells relative to the current one, live activation."""
import os, sys, math, importlib.util
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
spec=importlib.util.spec_from_file_location("nethra",os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"nethra.py"))
core=importlib.util.module_from_spec(spec); sys.modules["nethra"]=core; spec.loader.exec_module(core)
K=int(os.environ.get("K",6)); LAPS=int(os.environ.get("LAPS",6)); TOP=os.environ.get("TOP","1")=="1"
LEAK=float(os.environ.get("LEAK",1)); VERBOSE=os.environ.get("V","0")=="1"

def P_toward(f):
    A=f.current_interval_integral; out={}
    for (rel,mem),row in f._physical_incidences(f.current_event).items():
        q=row["g"]*(A.get(rel,0.0)-A.get(mem,0.0))
        if q>0: out[mem]=out.get(mem,0.0)+q
    return out

f=core.NethraField(top_only_conduction=TOP, leakage=LEAK)
cells=[f.new() for _ in range(K)]
name=lambda n: f"p{cells.index(n)}" if n in cells else f"N{f.nethra.index(n)-K+1}"
built=[]
rows=[]
for lap in range(LAPS):
    b0=len(f.nethra)
    for k in range(K):
        cells[k].push(1.0); f.step(1.0)
        if lap>=LAPS-2:
            P=P_toward(f); tot=sum(P.values()) or 1.0
            rel=[P.get(cells[(k+d)%K],0.0) for d in (-2,-1,0,1,2)]
            clo=f.closure(f.previous_explicit, f.current_source_event)
            # structural next: after routes (non-first routes) of Nethra refound by a before (first) route
            nxt=set()
            for n in clo:
                if n in cells or not n.routes: continue
                rs=list(n.routes)
                if rs[0]<=clo:
                    for r in rs[1:]: nxt|={m for m in r if m in cells}
            act=[cells[(k+d)%K].activation for d in (-2,-1,0,1,2)]
            rows.append((lap,k,rel,tot,sorted(name(n) for n in clo),sorted(name(n) for n in nxt),act))
    built.append(len(f.nethra)-b0)
print(f"K={K} top_only={TOP} leak={LEAK} built per lap {built}")
for n in f.nethra[K:]:
    print(" ", name(n), " | ".join("{"+",".join(sorted(map(name,r)))+"}" for r in n.routes))
if VERBOSE:
    for (rel,mem),row in sorted(f._physical_incidences(f.current_event).items(), key=lambda x:(f.nethra.index(x[0][0]),f.nethra.index(x[0][1]))):
        print(f"   g {name(rel)}-{name(mem)} {row['g']:.3f}")
print("lap k | P toward p(k-2) p(k-1) p(k) p(k+1) p(k+2) | P total all | closure | structural next | live act k-2..k+2")
for lap,k,rel,tot,clo,nxt,act in rows:
    print(f"{lap} {k} | "+" ".join(f"{x:.4f}" for x in rel)+f" | {tot:.4f} | {','.join(clo)} | {','.join(nxt)} | "+" ".join(f"{x:.3f}" for x in act))
