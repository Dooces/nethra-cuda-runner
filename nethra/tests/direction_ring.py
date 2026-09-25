"""Symbolic ring, one Nethra per position.  Reads last lap: directed conductances of N_(k+1)
(before member p_k, after member p_(k+1)); P toward p(k-1), p(k+1) (directed P); input-free
continuation (copy, evidence change off, step once with nothing pushed): activation of p(k+1) vs p(k-1)."""
import os,sys,math
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import nethra as core
K=int(os.environ.get("K",6)); LAPS=int(os.environ.get("LAPS",8)); ORDER=os.environ.get("ORDER","fwd")
def P_toward(f):
    A=f.current_interval_integral; out={}
    for (rel,mem),row in f._physical_incidences(f.current_event).items():
        d=A.get(rel,0.0)-A.get(mem,0.0)
        g=row["g"] if (d>0 or f.direction=="shared") else row["g_in"]
        q=g*d
        if q>0: out[mem]=out.get(mem,0.0)+q
    return out
def run(direction):
    f=core.NethraField(direction=direction); cells=[f.new() for _ in range(K)]
    import random; rng=random.Random(1)
    seq=list(range(K))
    built=[]; rows=[]
    for lap in range(LAPS):
        b0=len(f.nethra)
        order=seq if ORDER=="fwd" else (rng.sample(seq,K) if ORDER=="shuffle" else seq)
        for k in order:
            cells[k].push(1.0); f.step(1.0)
            if lap==LAPS-1 and ORDER=="fwd":
                P=P_toward(f); ahead=P.get(cells[(k+1)%K],0); behind=P.get(cells[(k-1)%K],0)
                g=core.NethraField.from_checkpoint_dict(f.checkpoint_dict()); g.topology_and_evidence_change=False
                now=[c.activation for c in g.nethra[:K]]; g.step(1.0)
                ex=[g.nethra[i].activation-now[i]*math.exp(-1) for i in range(K)]
                rows.append((k,ahead,behind,ex[(k+1)%K],ex[(k-1)%K]))
        built.append(len(f.nethra)-b0)
    return f,cells,built,rows
for direction in ("shared","split"):
    f,cells,built,rows=run(direction)
    name=lambda n: f"p{cells.index(n)}" if n in cells else f"N{f.nethra.index(n)-K+1}"
    print(f"== {direction} ORDER={ORDER} built/lap {built}")
    ph=f._physical_incidences(f.current_event)
    for n in f.nethra[K:]:
        rs=" | ".join("{"+",".join(sorted(map(name,r)))+"}" for r in n.routes)
        gs=" ".join(f"{name(m)}:{row['g']:.2f}/{row.get('g_in',row['g']):.2f}" for (r,m),row in ph.items() if r is n)
        print(f"  {name(n)} {rs}   g rel->mem/mem->rel {gs}")
    if rows:
        print("  k | P->p(k+1) P->p(k-1) | free-run gain p(k+1) p(k-1)")
        for k,a,b,ea,eb in rows: print(f"  {k} | {a:.4f} {b:.4f} | {ea:+.4f} {eb:+.4f}")
