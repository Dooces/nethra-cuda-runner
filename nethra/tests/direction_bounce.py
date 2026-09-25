"""Symbolic bounce: positions 0..L-1, one Nethra each, pushed 1.0; object bounces 0->L-1->0...
Topology dump; then over the last 2 passes: free step (copy, nothing pushed) gain at x+1 vs x-1
along the true motion (ahead vs behind), and at walls (reversal)."""
import os,sys,math
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import nethra as core
E=os.environ.get; L=int(E("L",5)); PASSES=int(E("PASSES",12)); DIR=E("DIR","shared"); DUMP=E("DUMP","1")=="1"
COND=E("COND","top_and_leaves")
xs=[0]
for k in range(PASSES):
    tgt=L-1 if k%2==0 else 0
    while xs[-1]!=tgt: xs.append(xs[-1]+(1 if tgt>xs[-1] else -1))
f=core.NethraField(direction=DIR, conduction=COND); p=[f.new() for _ in range(L)]
nm=lambda n: f"p{p.index(n)}" if n in p else f"N{f.nethra.index(n)-L+1}"
T=len(xs)-1; READ=2*(L-1); rows=[]
built=[]
for t in range(T):
    b=len(f.nethra)
    p[xs[t]].push(1.0); f.step(1.0)
    built.append(len(f.nethra)-b)
    if t>=T-READ:
        g=core.NethraField.from_checkpoint_dict(f.checkpoint_dict()); g.topology_and_evidence_change=False
        now=[c.activation for c in g.nethra[:L]]; g.step(1.0)
        gain=[g.nethra[i].activation-now[i]*math.exp(-1) for i in range(L)]
        x,xn=xs[t],xs[t+1]; back=x-(xn-x)
        ah=gain[xn]; bh=gain[back] if 0<=back<L else None
        rows.append((t,x,xn,ah,bh))
print(f"L={L} DIR={DIR} COND={COND} built per interval: {''.join(str(b) for b in built)}")
if DUMP:
    ph=f._physical_incidences(f.current_event)
    for n in f.nethra[L:]:
        rs=" | ".join("{"+",".join(sorted(map(nm,r)))+"}" for r in n.routes)
        gs=" ".join(f"{nm(m)}:{row['g']:.2f}/{row.get('g_in',row['g']):.2f}" for (r,m),row in ph.items() if r is n)
        print(f"  {nm(n)} {rs}   {gs}")
print("  t x->next | free-step gain at next, at the cell behind")
fw=0; nb=0
for t,x,xn,ah,bh in rows:
    print(f"  {t} {x}->{xn} | {ah:+.4f} {'' if bh is None else f'{bh:+.4f}'}")
    if bh is not None: nb+=1; fw+= ah>bh
print(f"  intervals with gain(next) > gain(behind): {fw}/{nb}")
