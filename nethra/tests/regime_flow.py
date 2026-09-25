"""At a read (after ctx+B), decompose prior flow P (completed interval integrals) into each outcome O by
relation, and what feeds those relations.  env READ=block index to dump, DIR."""
import os,sys,random,itertools
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import nethra as core, direction_variants; direction_variants.install(core)
E=os.environ.get; P=int(E("P",4)); k=int(E("K",6)); DIR=E("DIR","shared"); BLK=int(E("BLK",40)); READ=int(E("READ",191))
pairs=list(itertools.combinations(range(P),2)); rng=random.Random(7); rng.shuffle(pairs); ctx=pairs[:k]
A,B=P,P+1
f=core.NethraField(direction=DIR); L=[f.new() for _ in range(P+2+k)]
names=[f"f{i}" for i in range(P)]+["A","B"]+[f"O{j}" for j in range(k)]
def nm(n):
    i=f.nethra.index(n); return names[i] if i<len(names) else f"N{i}"
def show(xs):
    for x in xs: L[x].push(1.0)
    f.step(1.0)
def flows_into(target, Aint, ph):
    out=[]
    for (r,m),row in ph.items():
        if m is target: d=Aint.get(r,0)-Aint.get(m,0); g=row["g"] if (d>0 or DIR=="shared") else row["g_in"]; src=r
        elif r is target: d=Aint.get(m,0)-Aint.get(r,0); g=row.get("g_in",row["g"]) if (d>0 or DIR=="shared") else row["g"]; src=m
        else: continue
        q=g*d
        if q>0: out.append((q,src,g,Aint.get(src,0)))
    return sorted(out,key=lambda x:-x[0])
for blk in range(BLK*k):
    c=rng.randrange(k)
    for s in (A,B,P+2+c):
        show(list(ctx[c])+[s])
        if s==B and READ<0 and blk>=(BLK*3//4)*k:
            acts=[L[P+2+j].activation for j in range(k)]
            if acts[c]!=max(acts): print("wrong", blk, "right", c, "read", acts.index(max(acts)))
        if s==B and blk==READ:
            Aint=f.current_interval_integral; ph=f._physical_incidences(f.current_event)
            acts=[L[P+2+j].activation for j in range(k)]
            print(f"DIR={DIR} block {blk} ctx {[names[i] for i in ctx[c]]} right O{c}; O activations "+" ".join(f"O{j}:{a:.4f}" for j,a in enumerate(acts)))
            for j in sorted(range(k),key=lambda j:-acts[j])[:3]:
                fl=flows_into(L[P+2+j],Aint,ph)
                print(f"  into O{j} (ctx {[names[i] for i in ctx[j]]}) P {sum(q for q,*_ in fl):.4f}: "+", ".join(f"{nm(s)} {q:.4f} (g {g:.2f}, A {a:.3f})" for q,s,g,a in fl[:4]))
                for q,s,g,a in fl[:2]:
                    fi=flows_into(s,Aint,ph)
                    print(f"      into {nm(s)}: "+", ".join(f"{nm(x)} {qq:.4f} (g {gg:.2f})" for qq,x,gg,aa in fi[:6]))
            sys.exit()
