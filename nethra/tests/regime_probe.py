"""Split(-neg) stream; at each read (ctx+B of last quarter) re-run that interval on copies:
 directed (as is) / symmetric with g from max, mean, min, out-only, in-only of the two evidences."""
import os,sys,random,itertools
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import nethra as core, direction_variants; direction_variants.install(core)
E=os.environ.get; P=int(E("P",5)); k=int(E("K",10)); BLK=int(E("BLK",30))
pairs=list(itertools.combinations(range(P),2)); rng=random.Random(7); rng.shuffle(pairs); ctx=pairs[:k]
A,B=P,P+1
f=core.NethraField(direction="split"); L=[f.new() for _ in range(P+2+k)]
modes=["directed","max","mean","min","out","in"]; ok={m:[] for m in modes}
def probe(cp, xs, mode):
    cp=dict(cp); cp["parameters"]=dict(cp["parameters"])
    if mode!="directed":
        cp["parameters"]["direction"]="shared"
        ins={(r["relation"],tuple(r["route"]),r["member"]):r for r in cp["incidence_evidence_in"]}
        new=[]
        for r in cp["incidence_evidence"]:
            ri=ins[(r["relation"],tuple(r["route"]),r["member"])]
            co={json_key(c):c for c in r["conditions"]}; ci={json_key(c):c for c in ri["conditions"]}
            rows=[]
            for key in set(co)|set(ci):
                a=val(co.get(key)); b=val(ci.get(key))
                v={"max":max(a,b),"mean":(a+b)/2,"min":min(a,b),"out":a,"in":b}[mode]
                rows.append(setval(co.get(key) or ci.get(key), v))
            new.append(dict(r,conditions=rows))
        cp["incidence_evidence"]=new; cp.pop("incidence_evidence_in")
        cp["parameters"]["integrator"]="rk4"
    g=core.NethraField.from_checkpoint_dict(cp); g.topology_and_evidence_change=False
    for x in xs: g.nethra[x].push(1.0)
    g.step(1.0); return [g.nethra[P+2+j].activation for j in range(k)]
import json
def json_key(c): return json.dumps(c["signature"])
def val(c): return 0.0 if c is None else float(c["evidence"])
def setval(c,v): c=dict(c); c["evidence"]=v; return c
for blk in range(BLK*k):
    c=rng.randrange(k)
    for s in (A,B,P+2+c):
        xs=list(ctx[c])+[s]
        if s==B and blk>=(BLK*3//4)*k:
            cp=f.checkpoint_dict()
            for m in modes:
                acts=probe(cp,xs,m); ok[m].append(acts[c]==max(acts))
        for x in xs: L[x].push(1.0)
        f.step(1.0)
print(f"P={P} K={k} VAR={E('VAR','')}: "+" ".join(f"{m} {sum(v)/len(v):.2f}" for m,v in ok.items()))
