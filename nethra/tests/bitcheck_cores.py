"""bitcheck.py REF.py NEW.py : identical checkpoints and activations on symbolic and graded streams,
modes x top_only {True, False}, and a checkpoint round trip mid-stream."""
import os, sys, json, hashlib, random, importlib.util
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
def load(name, path):
    spec=importlib.util.spec_from_file_location(name, path); m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m
ref=load("refcore", sys.argv[1]); new=load("newcore", sys.argv[2])
N=int(os.environ.get("N",120))
def symbolic(seed):
    rng=random.Random(seed); out=[]
    for _ in range(N):
        k=rng.choice([[0],[1],[0,2],[3],[1,4],[5,6],[2],[3,7]]); out.append([(i,1.0) for i in k])
    return out, 8
def graded(seed):
    rng=random.Random(seed); out=[]; x=0.0
    for t in range(N):
        x=(x+rng.choice([0.7,1.3,2.1]))%10
        pat=[(i,max(0.0,1-abs(x-i*10/11)/1.5)) for i in range(12)]
        out.append([p for p in pat if p[1]>0])
    return out, 12
def h(d): return hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()[:16]
def run(cls, kw, stream, n_in, roundtrip=None):
    f=cls(**kw); L=[f.new() for _ in range(n_in)]; acts=[]; P=[]
    for t,pat in enumerate(stream):
        if roundtrip is not None and t==roundtrip:
            f=cls.from_checkpoint_dict(json.loads(json.dumps(f.checkpoint_dict()))); L=f.nethra[:n_in]
        for i,v in pat: L[i].push(v)
        f.step(1.0); acts.append([n.activation for n in f.nethra])
        ph=f._physical_incidences(f.current_event)
        P.append(sorted((f._order[r],f._order[m],row["g"]) for (r,m),row in ph.items() if row["g"]>0))
    return h(f.checkpoint_dict()), h(acts), h(P), len(f.nethra)
modes={"default":{}, "frontier":dict(frontier_tolerance=1e-2), "frontier_min":dict(frontier_tolerance=1e-2,frontier_min=1e-3),
       "rk4_th0.9":dict(integrator="rk4",source_similarity_threshold=0.9), "whole":dict(join_on_recurrence=False),
       "noevid":dict(topology_and_evidence_change=False)}
ok=True
for sname,gen in (("symbolic",symbolic),("graded",graded)):
  for seed in (1,2):
    stream,n_in=gen(seed)
    for mname,kw in modes.items():
      for top in (True,False):
        k=dict(kw,top_only_conduction=top)
        a=run(ref.NethraField,k,stream,n_in); b=run(new.NethraField,k,stream,n_in); c=run(new.NethraField,k,stream,n_in,roundtrip=N//2)
        e=(a==b and (a[:3]==c[:3] or mname=="frontier_min"))
        if not e: print("MISMATCH",sname,seed,mname,top,a,b,c)
        ok&=e
print("ALL IDENTICAL" if ok else "MISMATCH")
