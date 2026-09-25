"""parts.py CORE.py stream.json : ms/interval by part (wall clock wrappers, exclusive of nested wrapped parts)."""
import os, sys, json, time, importlib.util
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
spec=importlib.util.spec_from_file_location("nethra",sys.argv[1]); core=importlib.util.module_from_spec(spec); sys.modules["nethra"]=core; spec.loader.exec_module(core)
d=json.load(open(sys.argv[2])); f=core.NethraField.from_checkpoint_dict(d["checkpoint"])
C=core.NethraField; acc={}; stack=[]
def wrap(name):
    orig=getattr(C,name)
    def w(self,*a,**k):
        t0=time.perf_counter(); stack.append(0.0)
        try: return orig(self,*a,**k)
        finally:
            dt=time.perf_counter()-t0; inner=stack.pop()
            acc[name]=acc.get(name,0.0)+dt-inner
            if stack: stack[-1]+=dt
    setattr(C,name,w)
for n in ["step","closure","_canonical_source_event","_frontier","_physical_incidences","_edges","_compile_interval",
          "_etd_interval","_rk4_interval","_move_evidence_and_construct","update_residuals","_admit_by_parts","_admit_whole_support","_admit_sides"]:
    wrap(n)
t0=time.perf_counter()
for pl in d["measure"]:
    for i,v in pl: f.nethra[i].push(v)
    f.step(1.0)
T=(time.perf_counter()-t0)/len(d["measure"])*1000
print(f"total {T:.1f} ms/interval; by part (exclusive):")
for k,v in sorted(acc.items(), key=lambda x:-x[1]): print(f"  {k:30s} {v/len(d['measure'])*1000:6.2f}")
