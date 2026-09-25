"""usage: prof.py CORE.py stream.json [profile]  -> ms/interval over the measure intervals, hash of result"""
import os, sys, json, time, hashlib, importlib.util
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
spec=importlib.util.spec_from_file_location("nethra",sys.argv[1]); core=importlib.util.module_from_spec(spec); sys.modules["nethra"]=core; spec.loader.exec_module(core)
d=json.load(open(sys.argv[2])); f=core.NethraField.from_checkpoint_dict(d["checkpoint"])
def run():
    acts=hashlib.sha256()
    for pl in d["measure"]:
        for i,v in pl: f.nethra[i].push(v)
        f.step(1.0)
        acts.update(repr([n.activation for n in f.nethra]).encode())
    return acts.hexdigest()[:16]
t0=time.perf_counter()
if len(sys.argv)>3:
    import cProfile, pstats; pr=cProfile.Profile(); pr.enable(); h=run(); pr.disable()
    pstats.Stats(pr).sort_stats("tottime").print_stats(18)
else: h=run()
dt=(time.perf_counter()-t0)/len(d["measure"])*1000
ck=hashlib.sha256(json.dumps(f.checkpoint_dict(),sort_keys=True).encode()).hexdigest()[:16]
print(f"ms/interval {dt:.1f}  nethra {len(f.nethra)}  frontier mean {sum(f.frontier_sizes)/len(f.frontier_sizes):.0f}  acts {h}  ckpt {ck}")
