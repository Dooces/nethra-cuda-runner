"""Run independent Nethra jobs in parallel processes, one thread each (GIL-free; one field per process).
usage: python3 par.py JOBS_FILE [WORKERS]     JOBS_FILE: one shell command per line.  Output: JOBS_FILE.out/
Each field is serial by construction (interval t needs t-1), so the parallel unit is a run: seeds, conditions,
parameter points.  Scales with cores; a 1-core machine gets nothing from it."""
import sys, os, subprocess, concurrent.futures as cf
jobs = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
W = int(sys.argv[2]) if len(sys.argv) > 2 else (os.cpu_count() or 1)
out = sys.argv[1] + ".out"; os.makedirs(out, exist_ok=True)
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
def run(i):
    with open(os.path.join(out, f"{i:04d}.txt"), "w") as fh:
        fh.write(f"$ {jobs[i]}\n"); fh.flush()
        r = subprocess.run(jobs[i], shell=True, stdout=fh, stderr=subprocess.STDOUT, env=env)
    return i, r.returncode
with cf.ProcessPoolExecutor(W) as ex:
    for i, rc in ex.map(run, range(len(jobs))): print(f"job {i} exit {rc}: {jobs[i]}", flush=True)
