import os, sys, runpy
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
T=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.dirname(T)); sys.path.insert(0,T)
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import nethra, conduction_variants as cond; cond.install(nethra)
script=sys.argv[1] if os.path.isabs(sys.argv[1]) else os.path.join(T,sys.argv[1]); sys.argv=[script]+sys.argv[2:]
runpy.run_path(script, run_name="__main__")
