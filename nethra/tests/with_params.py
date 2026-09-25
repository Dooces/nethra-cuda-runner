"""usage: [LEAK=2] [COND=top|all|top_and_leaves] [DIR=shared|split] python3 with_params.py script.py args...  (runs a
nethra/tests script with a different default leakage / conduction / direction for every NethraField it creates)"""
import os, sys, runpy
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
T=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0,T)
import nethra
init=nethra.NethraField.__init__
LEAK=float(os.environ.get("LEAK","1")); COND=os.environ.get("COND"); DIR=os.environ.get("DIR")
def patched(self,*a,**k):
    k.setdefault("leakage",LEAK)
    if COND and "top_only_conduction" not in k: k.setdefault("conduction",COND)
    if DIR: k.setdefault("direction",DIR)
    init(self,*a,**k)
nethra.NethraField.__init__=patched
script=os.path.join(T,sys.argv[1]); sys.argv=[script]+sys.argv[2:]
runpy.run_path(script, run_name="__main__")
