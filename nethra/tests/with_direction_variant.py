"""usage: VAR=op [DIR=split] python3 with_direction_variant.py script.py args  (a nethra/tests script with
the direction_variants.py evidence change installed; DIR as in with_params.py)"""
import os, sys, runpy
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
T=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.dirname(T)); sys.path.insert(0,T)
import nethra, direction_variants
direction_variants.install(nethra)
runpy.run_path(os.path.join(T,"with_params.py"), run_name="__main__")
