#!/usr/bin/env python3
"""One-pass current-semantics AAPL depth probe for hash-seed determinism.

Uses NETHRA_PRICE_CACHE supplied by workflow so every process receives identical price bytes.
No prediction audit. Current NativeReplay only.
"""
import json,time
import aapl_residual_recursive_native as base
from aapl_residual_recursive_native import NativeReplay,fetch_aapl,intervals

TRAIN=20000

def main():
    base.FAST_SYNC=True
    pts=fetch_aapl();cur,el,k,r=intervals(pts)
    m=NativeReplay();m.reset_transient()
    t=time.perf_counter();mc=0
    for i in range(TRAIN-1):
        out=m.interval(float(cur[i]),float(el[i]),True,True)
        mc=max(mc,out["closure_size"])
    mature=m.maturity()
    print("HASH_RESULT",json.dumps({
      "relations":len(m.birth_members),"nethra":len(m.f.nethra),
      "incidences":len(m.incidence_e),
      "depth":max(m.depth.values(),default=0),
      "mature_depth":max((m.depth[x] for x in mature),default=0),
      "max_closure":mc,"created":m.created,"accounted":m.accounted,"reused":m.reused,
      "seconds":time.perf_counter()-t
    },sort_keys=True),flush=True)

if __name__=="__main__":main()
