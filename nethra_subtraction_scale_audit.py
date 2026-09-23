#!/usr/bin/env python3
"""Scale check for earned-structure-only subtraction on wider IID alphabets."""
from __future__ import annotations
import random,time
from nethra import NethraField
from nethra_subtraction_regression import OldSubtractionField

def run(cls,n_symbols,n_steps,seed):
    rng=random.Random(seed)
    f=cls(g_min=.20,g_max=1.50,tau=100.0,capacitance=1.0,leakage=.6,convergence_gain=0.0)
    leaves=[f.new() for _ in range(n_symbols)]
    t=time.perf_counter()
    for _ in range(n_steps):
        leaves[rng.randrange(n_symbols)].push(1.0)
        f.step(.05)
    return {
        "nethra":len(f.nethra),
        "relations":sum(bool(n.routes) for n in f.nethra),
        "history_keys":len(f.history_relation),
        "observed_pairs":len(f.history_count),
        "seconds":time.perf_counter()-t,
    }

def main():
    for n in (8,12):
        rows={}
        for label,cls in (("OLD",OldSubtractionField),("PATCHED",NethraField)):
            rows[label]=run(cls,n,500,7000+n)
            print("MODEL",n,label,rows[label],flush=True)
        ratio=rows["PATCHED"]["relations"]/max(1,rows["OLD"]["relations"])
        print("SCALE",n,rows,"relation_ratio",ratio)
        assert rows["PATCHED"]["relations"]<2000
        assert rows["PATCHED"]["seconds"]<55
    print("all_assertions_passed")
if __name__=="__main__":main()
