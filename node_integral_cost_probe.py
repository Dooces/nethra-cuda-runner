#!/usr/bin/env python3
"""Benchmark storing one activation integral per Nethra versus baseline RK4.

For conductance fixed over an interval:
    Phi_ij = integral(a_i-a_j)dt = A_i-A_j
    Q_ij = g_ij * Phi_ij

Thus per-node A_i should reconstruct every incidence charge without per-edge transient storage.
"""

import gc
import time
from statistics import median
from nethra import NethraField


def build(relations):
    f=NethraField(convergence_gain=0.0)
    base=[f.new() for _ in range(max(32,relations//4))]
    for i in range(relations):
        r=f.new()
        a=base[i%len(base)]
        b=base[(17*i+5)%len(base)]
        if a is b: b=base[(i+1)%len(base)]
        f._route(r,(a,b),frozenset(),5+(i%20))
    for i,n in enumerate(base[:8]): n.external=.1+.02*i
    return f


def snap(f): return {n:n.activation for n in f.nethra}
def restore(f,s):
    for n,v in s.items(): n.activation=v


def baseline(f,steps,dt):
    for _ in range(steps):
        a0={n:n.activation for n in f.nethra}
        k1=f._derivative_at(a0)
        a1={n:a0[n]+.5*dt*k1[n] for n in f.nethra}; k2=f._derivative_at(a1)
        a2={n:a0[n]+.5*dt*k2[n] for n in f.nethra}; k3=f._derivative_at(a2)
        a3={n:a0[n]+dt*k3[n] for n in f.nethra}; k4=f._derivative_at(a3)
        for n in f.nethra:
            n.activation=a0[n]+dt*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6


def node_integral(f,steps,dt):
    A={n:0.0 for n in f.nethra}
    for _ in range(steps):
        a0={n:n.activation for n in f.nethra}
        k1=f._derivative_at(a0)
        a1={n:a0[n]+.5*dt*k1[n] for n in f.nethra}; k2=f._derivative_at(a1)
        a2={n:a0[n]+.5*dt*k2[n] for n in f.nethra}; k3=f._derivative_at(a2)
        a3={n:a0[n]+dt*k3[n] for n in f.nethra}; k4=f._derivative_at(a3)
        for n in f.nethra:
            A[n]+=dt*(a0[n]+2*a1[n]+2*a2[n]+a3[n])/6
            n.activation=a0[n]+dt*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6
    return A


def timed(fn,f,initial,repeats=5):
    vals=[]; out=None; final=None
    for _ in range(repeats):
        restore(f,initial); gc.collect()
        t=time.perf_counter(); out=fn(f,30,.01); vals.append(time.perf_counter()-t)
        final=snap(f)
    return median(vals),out,final


def main():
    for relations in (100,500,1000,2000):
        f=build(relations)
        initial=snap(f)
        tb,_,fb=timed(baseline,f,initial)
        tn,A,fn=timed(node_integral,f,initial)
        maxdiff=max(abs(fb[n]-fn[n]) for n in f.nethra)

        # reconstruct all edge charges from A; this is the only representation needed afterward.
        qs=[g*(A[a]-A[b]) for a,b,g in f._edges()]
        assert maxdiff<1e-15
        assert len(qs)==len(f._edges())
        print({
            "relations":relations,
            "nethra":len(f.nethra),
            "edges":len(qs),
            "baseline_s":tb,
            "node_integral_s":tn,
            "ratio":tn/tb,
            "overhead_pct":100*(tn/tb-1),
            "stored_scalars":len(A),
            "edge_scalars_avoided":len(qs),
            "max_activation_diff":maxdiff,
        },flush=True)
    print("all_assertions_passed")


if __name__=="__main__":
    main()
