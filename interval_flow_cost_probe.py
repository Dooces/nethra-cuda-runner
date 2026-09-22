#!/usr/bin/env python3
"""Benchmark exact interval-current accumulation against unchanged RK4 field integration.

The integrated path records signed incidence charge using the same four RK4 stage states.
It must finish at the exact same activation state as the baseline path.  This benchmark measures
only the extra bookkeeping cost; it does not alter learning or topology.
"""

import gc
import resource
import time
from statistics import median

from nethra import NethraField


def build(relation_count):
    f=NethraField(convergence_gain=1.0)
    bases=[f.new() for _ in range(max(32, relation_count//4))]
    for i in range(relation_count):
        r=f.new()
        a=bases[i % len(bases)]
        b=bases[(i*17+7) % len(bases)]
        if a is b:
            b=bases[(i+1) % len(bases)]
        f._route(r,(a,b),frozenset(),5 + (i%20))
    for i,n in enumerate(bases[:8]):
        n.external=.1 + .03*i
    return f


def snapshot(f):
    return {n:n.activation for n in f.nethra}


def restore(f,s):
    for n,v in s.items(): n.activation=v


def rk4_baseline(f,steps,dt):
    for _ in range(steps):
        a0={n:n.activation for n in f.nethra}
        k1=f._derivative_at(a0)
        a1={n:a0[n]+.5*dt*k1[n] for n in f.nethra}; k2=f._derivative_at(a1)
        a2={n:a0[n]+.5*dt*k2[n] for n in f.nethra}; k3=f._derivative_at(a2)
        a3={n:a0[n]+dt*k3[n] for n in f.nethra}; k4=f._derivative_at(a3)
        for n in f.nethra:
            n.activation=a0[n]+dt*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.0


def rk4_integrated(f,steps,dt):
    es=f._edges()
    charge=[0.0]*len(es)

    def flows(state):
        return [g*(state[a]-state[b]) for a,b,g in es]

    for _ in range(steps):
        a0={n:n.activation for n in f.nethra}
        k1=f._derivative_at(a0)
        a1={n:a0[n]+.5*dt*k1[n] for n in f.nethra}; k2=f._derivative_at(a1)
        a2={n:a0[n]+.5*dt*k2[n] for n in f.nethra}; k3=f._derivative_at(a2)
        a3={n:a0[n]+dt*k3[n] for n in f.nethra}; k4=f._derivative_at(a3)

        q0,q1,q2,q3=flows(a0),flows(a1),flows(a2),flows(a3)
        for i in range(len(es)):
            charge[i]+=dt*(q0[i]+2*q1[i]+2*q2[i]+q3[i])/6.0

        for n in f.nethra:
            n.activation=a0[n]+dt*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.0
    return charge


def timed(fn,f,initial,steps,dt,repeats=3):
    vals=[]
    result=None
    final=None
    for _ in range(repeats):
        restore(f,initial)
        gc.collect()
        t=time.perf_counter()
        result=fn(f,steps,dt)
        vals.append(time.perf_counter()-t)
        final=snapshot(f)
    return median(vals),result,final


def main():
    steps=30
    dt=.01
    for relations in (100,500,1000,2000):
        f=build(relations)
        initial=snapshot(f)

        tb,_,fb=timed(rk4_baseline,f,initial,steps,dt)
        ti,charge,fi=timed(rk4_integrated,f,initial,steps,dt)

        maxdiff=max(abs(fb[n]-fi[n]) for n in f.nethra)
        assert maxdiff < 1e-15
        assert len(charge)==len(f._edges())

        print({
            "relations":relations,
            "nethra":len(f.nethra),
            "edges":len(f._edges()),
            "steps":steps,
            "baseline_s":tb,
            "integrated_s":ti,
            "ratio":ti/tb,
            "max_activation_diff":maxdiff,
            "charge_values":len(charge),
            "rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0,
        },flush=True)
    print("all_assertions_passed")


if __name__=="__main__":
    main()
