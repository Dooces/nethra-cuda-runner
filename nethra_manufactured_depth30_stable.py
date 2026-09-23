#!/usr/bin/env python3
"""Stable numerical execution of the exact manufactured depth-30 current-Nethra benchmark.

Learning semantics, generated data, external interval duration (0.15), observation boundaries, and
provisional construction are identical to nethra_manufactured_depth30_series.py.

Only RK4 numerical resolution changes: one externally observed interval is internally subdivided
when the current weighted graph would make a single 0.15 RK4 step unstable. External current remains
constant through all substeps. _complete_interval() and
_consider_completed_interval_provisional() still execute exactly once per observed interval.

For convergence_gain == 0, cached_derivative() is algebraically identical to
NethraField._derivative_at() because topology/evidence/current_event are fixed until the interval
boundary. A startup assertion verifies this against the current core.
"""
from __future__ import annotations

import math
import random
import atexit

import nethra_manufactured_depth30_series as bench


class StableDepthAuditField(bench.DepthAuditField):
    max_substeps_seen=1
    max_weighted_degree_seen=0.0
    intervals_executed=0

    def cached_derivative(self,activation,edges):
        current={n:n.external-self.leakage*activation[n] for n in self.nethra}
        for a,b,g in edges:
            q=g*(activation[a]-activation[b])
            current[a]-=q
            current[b]+=q
        return {n:v/self.capacitance for n,v in current.items()}

    def step(self,dt=.1):
        self.interval_no+=1
        dt=float(dt)
        if dt<=0.0:
            raise ValueError("dt must be positive")
        if self.convergence_gain!=0.0:
            raise ValueError("stable cached executor requires convergence_gain == 0")

        source_current={n:n.external for n in self.nethra if n.external!=0.0}
        explicit=frozenset(source_current)
        initial={n:n.activation for n in self.nethra}
        state=dict(initial)
        edges=self._edges()

        degree={n:0.0 for n in self.nethra}
        for a,b,g in edges:
            degree[a]+=g;degree[b]+=g
        max_degree=max(degree.values(),default=0.0)
        rate=(self.leakage+2.0*max_degree)/self.capacitance
        stable_h=2.0/rate if rate>0.0 else dt
        pieces=max(1,int(math.ceil(dt/min(dt,stable_h))))
        h=dt/pieces

        type(self).max_substeps_seen=max(type(self).max_substeps_seen,pieces)
        type(self).max_weighted_degree_seen=max(type(self).max_weighted_degree_seen,max_degree)
        type(self).intervals_executed+=1

        for _ in range(pieces):
            a0=state
            k1=self.cached_derivative(a0,edges)
            a1={n:a0[n]+.5*h*k1[n] for n in self.nethra}
            k2=self.cached_derivative(a1,edges)
            a2={n:a0[n]+.5*h*k2[n] for n in self.nethra}
            k3=self.cached_derivative(a2,edges)
            a3={n:a0[n]+h*k3[n] for n in self.nethra}
            k4=self.cached_derivative(a3,edges)
            state={
                n:a0[n]+h*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.0
                for n in self.nethra
            }

        delta={}
        for n in self.nethra:
            n.activation=state[n]
            if not math.isfinite(n.activation):
                raise FloatingPointError(("nonfinite activation",self.interval_no,pieces,max_degree))
            delta[n]=n.activation-initial[n]

        self._complete_interval(source_current,delta)
        self._consider_completed_interval_provisional(explicit)
        for n in self.nethra:
            n.external=0.0
        return delta


def verify_cached_equivalence():
    rng=random.Random(3200260923)
    f=StableDepthAuditField(
        g_min=.20,g_max=1.50,tau=100.0,
        capacitance=1.0,leakage=.6,convergence_gain=0.0,
    )
    nodes=[f.new() for _ in range(8)]
    for i in range(1,8):
        r=f.new()
        f._route(r,(nodes[i-1],nodes[i]),frozenset(),rng.randint(1,50))
    for n in f.nethra:
        n.activation=rng.uniform(-.2,.8)
        n.external=rng.uniform(-.1,.5)
    state={n:n.activation for n in f.nethra}
    edges=f._edges()
    core=f._derivative_at(state)
    cached=f.cached_derivative(state,edges)
    err=max(abs(core[n]-cached[n]) for n in f.nethra)
    print("CACHED_DERIVATIVE",{"max_abs_error":err,"edges":len(edges)},flush=True)
    assert err<1e-13,err


def report_executor():
    print("STABILITY_EXECUTOR",{
        "intervals_including_shadows":StableDepthAuditField.intervals_executed,
        "max_substeps_seen":StableDepthAuditField.max_substeps_seen,
        "max_weighted_degree_seen":StableDepthAuditField.max_weighted_degree_seen,
        "external_interval_dt":bench.DT,
    },flush=True)


def main():
    verify_cached_equivalence()
    atexit.register(report_executor)
    bench.DepthAuditField=StableDepthAuditField
    bench.main()


if __name__=="__main__":
    main()
