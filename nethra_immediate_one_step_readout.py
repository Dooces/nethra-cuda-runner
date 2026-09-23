#!/usr/bin/env python3
from nethra import NethraField
DT=.15
f=NethraField(g_min=.20,g_max=1.50,tau=100.,capacitance=1.,leakage=.6,convergence_gain=0.)
leaves=[f.new() for _ in range(4)];time=f.new()
for _ in range(300):
    for i in range(4):
        time.push(1.0);leaves[i].push(1.0);f.step(DT)
# advance naturally A then B so cue is B
for i in (0,1):
    time.push(1.0);leaves[i].push(1.0);f.step(DT)
for n in f.nethra:n.external=0.0
# Read derivative with no source, then with TIME only, without advancing.
d0=f.derivative()
time.push(1.0)
dt=f.derivative()
print("NO_SOURCE",[(i,d0[n]) for i,n in enumerate(leaves)])
print("TIME_ONLY_NOW",[(i,dt[n]) for i,n in enumerate(leaves)])
print("rank_no_source",sorted(range(4),key=lambda i:d0[leaves[i]],reverse=True))
print("rank_time_only",sorted(range(4),key=lambda i:dt[leaves[i]],reverse=True))
