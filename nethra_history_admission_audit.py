#!/usr/bin/env python3
"""Print all source-history evidence and the exact provisional admission quantities after training A-B-C-D."""
from nethra import NethraField

SYMBOLS=4
CYCLES=300
DT=.15

f=NethraField(g_min=.20,g_max=1.50,tau=100.0,capacitance=1.0,leakage=.6,convergence_gain=0.0)
leaves=[f.new() for _ in range(SYMBOLS)]
for _ in range(CYCLES):
    for i in range(SYMBOLS):
        leaves[i].push(1.0); f.step(DT)

names={n:(chr(65+i) if i<SYMBOLS else f"R{i-SYMBOLS+1}") for i,n in enumerate(f.nethra)}
def ev(e):
    return tuple(sorted((names.get(n,"?"),int(ch)) for n,ch in e))

print("total_histories",f.total_histories)
print("history_count_entries",len(f.history_count))
for (before,after),count in sorted(f.history_count.items(),key=lambda kv:str((ev(kv[0][0]),ev(kv[0][1])))):
    support=f.support_count[before]
    conditional=count/support if support else 0
    baseline=f.outcome_count[after]/f.total_histories if f.total_histories else 0
    subset_rows=[]
    for smaller,seen in f.support_count.items():
        if smaller < before and seen:
            val=f.history_count[(smaller,after)]/seen
            baseline=max(baseline,val)
            subset_rows.append((ev(smaller),seen,val))
    mapped=f.history_relation.get((before,after))
    print("PAIR",ev(before),"->",ev(after),
          "count",count,"support",support,
          "conditional",conditional,"baseline",baseline,
          "would_admit",count>=2 and conditional>baseline,
          "mapped",names.get(mapped))
    if subset_rows: print("  subsets",subset_rows)

print("history_relation")
for (before,after),r in f.history_relation.items():
    print(ev(before),"->",ev(after),names[r])
