#!/usr/bin/env python3
"""Trace why mature evidence reverses prospective signed support."""
from nethra import NethraField

DT=.15
SYMBOLS=4

def train(cycles,cue=1):
    f=NethraField(g_min=.20,g_max=1.50,tau=100.0,capacitance=1.0,leakage=.6,convergence_gain=0.0)
    leaves=[f.new() for _ in range(SYMBOLS)]
    start=(cue+1)%SYMBOLS
    for k in range(cycles*SYMBOLS):
        i=(start+k)%SYMBOLS
        leaves[i].push(1.0);f.step(DT)
    return f,leaves

def main():
    for cycles in (3,10,30,100,300):
        f,leaves=train(cycles,1)
        names={n:(chr(65+i) if i<4 else f"R{i-3}") for i,n in enumerate(f.nethra)}
        print("\nCYCLES",cycles,"event",tuple(sorted((names[n],ch) for n,ch in f.current_event)))
        print("ACT",[(names[n],round(n.activation,9)) for n in f.nethra])
        print("EDGES")
        for a,b,g in sorted(f._edges(),key=lambda x:(names[x[0]],names[x[1]])):
            print(names[a],names[b],round(g,9),round(g*(a.activation-b.activation),9))
        print("ROUTES")
        for r in f.nethra:
            if not r.routes:continue
            for route,conds in r.routes.items():
                print(names[r],tuple(sorted(names[n] for n in route)),
                      [(tuple(sorted((names[n],ch) for n,ch in sig)),ev) for sig,ev in conds.items()])
if __name__=="__main__":main()
