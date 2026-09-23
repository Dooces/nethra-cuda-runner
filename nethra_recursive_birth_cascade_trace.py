#!/usr/bin/env python3
"""Trace the first few cycles to show exactly when recursive descriptions change under new topology."""
from __future__ import annotations
from nethra import NethraField

DT=.15
SYMBOLS=4
CYCLES=6

class BirthTraceField(NethraField):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self.interval_no=0
        self.births=[]

    def _mint_history(self,before,after,evidence,history_key=None):
        key=(before,after) if history_key is None else history_key
        pre=set(self.nethra)
        r=super()._mint_history(before,after,evidence,history_key)
        if r is not None and r not in pre:
            self.births.append((self.interval_no,r,key,before,after,evidence))
        return r

    def step(self,dt=.1):
        self.interval_no+=1
        out=super().step(dt)
        nm={n:(chr(65+i) if i<SYMBOLS else f"R{i-SYMBOLS+1}") for i,n in enumerate(self.nethra)}
        def ev(e): return tuple(sorted((nm[n],int(ch)) for n,ch in e))
        print("INTERVAL",self.interval_no,
              "source",ev(self.current_source_event),
              "description",ev(self.current_event),
              "closure",tuple(sorted(nm[n] for n in self.previous_closure)),
              "relations",sum(bool(n.routes) for n in self.nethra))
        while self.births:
            it,r,key,before,after,evidence=self.births.pop(0)
            print("BIRTH",it,nm[r],
                  "source_key",ev(key[0]),"->",ev(key[1]),
                  "description",ev(before),"->",ev(after),
                  "evidence",evidence,
                  "routes",[
                    (tuple(sorted(nm[n] for n in route)),
                     [(ev(sig),int(v)) for sig,v in conds.items()])
                    for route,conds in r.routes.items()
                  ])
        return out

def main():
    f=BirthTraceField(g_min=.20,g_max=1.50,tau=100.0,capacitance=1.0,leakage=.6,convergence_gain=0.0)
    leaves=[f.new() for _ in range(SYMBOLS)]
    for _ in range(CYCLES):
        for i in range(SYMBOLS):
            leaves[i].push(1.0)
            f.step(DT)

if __name__=="__main__":main()
