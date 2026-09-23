#!/usr/bin/env python3
"""Diagnostic: subtract only already-materialized Nethra histories from new construction.

The stock provisional constructor compares a candidate against every smaller transient history,
including one-shot histories that never earned a Nethra. This variant changes only that diagnostic
comparison: a smaller history may suppress construction only if that exact smaller->outcome history
already exists in history_relation.

No field dynamics, route matching, evidence updates, or admission count threshold are changed.
"""
from __future__ import annotations
import copy
from types import MethodType
from nethra import NethraField

SYMBOLS=4
CYCLES=300
TRAIN_DT=.15
DT=.01
STEPS=300
PREFIX=8

class ExistingOnlyField(NethraField):
    def _consider_completed_interval_provisional(self, explicit):
        explicit=frozenset(explicit)
        closed=self.closure(explicit,self.current_event)

        source_observed=explicit|self.previous_explicit
        source_event=frozenset(
            (n,int(n in explicit)-int(n in self.previous_explicit))
            for n in source_observed
        )
        description_observed=closed|self.previous_closure
        description_event=frozenset(
            (n,int(n in closed)-int(n in self.previous_closure))
            for n in description_observed
        )

        before_source=self.previous_source_event
        before_description=self.previous_event
        if before_source and source_event:
            prior=self.support_count[before_source]
            if prior:
                present_now={n for n,change in source_event if change>=0}
                predicted=self.next_presence_sum[before_source]
                residual={
                    n:(1.0 if n in present_now else 0.0)-predicted[n]/prior
                    for n in self.nethra
                }
                self.update_residuals(residual)

            key=(before_source,source_event)
            self.history_count[key]+=1
            self.support_count[before_source]+=1
            self.outcome_count[source_event]+=1
            self.total_histories+=1
            for n,change in source_event:
                if change>=0:self.next_presence_sum[before_source][n]+=1

            count=self.history_count[key]
            conditional=count/self.support_count[before_source]
            baseline=self.outcome_count[source_event]/self.total_histories
            for smaller,seen in self.support_count.items():
                smaller_key=(smaller,source_event)
                if smaller<before_source and seen and smaller_key in self.history_relation:
                    baseline=max(baseline,self.history_count[smaller_key]/seen)
            if count>=2 and conditional>baseline:
                increment=count if key not in self.history_relation else 1
                self._mint_history(before_description,description_event,increment,history_key=key)

        self.previous_explicit=explicit
        self.previous_closure=closed
        self.previous_source_event=source_event
        self.current_source_event=source_event
        self.previous_event=description_event
        self.current_event=description_event
        return source_event

def make(cls):
    f=cls(g_min=.20,g_max=1.50,tau=100.0,capacitance=1.0,leakage=.6,convergence_gain=0.0)
    leaves=[f.new() for _ in range(SYMBOLS)]
    for _ in range(CYCLES):
        for i in range(SYMBOLS):
            leaves[i].push(1.);f.step(TRAIN_DT)
    return f

def setup(f,cue):
    g=copy.deepcopy(f);leaves=g.nethra[:SYMBOLS]
    for n in g.nethra:n.activation=0.;n.external=0.
    g.previous_explicit=frozenset();g.previous_closure=frozenset()
    g.previous_source_event=frozenset();g.current_source_event=frozenset()
    g.previous_event=frozenset();g.current_event=frozenset()
    g.previous_interval_source={};g.current_interval_source={}
    g.previous_interval_delta={};g.current_interval_delta={}
    start=(cue-PREFIX+1)%SYMBOLS
    for k in range(PREFIX):
        leaves[(start+k)%SYMBOLS].push(1.);g.step(TRAIN_DT)
    for n in g.nethra:n.external=0.
    return g,leaves

def raw_step(f):
    a0={n:n.activation for n in f.nethra};k1=f._derivative_at(a0)
    a1={n:a0[n]+.5*DT*k1[n] for n in f.nethra};k2=f._derivative_at(a1)
    a2={n:a0[n]+.5*DT*k2[n] for n in f.nethra};k3=f._derivative_at(a2)
    a3={n:a0[n]+DT*k3[n] for n in f.nethra};k4=f._derivative_at(a3)
    for n in f.nethra:n.activation=a0[n]+DT*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.

def ungated_route_evidence(self,route,conditions,event):
    return max(conditions.values(),default=0)

def evaluate(f,ungated=False):
    rows=[]
    for cue in range(SYMBOLS):
        g,leaves=setup(f,cue);expected=(cue+1)%SYMBOLS
        if ungated:
            g._route_evidence=MethodType(ungated_route_evidence,g)
        cand=[i for i in range(SYMBOLS) if i!=cue]
        d0=g.derivative();charge={i:0. for i in range(SYMBOLS)}
        for _ in range(STEPS):
            for a,b,cond in g._edges():
                q=cond*(a.activation-b.activation)
                if a.routes and b in leaves and q>0:charge[leaves.index(b)]+=q*DT
                elif b.routes and a in leaves and q<0:charge[leaves.index(a)]+=-q*DT
            raw_step(g)
        do=sorted(cand,key=lambda i:d0[leaves[i]],reverse=True)
        qo=sorted(cand,key=lambda i:charge[i],reverse=True)
        rows.append((cue,expected,do.index(expected)+1,qo.index(expected)+1,do,qo,charge))
    return rows

def describe(f):
    names={n:(chr(65+i) if i<SYMBOLS else f"R{i-SYMBOLS+1}") for i,n in enumerate(f.nethra)}
    def ev(e):return tuple(sorted((names.get(n,"?"),int(ch)) for n,ch in e))
    print("nethra",len(f.nethra),"relations",sum(bool(n.routes) for n in f.nethra),
          "history_keys",len(f.history_relation))
    for (a,b),r in f.history_relation.items():print("H",ev(a),"->",ev(b),names[r])

def main():
    for label,cls in (("CURRENT",NethraField),("EXISTING_ONLY",ExistingOnlyField)):
        print("\n",label)
        f=make(cls);describe(f)
        for gate_label,ungated in (("GATED",False),("UNGATED",True)):
            print(gate_label)
            rows=evaluate(f,ungated)
            for r in rows:print("eval",r)
            print("dadt_rank1",sum(r[2]==1 for r in rows),"charge_rank1",sum(r[3]==1 for r in rows))
if __name__=="__main__":main()
