#!/usr/bin/env python3
"""Isolate why current AAPL recursive depth is far below the historical ~74-depth run.

Same current code, same fetched AAPL data, same adaptive/stable numerical integrator, one 20k-price
pass. Only structural refinding semantics differ:

CURRENT:
  NativeReplay.structural_step() recomputes the recursive event as current Nethra are refound.

LEGACY_STALE_EVENT:
  exact historical structural_step used by the run that reached depth 74:
    closed = f.closure(explicit, f.current_event)
  where f.current_event is the previous interval's description event; description_event is then
  computed after closure from closed vs previous_closure.

Thus this counterfactual isolates stale previous-event refinding from numerical integration,
plasticity, _mint_history(), _accounted(), depth provenance, source inputs, and data.
"""
from __future__ import annotations
import json,time
import aapl_residual_recursive_native as base
from aapl_residual_recursive_native import NativeReplay,fetch_aapl,intervals

TRAIN=20000

class LegacyStaleEventReplay(NativeReplay):
    def structural_step(self,explicit,residual):
        explicit=frozenset(explicit)
        cache_key=(explicit,self.f.current_event)
        closed=self.closure_cache.get(cache_key)
        if closed is None:
            closed=self.f.closure(explicit,self.f.current_event)
            self.closure_cache[cache_key]=closed

        source_observed=explicit|self.f.previous_explicit
        source_event=frozenset(
            (n,int(n in explicit)-int(n in self.f.previous_explicit))
            for n in source_observed
        )
        description_observed=closed|self.f.previous_closure
        description_event=frozenset(
            (n,int(n in closed)-int(n in self.f.previous_closure))
            for n in description_observed
        )
        before=self.f.previous_event
        relation=None
        if before and description_event and residual>base.ADMISSION_RESIDUAL:
            key=(before,description_event)
            relation=self.f.history_relation.get(key)
            if relation is not None:
                self.reused+=1
            else:
                n_before=len(self.f.nethra)
                relation=self.f._mint_history(before,description_event,1,history_key=key)
                if len(self.f.nethra)>n_before:
                    self.sync_topology(relation)
                elif relation is not None:
                    self.accounted+=1
                    if base.FAST_SYNC:self.pending_route_relations.add(relation)

        self.f.previous_explicit=explicit
        self.f.previous_closure=closed
        self.f.previous_source_event=source_event
        self.f.current_source_event=source_event
        self.f.previous_event=description_event
        self.f.current_event=description_event
        return relation,len(closed)

def run(cls,currents,elapsed):
    m=cls();m.reset_transient()
    max_closure=0
    t=time.perf_counter()
    for i in range(TRAIN-1):
        out=m.interval(float(currents[i]),float(elapsed[i]),True,True)
        max_closure=max(max_closure,out["closure_size"])
    mature=m.maturity()
    return {
        "relations":len(m.birth_members),"nethra":len(m.f.nethra),
        "incidences":len(m.incidence_e),
        "constructed_depth":max(m.depth.values(),default=0),
        "mature_depth":max((m.depth[r] for r in mature),default=0),
        "mature_relations":len(mature),"created":m.created,
        "reused":m.reused,"accounted":m.accounted,"max_closure":max_closure,
        "seconds":time.perf_counter()-t,
    }

def main():
    base.FAST_SYNC=True
    pts=fetch_aapl();cur,el,k,raw=intervals(pts)
    print("DATA",json.dumps({"points":len(pts),"train":TRAIN,
          "start":pts[0].date,"end":pts[TRAIN-1].date},sort_keys=True),flush=True)
    current=run(NativeReplay,cur,el)
    print("CURRENT",json.dumps(current,sort_keys=True),flush=True)
    legacy=run(LegacyStaleEventReplay,cur,el)
    print("LEGACY_STALE_EVENT",json.dumps(legacy,sort_keys=True),flush=True)
    print("RATIO",json.dumps({
        "relation_ratio":legacy["relations"]/max(1,current["relations"]),
        "depth_ratio":legacy["constructed_depth"]/max(1,current["constructed_depth"]),
    },sort_keys=True),flush=True)
    print("all_assertions_passed",flush=True)
if __name__=="__main__":main()
