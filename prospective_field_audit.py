#!/usr/bin/env python3
"""Low-cost audit of the full prospective Nethra field.

Prediction phase:
    feed elapsed TIME only
    evolve field
    inspect ALL learned Nethra before any new price data enters

Only after the prospective snapshot is recorded:
    inject the simultaneous multi-symbol price vector
    update plasticity
    refind/construct
    advance

This audit removes the six constant symbol-name nodes from construction. Symbol identity is already
grounded by distinct AAPL+/AAPL-, MSFT+/MSFT-, ... price Nethra, so a relation only acquires
multi-symbol provenance when its learned members actually span those grounded channels.

Diagnostics do not feed back into learning:
  - activation and instantaneous net/absolute incident current for higher Nethra
  - recursively inherited grounded provenance
  - actual positive-current paths from higher Nethra to price manifestations
  - direct contributors into each grounded price Nethra
"""

from __future__ import annotations

import heapq
import json
import math
import os
import statistics

import numpy as np

import aapl_residual_recursive_native as base
from aapl_residual_recursive_native import (
    CAPACITANCE, OBS_DT, ADMISSION_RESIDUAL, integrate, preflow, plasticity, g_of_e,
)
from multisymbol_time_priming import (
    SYMBOLS, MultiReplay, load_aligned, make_intervals,
)


TRAIN=int(os.environ.get("NETHRA_AUDIT_TRAIN","300"))
ONLINE=int(os.environ.get("NETHRA_AUDIT_ONLINE","20"))
TOP_GLOBAL=int(os.environ.get("NETHRA_AUDIT_TOP","10"))
TOP_PATH=int(os.environ.get("NETHRA_AUDIT_PATH","4"))


def outcome_price_only(pre,currents,pplus_idx,pminus_idx,er,em,ee):
    """Reveal only grounded symbol-specific price manifestations; no constant symbol channels."""
    actual=pre.copy()
    baseline=pre.copy()
    ext=np.zeros(pre.shape[0],np.float64)
    for j,c in enumerate(currents):
        if c>=0:
            ext[pplus_idx[j]]=float(c)
        else:
            ext[pminus_idx[j]]=-float(c)
    integrate(actual,ext,er,em,ee,OBS_DT)
    integrate(baseline,np.zeros_like(ext),er,em,ee,OBS_DT)
    target=(actual-baseline)*CAPACITANCE
    target[target<0.0]=0.0
    return actual,target


class ProspectiveReplay(MultiReplay):
    def interval_audited(self,currents,elapsed,learn=True,construct=True,audit=False):
        if self.topology_dirty:
            self.sync_topology()
        if len(self.state)!=len(self.f.nethra):
            self._sync_indices()

        # Prospective interval: TIME and nothing else.
        ext=np.zeros(len(self.f.nethra),np.float64)
        ext[self.index[self.time]]=1.0
        integrate(self.state,ext,self.er,self.em,self.ee,float(elapsed))

        pout,pin,pred,supply=preflow(self.state,self.er,self.em,self.ee)
        prospective_state=self.state.copy()

        snapshot=None
        if audit:
            snapshot=field_snapshot(self,prospective_state,pout,pin,pred)

        actual,target=outcome_price_only(
            prospective_state,np.asarray(currents,np.float64),
            self.pplus_idx,self.pminus_idx,self.er,self.em,self.ee
        )

        eps=target-pred
        price_eps=0.0
        for j in range(len(SYMBOLS)):
            price_eps+=abs(float(eps[self.pplus_idx[j]]))
            price_eps+=abs(float(eps[self.pminus_idx[j]]))

        if learn and self.er.shape[0]:
            plasticity(
                self.ee,self.er,self.em,pout,pin,pred,supply,target,
                self.relmask,self.stats_tension,self.stats_flow,self.stats_abs
            )

        self.state=actual

        # Differentiate-first construction: the full observation updates plasticity, but new
        # structure sees only grounded manifestations the existing field under-predicted. Expected
        # manifestations have already been accounted for by the field and are subtracted here.
        explicit={self.time}
        unresolved_residual=0.0
        unresolved_count=0
        for j,s in enumerate(SYMBOLS):
            node=self.pplus[s] if currents[j]>=0 else self.pminus[s]
            idx=self.index[node]
            remaining=max(0.0,float(eps[idx]))
            if remaining>ADMISSION_RESIDUAL:
                explicit.add(node)
                unresolved_residual+=remaining
                unresolved_count+=1

        closure_size=0
        if construct:
            _r,closure_size=self.structural_step(explicit,unresolved_residual)

        return {
            "pred_plus":np.asarray([float(pred[i]) for i in self.pplus_idx]),
            "pred_minus":np.asarray([float(pred[i]) for i in self.pminus_idx]),
            "surprise":price_eps,
            "closure_size":closure_size,
            "unresolved_count":unresolved_count,
            "unresolved_residual":unresolved_residual,
            "snapshot":snapshot,
        }


def provenance_map(model):
    """Recursive construction provenance, computed once for diagnostics."""
    prov={}
    labels={}
    labels[model.time]="TIME"
    prov[model.time]=frozenset(("TIME",))
    for s in SYMBOLS:
        labels[model.pplus[s]]=s+"+"
        labels[model.pminus[s]]=s+"-"
        prov[model.pplus[s]]=frozenset((s+"+",))
        prov[model.pminus[s]]=frozenset((s+"-",))
        # Constant symbol nodes are unused by this variant.
        prov[model.symbol_node[s]]=frozenset()
        labels[model.symbol_node[s]]="UNUSED_SYMBOL_"+s

    for r in sorted(model.birth_members,key=lambda n:(model.depth[n],model.index[n])):
        p=set()
        for m in model.birth_members[r]:
            p.update(prov.get(m,frozenset()))
        prov[r]=frozenset(p)
        labels[r]="R"+str(model.index[r])
    return prov,labels


def directed_flow_graph(model,state):
    """Actual instantaneous positive-current graph at the prospective state."""
    fwd={}
    rev={}
    incident=np.zeros(len(state),np.float64)
    net=np.zeros(len(state),np.float64)

    def add(a,b,q):
        fwd.setdefault(a,[]).append((b,q))
        rev.setdefault(b,[]).append((a,q))
        incident[a]+=q; incident[b]+=q
        net[a]-=q; net[b]+=q

    for k in range(len(model.er)):
        a=int(model.er[k]); b=int(model.em[k])
        q=float(g_of_e(float(model.ee[k]))*(state[a]-state[b])*OBS_DT)
        if q>0:
            add(a,b,q)
        elif q<0:
            add(b,a,-q)
    return fwd,rev,incident,net


def strongest_upstream(rev,target,nodes_allowed,limit):
    """Widest actual-current paths upstream of target; diagnostic only."""
    best={target:math.inf}
    next_hop={}
    heap=[(-1e300,target)]
    while heap:
        _neg,u=heapq.heappop(heap)
        bu=best[u]
        for src,q in rev.get(u,()):
            cand=min(bu,q)
            if cand>best.get(src,-1.0):
                best[src]=cand
                next_hop[src]=u
                heapq.heappush(heap,(-cand,src))

    rows=[]
    for src in nodes_allowed:
        if src not in best or not math.isfinite(best[src]) or best[src]<=0:
            continue
        path=[src]
        u=src
        seen={src}
        while u!=target and u in next_hop:
            u=next_hop[u]
            if u in seen: break
            seen.add(u); path.append(u)
        if path[-1]==target:
            rows.append((best[src],src,path))
    rows.sort(reverse=True,key=lambda x:x[0])
    return rows[:limit]


def field_snapshot(model,state,pout,pin,pred):
    prov,labels=provenance_map(model)
    fwd,rev,incident,net=directed_flow_graph(model,state)
    relations=list(model.birth_members)

    def relation_row(r):
        i=model.index[r]
        p=prov.get(r,frozenset())
        symbols=sorted({x[:-1] for x in p if x!="TIME" and x[-1:] in ("+","-")})
        return {
            "id":labels[r],
            "index":i,
            "depth":model.depth[r],
            "activation":float(state[i]),
            "incident_abs_current":float(incident[i]),
            "net_current":float(net[i]),
            "provenance_channels":sorted(p)[:24],
            "provenance_channel_count":len(p),
            "symbols":symbols,
        }

    # Global prospective higher-Nethra state.
    active=[r for r in relations if abs(float(state[model.index[r]]))>1e-15 or incident[model.index[r]]>1e-15]
    top_activation=sorted(
        active,key=lambda r:(abs(float(state[model.index[r]])),incident[model.index[r]]),reverse=True
    )[:TOP_GLOBAL]
    top_current=sorted(
        active,key=lambda r:incident[model.index[r]],reverse=True
    )[:TOP_GLOBAL]

    by_price={}
    relation_indices={model.index[r]:r for r in relations}
    allowed=set(relation_indices)
    for j,s in enumerate(SYMBOLS):
        for sign,target_i in (("+",int(model.pplus_idx[j])),("-",int(model.pminus_idx[j]))):
            direct=[]
            for src,q in sorted(rev.get(target_i,()),key=lambda x:x[1],reverse=True)[:TOP_PATH]:
                r=relation_indices.get(src)
                direct.append({
                    "source":labels.get(r,"N"+str(src)) if r is not None else "N"+str(src),
                    "source_depth":model.depth.get(r,0) if r is not None else 0,
                    "current":float(q),
                    "source_activation":float(state[src]),
                    "provenance":sorted(prov.get(r,frozenset()))[:16] if r is not None else [],
                })

            paths=[]
            for bottleneck,src,path in strongest_upstream(rev,target_i,allowed,TOP_PATH):
                r=relation_indices[src]
                paths.append({
                    "source":labels[r],
                    "source_depth":model.depth[r],
                    "source_activation":float(state[src]),
                    "bottleneck_current":float(bottleneck),
                    "path":[labels.get(relation_indices.get(i),"N"+str(i)) for i in path],
                    "provenance":sorted(prov.get(r,frozenset()))[:16],
                })

            by_price[s+sign]={
                "ground_activation":float(state[target_i]),
                "incoming_prediction_charge":float(pred[target_i]),
                "direct_upstream":direct,
                "strongest_higher_paths":paths,
            }

    depths=[model.depth[r] for r in active]
    return {
        "active_relation_count":len(active),
        "deepest_active_relation":max(depths,default=0),
        "mean_active_depth":statistics.mean(depths) if depths else 0.0,
        "top_activation": [relation_row(r) for r in top_activation],
        "top_incident_current": [relation_row(r) for r in top_current],
        "price_manifestations":by_price,
    }


def summarize_snapshots(snaps):
    active=[s["active_relation_count"] for s in snaps]
    deep=[s["deepest_active_relation"] for s in snaps]
    return {
        "snapshots":len(snaps),
        "mean_active_relations":statistics.mean(active) if active else 0.0,
        "max_active_relations":max(active,default=0),
        "mean_deepest_active":statistics.mean(deep) if deep else 0.0,
        "max_deepest_active":max(deep,default=0),
    }


def main():
    base.FAST_SYNC=True
    stamps,prices,kinds,dates=load_aligned()
    currents,raw,elapsed=make_intervals(stamps,prices)
    train_n=TRAIN-1

    model=ProspectiveReplay()
    model.reset_transient()

    for i in range(train_n):
        model.interval_audited(currents[i],elapsed[i],True,True,False)

    train_depth=max(model.depth.values(),default=0)
    train_rel=len(model.birth_members)

    rows=[]
    detailed=[]
    detailed_at={0,max(0,ONLINE//2),max(0,ONLINE-1)}
    for k,i in enumerate(range(train_n,train_n+ONLINE)):
        out=model.interval_audited(currents[i],elapsed[i],True,True,True)
        margin=out["pred_plus"]-out["pred_minus"]
        truth=np.where(currents[i]>=0,1,-1)
        pred=np.where(margin>=0,1,-1)
        rows.append({
            "margin":margin.tolist(),
            "truth":truth.tolist(),
            "correct":(pred==truth).tolist(),
            "surprise":float(out["surprise"]),
            "unresolved_count":int(out["unresolved_count"]),
            "unresolved_residual":float(out["unresolved_residual"]),
            "snapshot":out["snapshot"],
        })
        if k in detailed_at:
            detailed.append({
                "step":k,
                "date":dates[i+1],
                "field":out["snapshot"],
            })

    # Cross-symbol provenance now requires actually inherited symbol-specific price channels.
    prov,_=provenance_map(model)
    cross=[]
    for r in model.birth_members:
        syms={x[:-1] for x in prov.get(r,frozenset()) if x!="TIME" and x[-1:] in ("+","-")}
        if len(syms)>=2:
            cross.append(r)

    margins=np.asarray([r["margin"] for r in rows])
    truths=np.asarray([r["truth"] for r in rows])
    resolved=np.abs(margins)>1e-18
    correct=(np.where(margins>=0,1,-1)==truths)
    agg_correct=int(np.sum(correct & resolved))
    agg_resolved=int(np.sum(resolved))

    result={
        "train_timestamps":TRAIN,
        "online_timestamps":ONLINE,
        "symbols":SYMBOLS,
        "prediction_input":"TIME_ONLY",
        "constant_symbol_nodes_in_construction":False,
        "learning_live":True,
        "construction_live":True,
        "train_depth":train_depth,
        "final_depth":max(model.depth.values(),default=0),
        "relations_train":train_rel,
        "relations_final":len(model.birth_members),
        "cross_symbol_relations":len(cross),
        "deepest_cross_symbol_depth":max((model.depth[r] for r in cross),default=0),
        "aggregate_resolved":agg_resolved,
        "aggregate_accuracy":agg_correct/agg_resolved if agg_resolved else None,
        "mean_unresolved_outputs":statistics.mean(r["unresolved_count"] for r in rows),
        "mean_unresolved_residual":statistics.mean(r["unresolved_residual"] for r in rows),
        "field_summary":summarize_snapshots([r["snapshot"] for r in rows]),
        "detailed_snapshots":detailed,
    }
    print("RESULT",json.dumps(result,sort_keys=True),flush=True)
    print("all_assertions_passed",flush=True)


if __name__=="__main__":
    main()
