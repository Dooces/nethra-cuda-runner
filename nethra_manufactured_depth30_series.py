#!/usr/bin/env python3
"""Long deterministic manufactured-series audit for the CURRENT Nethra core.

Goal
----
Give current Nethra a long, repeatedly cycled stream with explicitly discoverable dependencies at
4, 8, 16 and 30 observed intervals, including cases where the final 29 observations before two
different outcomes are byte-for-byte identical and only the cue 30 observations back distinguishes
the continuation.

Nothing in the learner is told those dependency lengths, cue identities, motif boundaries, or
outcomes. They exist only in the generator/auditor.

The generated supercycle is fixed once and repeated continuously. There is no per-cycle shuffling
and no random next event. A seeded permutation is used only once to choose the fixed motif order.

Learning uses NethraField.step() directly. The only subclass behavior records relation birth depth
after calling the unchanged core _mint_history().

After learning settles or reaches MAX_CYCLES, the existing passive one-step prediction audit runs on
one more supercycle using disposable TIME-only shadows. Prediction never feeds back.
"""
from __future__ import annotations

import copy
import json
import math
import random
import time
from collections import Counter, defaultdict

from nethra import NethraField

DT=float(__import__("os").environ.get("NETHRA_SYNTH_DT","0.15"))
MAX_CYCLES=int(__import__("os").environ.get("NETHRA_SYNTH_MAX_CYCLES","256"))
MIN_CYCLES=int(__import__("os").environ.get("NETHRA_SYNTH_MIN_CYCLES","64"))
SETTLE_CYCLES=int(__import__("os").environ.get("NETHRA_SYNTH_SETTLE_CYCLES","32"))
RELATION_SAFETY=int(__import__("os").environ.get("NETHRA_SYNTH_RELATION_SAFETY","8000"))

DEPTHS=(4,8,16,30)
VARIANTS=3
BODY_SYMBOLS=tuple(f"B{i}" for i in range(8))
CUES=("C0","C1")
OUTCOMES=("O0","O1")
SEPARATORS=("Z0","Z1","Z2")
ALL_SYMBOLS=BODY_SYMBOLS+CUES+OUTCOMES+SEPARATORS


class DepthAuditField(NethraField):
    """Unchanged core semantics plus passive provenance bookkeeping for newly minted relations."""
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self.birth_depth={}
        self.birth_interval={}
        self.birth_members={}
        self.interval_no=0
        self.created_relations=0

    def _mint_history(self,before,after,evidence,history_key=None):
        pre=set(self.nethra)
        r=super()._mint_history(before,after,evidence,history_key)
        if r is not None and r not in pre:
            members=set()
            for route in r.routes:
                members.update(route)
            self.birth_members[r]=frozenset(members)
            self.birth_depth[r]=1+max((self.birth_depth.get(m,0) for m in members),default=0)
            self.birth_interval[r]=self.interval_no
            self.created_relations+=1
        return r

    def step(self,dt=.1):
        self.interval_no+=1
        return super().step(dt)


def body_for(depth,variant):
    """Deterministic shared body of length depth-1.

    Cue 0 and cue 1 receive exactly the same body for a given (depth,variant), ensuring that no
    suffix shorter than depth can distinguish their two different outcomes.
    """
    n=depth-1
    x=(depth*5+variant*11+3)%17
    out=[]
    for i in range(n):
        # Small alphabet, recurrent internal motifs, and a nonlinear deterministic recurrence.
        x=(x*x + 3*x + 5 + i*7 + variant*13 + depth)%97
        idx=(x + (i//3) + (i%5)*variant + depth//4)%len(BODY_SYMBOLS)
        out.append(BODY_SYMBOLS[idx])
    return tuple(out)


def make_supercycle():
    motifs=[]
    for depth in DEPTHS:
        for variant in range(VARIANTS):
            # Each body occurs under both cues and therefore both outcomes.
            for cue in (0,1):
                motifs.append((depth,variant,cue))

    # Fixed once. Every training cycle is the exact same deterministic order.
    rng=random.Random(20260923)
    rng.shuffle(motifs)

    stream=[]
    target_meta={}
    motif_rows=[]
    for motif_id,(depth,variant,cue) in enumerate(motifs):
        sep=SEPARATORS[variant]
        body=body_for(depth,variant)
        start=len(stream)
        stream.append(sep)
        stream.append(CUES[cue])
        stream.extend(body)
        outcome_pos=len(stream)
        stream.append(OUTCOMES[cue])
        target_meta[outcome_pos]={
            "designed_depth":depth,
            "variant":variant,
            "cue":cue,
            "motif_id":motif_id,
        }
        motif_rows.append({
            "motif_id":motif_id,
            "designed_depth":depth,
            "variant":variant,
            "cue":cue,
            "start":start,
            "outcome_pos":outcome_pos,
            "length":len(stream)-start,
        })
    return tuple(stream),target_meta,motif_rows


def cyclic_context_requirement(stream,pos,max_k=40):
    """Minimum suffix length that uniquely determines stream[pos] on this fixed cyclic supercycle."""
    n=len(stream)
    target=stream[pos]
    for k in range(1,max_k+1):
        context=tuple(stream[(pos-k+j)%n] for j in range(k))
        outcomes=set()
        for q in range(n):
            c=tuple(stream[(q-k+j)%n] for j in range(k))
            if c==context:
                outcomes.add(stream[q])
        if outcomes=={target}:
            return k
    return None


def verify_generator(stream,target_meta):
    rows=[]
    for pos,meta in sorted(target_meta.items()):
        actual=cyclic_context_requirement(stream,pos,40)
        rows.append({**meta,"pos":pos,"actual_min_context":actual})
    by_design=defaultdict(list)
    for r in rows:by_design[r["designed_depth"]].append(r["actual_min_context"])
    return rows,{d:sorted(v) for d,v in by_design.items()}


def route_evidence_total(r):
    return sum(float(v) for c in r.routes.values() for v in c.values())


def relation_snapshot(f):
    rel=[n for n in f.nethra if n.routes]
    depth=max((f.birth_depth.get(r,0) for r in rel),default=0)
    hist=Counter(f.birth_depth.get(r,0) for r in rel)
    deepest=sorted(rel,key=lambda r:(f.birth_depth.get(r,0),route_evidence_total(r)),reverse=True)[:8]
    keys_by_relation=defaultdict(list)
    for key,r in f.history_relation.items():
        keys_by_relation[r].append(key)
    deep_rows=[]
    for r in deepest:
        keys=keys_by_relation.get(r,[])
        recurrence=sum(f.history_count[k] for k in keys)
        deep_rows.append({
            "depth":f.birth_depth.get(r,0),
            "route_count":len(r.routes),
            "route_evidence":route_evidence_total(r),
            "mapped_histories":len(keys),
            "history_recurrence":recurrence,
            "birth_interval":f.birth_interval.get(r),
        })
    return {
        "nethra":len(f.nethra),
        "relations":len(rel),
        "max_depth":depth,
        "depth_histogram":dict(sorted(hist.items())),
        "deepest":deep_rows,
    }


def feed(f,time_node,symbol_node):
    time_node.push(1.0)
    symbol_node.push(1.0)
    f.step(DT)


def train():
    stream,target_meta,motifs=make_supercycle()
    req_rows,req_summary=verify_generator(stream,target_meta)

    print("GENERATOR",json.dumps({
        "supercycle_length":len(stream),
        "motifs":len(motifs),
        "alphabet":len(ALL_SYMBOLS),
        "designed_depths":DEPTHS,
        "variants_per_depth":VARIANTS,
        "outcome_targets_per_cycle":len(target_meta),
        "actual_min_context_by_design":req_summary,
        "randomness_during_cycles":False,
    },sort_keys=True),flush=True)

    # Assert the manufactured problem really contains the requested dependency lengths.
    for d in DEPTHS:
        vals=req_summary[d]
        if not vals or any(v!=d for v in vals):
            raise AssertionError(("generator context depth mismatch",d,vals))

    f=DepthAuditField(
        g_min=.20,g_max=1.50,tau=100.0,
        capacitance=1.0,leakage=.6,convergence_gain=0.0,
    )
    time_node=f.new()
    symbols={name:f.new() for name in ALL_SYMBOLS}

    report_cycles={1,2,4,8,16,32,64,96,128,160,192,224,256}
    stable=0
    last_rel=-1
    last_depth=-1
    final_cycle=0
    t0=time.perf_counter()

    for cycle in range(1,MAX_CYCLES+1):
        for name in stream:
            feed(f,time_node,symbols[name])

        snap=relation_snapshot(f)
        unchanged=(snap["relations"]==last_rel and snap["max_depth"]==last_depth)
        stable=stable+1 if unchanged else 0
        last_rel=snap["relations"];last_depth=snap["max_depth"]
        final_cycle=cycle

        if cycle in report_cycles or cycle==MAX_CYCLES or stable==SETTLE_CYCLES:
            print("TRAIN",json.dumps({
                "cycle":cycle,
                "intervals":cycle*len(stream),
                "relations":snap["relations"],
                "nethra":snap["nethra"],
                "max_depth":snap["max_depth"],
                "stable_cycles":stable,
                "seconds":time.perf_counter()-t0,
            },sort_keys=True),flush=True)

        if snap["relations"]>RELATION_SAFETY:
            raise RuntimeError(f"relation safety exceeded: {snap['relations']}")
        if cycle>=MIN_CYCLES and stable>=SETTLE_CYCLES:
            break

    trained=relation_snapshot(f)
    print("TRAINED",json.dumps({
        "cycles":final_cycle,
        "intervals":final_cycle*len(stream),
        **trained,
        "seconds":time.perf_counter()-t0,
    },sort_keys=True),flush=True)

    return f,time_node,symbols,stream,target_meta,req_rows,trained


def passive_predict(f,time_node,symbols):
    shadow=copy.deepcopy(f)
    # Object identities differ in deepcopy. Stable index positions map live nodes to shadow nodes.
    live_index={n:i for i,n in enumerate(f.nethra)}
    stime=shadow.nethra[live_index[time_node]]
    ssymbols={name:shadow.nethra[live_index[node]] for name,node in symbols.items()}

    before={name:n.activation for name,n in ssymbols.items()}
    stime.push(1.0)
    shadow.step(DT)
    decay=math.exp(-shadow.leakage*DT/shadow.capacitance)
    scores={
        name:ssymbols[name].activation-before[name]*decay
        for name in ssymbols
    }
    order=sorted(scores,key=scores.get,reverse=True)
    return scores,order


def audit(f,time_node,symbols,stream,target_meta):
    rows=[]
    all_correct=0
    all_n=0

    for pos,actual in enumerate(stream):
        scores,order=passive_predict(f,time_node,symbols)
        rank=order.index(actual)+1
        all_n+=1
        all_correct+=rank==1

        if pos in target_meta:
            meta=target_meta[pos]
            top1=order[0];top2=order[1]
            rows.append({
                **meta,
                "actual":actual,
                "rank":rank,
                "top1":top1,
                "top1_score":scores[top1],
                "top2":top2,
                "top2_score":scores[top2],
                "margin":scores[top1]-scores[top2],
                "actual_score":scores[actual],
                "top5":[(x,scores[x]) for x in order[:5]],
                "correct":rank==1,
            })

        # Reality is revealed only after the disposable prediction.
        feed(f,time_node,symbols[actual])

    by_depth={}
    for d in DEPTHS:
        rr=[r for r in rows if r["designed_depth"]==d]
        by_depth[d]={
            "n":len(rr),
            "top1":sum(r["rank"]==1 for r in rr)/len(rr),
            "top3":sum(r["rank"]<=3 for r in rr)/len(rr),
            "mean_rank":sum(r["rank"] for r in rr)/len(rr),
            "mean_margin":sum(r["margin"] for r in rr)/len(rr),
        }

    print("PREDICTION_SUMMARY",json.dumps({
        "all_symbols_n":all_n,
        "all_symbols_top1":all_correct/all_n,
        "designed_outcomes":len(rows),
        "by_designed_context_depth":by_depth,
    },sort_keys=True),flush=True)

    for r in rows:
        if r["designed_depth"] in (16,30):
            print("DEEP_PRED",json.dumps(r,sort_keys=True),flush=True)

    return rows,by_depth


def main():
    f,time_node,symbols,stream,target_meta,req_rows,trained=train()
    rows,by_depth=audit(f,time_node,symbols,stream,target_meta)

    print("FINAL",json.dumps({
        "trained_depth":trained["max_depth"],
        "trained_relations":trained["relations"],
        "target_recursive_depth":30,
        "generator_verified_context_depths":sorted(set(r["actual_min_context"] for r in req_rows)),
        "prediction_by_depth":by_depth,
        "post_audit":relation_snapshot(f),
    },sort_keys=True),flush=True)
    print("all_assertions_passed",flush=True)


if __name__=="__main__":
    main()
