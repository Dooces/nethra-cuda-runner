#!/usr/bin/env python3
"""Wide-library deterministic Nethra plateau -> one-step prediction audit.

Purpose:
  Test the CURRENT Nethra core on a substantially wider external alphabet without auditing
  predictions during learning.

Stream:
  - 64 externally grounded symbol Nethra: S00..S63.
  - A fixed deterministic supercycle assembled from 32 reusable 16-symbol motifs.
  - Motifs deliberately reuse symbols and partial submotifs under different surrounding contexts,
    so the stream is not a single trivial 64-symbol rotation.
  - The supercycle is generated once and then repeated exactly. There is no per-cycle shuffling and
    no random next event.

Learning:
  - ordinary Nethra learning/construction only;
  - no prediction clone, no top-k calculation, no correctness scoring;
  - structural checkpoints report only Nethra/relation count and constructed depth;
  - plateau = unchanged relation count and max depth for PLATEAU_CYCLES complete supercycles after
    MIN_CYCLES.

After plateau:
  - audit AUDIT_STEPS consecutive real next observations;
  - before each real observation, clone the complete settled live field;
  - supply TIME only for one ordinary interval on the disposable clone;
  - rank all 64 input Nethra by finite-horizon support above passive leakage;
  - log top-k, rank, and margin;
  - discard clone;
  - reveal the actual symbol to the untouched live field and continue normally.

The stable RK4 executor changes only internal numerical subdivision when required. Field equations,
external interval duration, learning/construction, and interval-boundary semantics are unchanged.
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from collections import Counter

from nethra_manufactured_depth30_stable import StableDepthAuditField

DT=float(os.environ.get("NETHRA_WIDE_DT","0.15"))
MIN_CYCLES=int(os.environ.get("NETHRA_WIDE_MIN_CYCLES","24"))
MAX_CYCLES=int(os.environ.get("NETHRA_WIDE_MAX_CYCLES","96"))
PLATEAU_CYCLES=int(os.environ.get("NETHRA_WIDE_PLATEAU_CYCLES","20"))
AUDIT_STEPS=int(os.environ.get("NETHRA_WIDE_AUDIT_STEPS","48"))
TOPK=int(os.environ.get("NETHRA_WIDE_TOPK","5"))
RELATION_SAFETY=int(os.environ.get("NETHRA_WIDE_RELATION_SAFETY","12000"))

N_SYMBOLS=64
N_MOTIFS=32
MOTIF_LEN=16
SYMBOL_NAMES=tuple(f"S{i:02d}" for i in range(N_SYMBOLS))


def make_motifs():
    """Build reusable motifs with shared internal fragments and different contexts."""
    # Eight reusable 4-symbol atoms. Each atom deliberately overlaps the 64-symbol library with
    # other atoms; later motifs recombine these atoms in different orders and phase shifts.
    atoms=[]
    for a in range(16):
        # Partition the 64-symbol library exactly once across the atom library. Motifs then
        # reuse/reorder these atoms under different surrounding contexts.
        atoms.append(tuple(
            f"S{(4*a + off):02d}"
            for off in (0, 1, 2, 3)
        ))

    motifs=[]
    for m in range(N_MOTIFS):
        # Four atoms per motif; adjacent motifs share some atoms but in different positions.
        ids=((m*5+0)%16,(m*5+3)%16,(m*7+1)%16,(m*11+6)%16)
        parts=[]
        for j,aid in enumerate(ids):
            atom=atoms[aid]
            # Fixed phase reversal on some placements gives the same members different temporal use.
            if (m+j)%3==0:
                atom=atom[1:]+atom[:1]
            elif (m+j)%5==0:
                atom=tuple(reversed(atom))
            parts.extend(atom)
        motifs.append(tuple(parts))
    return tuple(motifs)


def make_supercycle():
    motifs=make_motifs()

    # Fixed macro order chosen once. The same order repeats forever.
    order=list(range(N_MOTIFS))
    rng=random.Random(20260923)
    rng.shuffle(order)

    # A second deterministic pass reuses selected motifs later, separated by different neighbors.
    order2=[(x*9+5)%N_MOTIFS for x in order]
    macro=tuple(order + order2)

    stream=[]
    motif_boundaries=[]
    for pos,m in enumerate(macro):
        motif_boundaries.append((len(stream),m))
        stream.extend(motifs[m])

    # Verify every external symbol is represented.
    used=set(stream)
    missing=[s for s in SYMBOL_NAMES if s not in used]
    if missing:
        raise AssertionError(("wide generator omitted symbols",missing))

    return tuple(stream),motifs,macro,motif_boundaries


def relation_snapshot(f):
    rel=[n for n in f.nethra if n.routes]
    depth=max((f.birth_depth.get(r,0) for r in rel),default=0)
    hist=Counter(f.birth_depth.get(r,0) for r in rel)
    return {
        "nethra":len(f.nethra),
        "relations":len(rel),
        "max_depth":depth,
        "depth_histogram":dict(sorted(hist.items())),
    }


def feed(f,time_node,node):
    time_node.push(1.0)
    node.push(1.0)
    f.step(DT)


def train():
    stream,motifs,macro,bounds=make_supercycle()
    print("GENERATOR",json.dumps({
        "alphabet_size":N_SYMBOLS,
        "motifs":N_MOTIFS,
        "motif_length":MOTIF_LEN,
        "motif_placements_per_supercycle":len(macro),
        "supercycle_length":len(stream),
        "unique_symbols_used":len(set(stream)),
        "per_cycle_randomness":False,
    },sort_keys=True),flush=True)

    f=StableDepthAuditField(
        g_min=.20,g_max=1.50,tau=100.0,
        capacitance=1.0,leakage=.6,convergence_gain=0.0,
    )
    time_node=f.new()
    symbols={name:f.new() for name in SYMBOL_NAMES}

    stable=0
    prev_rel=None
    prev_depth=None
    t0=time.perf_counter()
    final_cycle=0

    report={1,2,4,8,12,16,20,24,32,48,64,80,96}

    for cycle in range(1,MAX_CYCLES+1):
        for name in stream:
            feed(f,time_node,symbols[name])

        snap=relation_snapshot(f)
        same=(snap["relations"]==prev_rel and snap["max_depth"]==prev_depth)
        stable=stable+1 if same else 0
        prev_rel=snap["relations"]
        prev_depth=snap["max_depth"]
        final_cycle=cycle

        if cycle in report or stable==PLATEAU_CYCLES or cycle==MAX_CYCLES:
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
            raise RuntimeError(("relation safety exceeded",snap["relations"]))

        if cycle>=MIN_CYCLES and stable>=PLATEAU_CYCLES:
            break

    snap=relation_snapshot(f)
    print("PLATEAU",json.dumps({
        "cycle":final_cycle,
        "intervals":final_cycle*len(stream),
        "relations":snap["relations"],
        "nethra":snap["nethra"],
        "max_depth":snap["max_depth"],
        "stable_cycles":stable,
        "depth_histogram":snap["depth_histogram"],
        "seconds":time.perf_counter()-t0,
    },sort_keys=True),flush=True)
    return f,time_node,symbols,stream,snap


def passive_predict(f,time_node,symbols):
    live_index={n:i for i,n in enumerate(f.nethra)}
    shadow=copy.deepcopy(f)
    stime=shadow.nethra[live_index[time_node]]
    ss={name:shadow.nethra[live_index[node]] for name,node in symbols.items()}

    before={name:n.activation for name,n in ss.items()}
    stime.push(1.0)
    shadow.step(DT)

    decay=math.exp(-shadow.leakage*DT/shadow.capacitance)
    scores={name:ss[name].activation-before[name]*decay for name in SYMBOL_NAMES}
    order=sorted(SYMBOL_NAMES,key=scores.get,reverse=True)
    return scores,order


def audit(f,time_node,symbols,stream,plateau):
    rows=[]
    for step in range(AUDIT_STEPS):
        actual=stream[step%len(stream)]

        scores,order=passive_predict(f,time_node,symbols)
        rank=order.index(actual)+1
        top1,top2=order[0],order[1]
        row={
            "step":step+1,
            "actual":actual,
            "rank":rank,
            "top_k":[(x,scores[x]) for x in order[:TOPK]],
            "actual_score":scores[actual],
            "margin":scores[top1]-scores[top2],
            "correct_top1":rank==1,
        }
        rows.append(row)
        print("PRED",json.dumps(row,sort_keys=True),flush=True)

        # Only now reveal the real next observation.
        feed(f,time_node,symbols[actual])

    n=len(rows)
    ranks=[r["rank"] for r in rows]
    summary={
        "n":n,
        "top1":sum(r==1 for r in ranks)/n,
        "top3":sum(r<=3 for r in ranks)/n,
        "top5":sum(r<=5 for r in ranks)/n,
        "mean_rank":sum(ranks)/n,
        "median_rank":sorted(ranks)[n//2],
        "mean_margin":sum(r["margin"] for r in rows)/n,
        "plateau_relations":plateau["relations"],
        "plateau_depth":plateau["max_depth"],
        "post_audit_relations":relation_snapshot(f)["relations"],
        "post_audit_depth":relation_snapshot(f)["max_depth"],
    }
    print("PREDICTION_SUMMARY",json.dumps(summary,sort_keys=True),flush=True)
    return summary


def main():
    t0=time.perf_counter()
    f,time_node,symbols,stream,plateau=train()
    summary=audit(f,time_node,symbols,stream,plateau)
    print("FINAL",json.dumps({
        "alphabet_size":N_SYMBOLS,
        "supercycle_length":len(stream),
        "plateau":plateau,
        "prediction":summary,
        "seconds_total":time.perf_counter()-t0,
    },sort_keys=True),flush=True)
    print("all_assertions_passed",flush=True)


if __name__=="__main__":
    main()
