#!/usr/bin/env python3
"""Keep admission confidence and prediction confidence mechanically separate.

A. Learning admission:
   Sweep one scalar admission threshold over the existing recursive-depth fixture. The threshold is
   consulted only before a new relation is materialized. Once admitted, ordinary signed plasticity
   runs continuously and never sees the threshold.

B. Prediction:
   Train the corrected core on a repeated A->B->C->D cycle. At the instant external input ends,
   read only endogenous positive relation->leaf current:

       C_m = sum_R max(0, g_Rm * (a_R - a_m))

   No threshold, winner switch, decoder, transition table, or outcome is fed back. Report rank and
   competition margin as experience grows.
"""
from __future__ import annotations
import math

from nethra import NethraField
from recursive_depth_gate_probe import learn_depth

SYMBOLS=4
DT=.15


def admission_sweep():
    thresholds=(0.0,1e-18,1e-15,1e-12,1e-10,1e-8,1e-6,1e-4)
    rows=[]
    for theta in thresholds:
        result=learn_depth(theta,phase=0)
        first=result["rows"][0] if result["rows"] else None
        row={
            "theta":theta,
            "learned_depth":result["learned_depth"],
            "stop":result["stop_reason"],
            "first_initial":None if first is None else first["initial_prediction"],
            "first_final":None if first is None else first["final_prediction"],
            "first_evidence":None if first is None else first["evidence"],
            "first_admission_score":None if first is None else first["admission_score"],
        }
        rows.append(row)
        print("ADMISSION",row)

    depths=[r["learned_depth"] for r in rows]
    # Increasing the admission threshold can only remove admissions in this deterministic fixture.
    assert all(depths[i]>=depths[i+1] for i in range(len(depths)-1)),depths

    # For every threshold that admits depth 1, the depth-1 plasticity trajectory is identical:
    # the threshold is not consulted again once the relation exists.
    admitted=[r for r in rows if r["first_final"] is not None]
    if admitted:
        ref=admitted[0]
        for r in admitted[1:]:
            assert abs(r["first_final"]-ref["first_final"])<1e-15
            assert abs(r["first_evidence"]-ref["first_evidence"])<1e-12
    print("ADMISSION_DEPTHS",depths)
    print("ADMISSION_POST_MATERIALIZATION_IDENTICAL",True)
    return rows


def train_ending_at(cycles,cue):
    """Same cyclic experience, phase-rotated so the final observed source is cue."""
    f=NethraField(
        g_min=.20,g_max=1.50,tau=100.0,
        capacitance=1.0,leakage=.6,convergence_gain=0.0,
    )
    leaves=[f.new() for _ in range(SYMBOLS)]
    start=(cue+1)%SYMBOLS
    for k in range(cycles*SYMBOLS):
        i=(start+k)%SYMBOLS
        leaves[i].push(1.0)
        f.step(DT)
    assert (start+cycles*SYMBOLS-1)%SYMBOLS==cue
    for n in f.nethra:
        n.external=0.0
    return f,leaves


def integrated_prospective_current(f,leaves,duration):
    """Integrate relation->leaf endogenous charge over a finite source-free interval.

    The original learned field is not mutated. No observation, learning, construction, threshold,
    or symbolic continuation is run during the prospective interval.
    """
    import copy
    g=copy.deepcopy(f)
    gleaves=g.nethra[:SYMBOLS]
    score={leaf:0.0 for leaf in gleaves}
    steps=max(1,round(duration/.005))
    dt=duration/steps

    for n in g.nethra:
        n.external=0.0

    for _ in range(steps):
        edges=g._edges()

        def accumulate(state,weight):
            for a,b,conductance in edges:
                if a.routes and b in score:
                    q=conductance*(state[a]-state[b])
                    if q>0.0: score[b]+=weight*q
                elif b.routes and a in score:
                    q=conductance*(state[b]-state[a])
                    if q>0.0: score[a]+=weight*q

        a0={n:n.activation for n in g.nethra}
        k1=g._derivative_at(a0)
        a1={n:a0[n]+.5*dt*k1[n] for n in g.nethra};k2=g._derivative_at(a1)
        a2={n:a0[n]+.5*dt*k2[n] for n in g.nethra};k3=g._derivative_at(a2)
        a3={n:a0[n]+dt*k3[n] for n in g.nethra};k4=g._derivative_at(a3)

        accumulate(a0,dt/6.0)
        accumulate(a1,dt/3.0)
        accumulate(a2,dt/3.0)
        accumulate(a3,dt/6.0)

        for n in g.nethra:
            n.activation=a0[n]+dt*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.0

    return [score[n] for n in gleaves]


def score_row(cycles,cue,horizon):
    f,leaves=train_ending_at(cycles,cue)
    expected=(cue+1)%SYMBOLS
    scores=integrated_prospective_current(f,leaves,horizon)
    candidates=[i for i in range(SYMBOLS) if i!=cue]
    order=sorted(candidates,key=lambda i:scores[i],reverse=True)
    rival=max(scores[i] for i in candidates if i!=expected)
    expected_score=scores[expected]
    margin=expected_score-rival
    total=sum(scores[i] for i in candidates)
    return {
        "cycles":cycles,
        "horizon":horizon,
        "cue":cue,
        "expected":expected,
        "relations":sum(bool(n.routes) for n in f.nethra),
        "expected_score":expected_score,
        "rival_score":rival,
        "margin":margin,
        "normalized_margin":margin/total if total>0.0 else 0.0,
        "rank":order.index(expected)+1,
        "scores":scores,
    }


def prediction_sweep():
    # First hold the prospective interval fixed at one observed interval and vary experience.
    cycles_grid=(2,3,5,10,30,100,300)
    all_rows=[]
    for cycles in cycles_grid:
        rows=[score_row(cycles,cue,DT) for cue in range(SYMBOLS)]
        all_rows.extend(rows)
        for row in rows: print("PREDICTION",row)
        print("PREDICTION_SUMMARY",{
            "cycles":cycles,
            "horizon":DT,
            "rank1":sum(r["rank"]==1 for r in rows),
            "positive_margin":sum(r["margin"]>0.0 for r in rows),
            "mean_margin":sum(r["margin"] for r in rows)/len(rows),
            "mean_normalized_margin":sum(r["normalized_margin"] for r in rows)/len(rows),
        })

    # Then hold mature experience fixed and vary only how long the source-free field is allowed to
    # express its continuation. This is the finite-interval analogue of "confidence develops".
    for horizon in (.025,.05,.10,.15,.30,.60,1.20):
        rows=[score_row(300,cue,horizon) for cue in range(SYMBOLS)]
        all_rows.extend(rows)
        print("HORIZON_SUMMARY",{
            "cycles":300,
            "horizon":horizon,
            "rank1":sum(r["rank"]==1 for r in rows),
            "positive_margin":sum(r["margin"]>0.0 for r in rows),
            "mean_margin":sum(r["margin"] for r in rows)/len(rows),
            "mean_normalized_margin":sum(r["normalized_margin"] for r in rows)/len(rows),
            "scores":[r["scores"] for r in rows],
        })

    assert all(math.isfinite(r["margin"]) for r in all_rows)
    print("PREDICTION_THRESHOLD_USED",False)
    return all_rows

def main():
    print("=== LEARNING_ADMISSION_ONLY ===")
    admission_sweep()
    print("=== CONTINUOUS_PREDICTION_CURRENT ===")
    prediction_sweep()
    print("all_assertions_passed")


if __name__=="__main__":
    main()
