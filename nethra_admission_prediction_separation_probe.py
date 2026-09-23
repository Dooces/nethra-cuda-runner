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


def prospective_current(f,leaves):
    """Read endogenous relation->leaf current from the live field without mutating it."""
    before=tuple((n.activation,n.external) for n in f.nethra)
    score={leaf:0.0 for leaf in leaves}
    contributions={leaf:[] for leaf in leaves}

    for a,b,g in f._edges():
        if a.routes and b in score:
            q=g*(a.activation-b.activation)
            if q>0.0:
                score[b]+=q
                contributions[b].append(q)
        elif b.routes and a in score:
            q=g*(b.activation-a.activation)
            if q>0.0:
                score[a]+=q
                contributions[a].append(q)

    after=tuple((n.activation,n.external) for n in f.nethra)
    assert before==after
    return score,contributions


def prediction_sweep():
    cycles_grid=(2,3,5,10,30,100,300)
    all_rows=[]
    for cycles in cycles_grid:
        rows=[]
        for cue in range(SYMBOLS):
            f,leaves=train_ending_at(cycles,cue)
            expected=(cue+1)%SYMBOLS
            score,parts=prospective_current(f,leaves)
            candidates=[i for i in range(SYMBOLS) if i!=cue]
            order=sorted(candidates,key=lambda i:score[leaves[i]],reverse=True)
            rank=order.index(expected)+1
            rival=max(score[leaves[i]] for i in candidates if i!=expected)
            expected_score=score[leaves[expected]]
            margin=expected_score-rival
            total=sum(score[leaves[i]] for i in candidates)
            normalized=margin/total if total>0.0 else 0.0
            row={
                "cycles":cycles,
                "cue":cue,
                "expected":expected,
                "relations":sum(bool(n.routes) for n in f.nethra),
                "expected_score":expected_score,
                "rival_score":rival,
                "margin":margin,
                "normalized_margin":normalized,
                "rank":rank,
                "scores":[score[n] for n in leaves],
                "expected_contributors":len(parts[leaves[expected]]),
            }
            rows.append(row)
            all_rows.append(row)
            print("PREDICTION",row)

        rank1=sum(r["rank"]==1 for r in rows)
        positive=sum(r["margin"]>0.0 for r in rows)
        mean_margin=sum(r["margin"] for r in rows)/len(rows)
        mean_norm=sum(r["normalized_margin"] for r in rows)/len(rows)
        print(
            "PREDICTION_SUMMARY",
            {"cycles":cycles,"rank1":rank1,"positive_margin":positive,
             "mean_margin":mean_margin,"mean_normalized_margin":mean_norm},
        )

    # Readout must remain a continuous observation: scores can be zero/negative-margin, and no
    # prediction threshold has authority to change topology or field state.
    assert all(math.isfinite(r["margin"]) for r in all_rows)
    print("PREDICTION_READOUT_MUTATION",False)
    return all_rows


def main():
    print("=== LEARNING_ADMISSION_ONLY ===")
    admission_sweep()
    print("=== CONTINUOUS_PREDICTION_CURRENT ===")
    prediction_sweep()
    print("all_assertions_passed")


if __name__=="__main__":
    main()
