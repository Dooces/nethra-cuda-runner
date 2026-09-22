#!/usr/bin/env python3
from __future__ import annotations
import math, os, random, sys
from pathlib import Path

PROJECT=Path("/home/dooces/Projects/20260918")
if not PROJECT.is_dir():
    raise RuntimeError(f"missing Fedora project dir: {PROJECT}")
sys.path.insert(0,str(PROJECT))

import nethra_L75_learning_core_frozen as L75
import nethra_F61_field_core_frozen as F61

L75.verify_frozen_dependencies()

DT=.01
PULSE=.30
SAMPLE_INTERVAL=.25
SAMPLES=32


def append_input_nethra(s,label):
    nid=len(s.nethra)
    s.nethra.append(F61.Nethra(nid,label))
    return nid


def delayed_timeline(lag:int,trials:int,seed:int,effect=.92):
    A,B,N1,N2,Y,Z=range(6)
    rng=random.Random(seed); scheduled={}; timeline=[]
    for t in range(trials+lag+2):
        u=rng.random()
        if u<.20:
            ex=frozenset((A,B))
        elif u<.35:
            ex=frozenset((A,N1))
        elif u<.50:
            ex=frozenset((B,N2))
        else:
            ex=frozenset((N1,N2))
        if ex==frozenset((A,B)):
            scheduled[t+lag]=Y if rng.random()<effect else Z
        y=scheduled.get(t,Y if rng.random()<.25 else Z)
        timeline.append((ex,y))
    return timeline


def crystallize_exact_old(lag:int, seed:int):
    A,B,N1,N2,Y,Z=range(6)
    q=frozenset((A,B))
    timeline=delayed_timeline(lag,9000,seed)
    s=L75.ApplicabilityStructure(["A","B","N1","N2","Y","Z"])

    lag_ids=[]
    for d in range(1,lag+1):
        lag_ids.append(append_input_nethra(s,f"delta_t[{d}]"))
    lag_nid=lag_ids[-1]

    source_context,_=s.add_relation(q,"remembered-source-context")
    temporal_nid,_=s.add_relation((source_context,lag_nid),f"temporal-context@{lag}")
    perspective_nid,_=s.add_relation((temporal_nid,Y),f"temporal-perspective@{lag}")
    s.register_perspective_condition(
        perspective_nid,temporal_nid,Y,frozenset((temporal_nid,))
    )

    evidence=0
    for t in range(lag,len(timeline)):
        past_ex=timeline[t-lag][0]
        y_now=timeline[t][1]
        if q<=past_ex and y_now==Y:
            rr=L75.SourceRecord(
                L75.v56.Episode(s.closure({temporal_nid}),y_now,"temporal-crystal"),
                frozenset((temporal_nid,))
            )
            s.strengthen_from_explicit_episode({temporal_nid,y_now})
            s.strengthen_applicability_episode(rr)
            evidence+=1

    return dict(
        s=s,A=A,B=B,Y=Y,Z=Z,lag=lag,lag_nid=lag_nid,
        source_context=source_context,temporal_nid=temporal_nid,
        perspective_nid=perspective_nid,evidence=evidence,
    )


def rk4_step(field,a,j,trace,dt=DT):
    y0=tuple(a);k1=field.derivative(y0,j,trace)
    y1=tuple(y+.5*dt*k for y,k in zip(y0,k1));k2=field.derivative(y1,j,trace)
    y2=tuple(y+.5*dt*k for y,k in zip(y0,k2));k3=field.derivative(y2,j,trace)
    y3=tuple(y+dt*k for y,k in zip(y0,k3));k4=field.derivative(y3,j,trace)
    return tuple(y+dt*(u+2*v+2*w+z)/6 for y,u,v,w,z in zip(y0,k1,k2,k3,k4))


def advance(field,a,jdict,duration,trace):
    j=[0.0]*len(field.nethra)
    for nid,v in jdict.items():j[nid]=float(v)
    j=tuple(j)
    for _ in range(round(duration/DT)):
        a=rk4_step(field,a,j,trace,DT)
    return a


def manual_temporal_control(cr):
    s=cr["s"];T=cr["temporal_nid"];Y=cr["Y"];Z=cr["Z"]
    f=s.contextual_freeze({T})
    tr=f.initial_trace_state()
    a=(0.0,)*len(f.nethra)
    a=advance(f,a,{T:1.0},.75,tr)
    return a[Y],a[Z],a[Y]-a[Z]


def autonomous_curve(cr):
    s=cr["s"];A=cr["A"];B=cr["B"];Y=cr["Y"];Z=cr["Z"]
    # During the cue, the only independent source support is X={A,B}.
    f_on=s.contextual_freeze({A,B})
    # After the cue, no temporal provenance transducer and no source support exist.
    f_off=s.contextual_freeze(set())
    tr=f_on.initial_trace_state()
    a=(0.0,)*len(f_on.nethra)
    a=advance(f_on,a,{A:1.0,B:1.0},PULSE,tr)

    rows=[]
    elapsed=0.0
    rows.append((elapsed,a[Y],a[Z],a[Y]-a[Z],a[cr["source_context"]],a[cr["temporal_nid"]]))
    for _ in range(SAMPLES):
        a=advance(f_off,a,{},SAMPLE_INTERVAL,tr)
        elapsed+=SAMPLE_INTERVAL
        rows.append((elapsed,a[Y],a[Z],a[Y]-a[Z],a[cr["source_context"]],a[cr["temporal_nid"]]))
    return rows


def source_only_same_duration_unlearned(lag:int):
    # Same structural graph but no replayed temporal evidence.
    A,B,N1,N2,Y,Z=range(6)
    s=L75.ApplicabilityStructure(["A","B","N1","N2","Y","Z"])
    lag_ids=[append_input_nethra(s,f"delta_t[{d}]") for d in range(1,lag+1)]
    lag_nid=lag_ids[-1]
    source_context,_=s.add_relation((A,B),"remembered-source-context")
    temporal_nid,_=s.add_relation((source_context,lag_nid),f"temporal-context@{lag}")
    perspective_nid,_=s.add_relation((temporal_nid,Y),f"temporal-perspective@{lag}")
    s.register_perspective_condition(perspective_nid,temporal_nid,Y,frozenset((temporal_nid,)))
    cr=dict(s=s,A=A,B=B,Y=Y,Z=Z,lag=lag,lag_nid=lag_nid,
            source_context=source_context,temporal_nid=temporal_nid,perspective_nid=perspective_nid,evidence=0)
    return autonomous_curve(cr)


def peak(curve,col=1):
    # Return elapsed time and value for absolute Y activation and target margin.
    row=max(curve,key=lambda r:r[col])
    return row[0],row[col]


def interpolate_sample(curve,interval_index):
    # One "interval" in this audit is SAMPLE_INTERVAL of free F61 evolution.
    idx=min(interval_index,len(curve)-1)
    return curve[idx]


def max_curve_diff(a,b,col=1):
    return max(abs(x[col]-y[col]) for x,y in zip(a,b))


def main():
    lags=(1,3,5,8)
    crystals={lag:crystallize_exact_old(lag,74000+lag) for lag in lags}

    print("=== OLD MANUAL TEMPORAL-HANDLE CONTROL ===")
    for lag,cr in crystals.items():
        print("lag",lag,"evidence",cr["evidence"],"manual_temporal",manual_temporal_control(cr))

    print("=== AUTONOMOUS X-ONLY, NO TEMPORAL PROVENANCE ===")
    curves={}
    for lag,cr in crystals.items():
        c=autonomous_curve(cr);curves[lag]=c
        py=peak(c,1);pm=peak(c,3)
        learned_row=interpolate_sample(c,lag)
        print("lag",lag,
              "Y_peak",py,
              "margin_peak",pm,
              "at_nominal_lag",learned_row,
              "curve",[(round(r[0],2),r[1],r[3]) for r in c])

    print("=== CROSS-LAG CURVE DIFFERENCES ===")
    base=curves[1]
    for lag in (3,5,8):
        print("lag1_vs",lag,
              "max_Y_diff",max_curve_diff(base,curves[lag],1),
              "max_margin_diff",max_curve_diff(base,curves[lag],3))

    print("=== LEARNED VS UNLEARNED LAG3 ===")
    un=source_only_same_duration_unlearned(3)
    print("learned_peak",peak(curves[3],1),"unlearned_peak",peak(un,1))
    print("max_Y_diff",max_curve_diff(curves[3],un,1),
          "max_margin_diff",max_curve_diff(curves[3],un,3))

    # Key verdicts.
    manual_ok=all(manual_temporal_control(cr)[2]>0 for cr in crystals.values())
    peak_times={lag:peak(c,1)[0] for lag,c in curves.items()}
    tracks_lag=len(set(round(v,8) for v in peak_times.values()))>1
    curve_distinct=max(
        max_curve_diff(curves[a],curves[b],1)
        for i,a in enumerate(lags) for b in lags[i+1:]
    )
    learned_changes_autonomous=max_curve_diff(curves[3],un,1)

    print("manual_control_positive",manual_ok)
    print("autonomous_peak_times",peak_times)
    print("autonomous_tracks_learned_lag",tracks_lag)
    print("max_cross_lag_Y_curve_difference",curve_distinct)
    print("learned_evidence_changes_X_only_lag3_curve",learned_changes_autonomous)

    assert manual_ok
    print("all_assertions_passed")


if __name__=="__main__":
    main()
