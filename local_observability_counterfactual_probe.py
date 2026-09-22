#!/usr/bin/env python3
"""Counterfactual local-observability experiment for Nethra.

For every physical incidence and both local orientations, compare:

    L_tau(i->j) = sum_k Phi_ij(k) * epsilon_j(k+tau)

against the actual replay sensitivity of the SAME accumulated residual loss at j:

    G_ij(j) = -[E_j(g_ij+h)-E_j(g_ij-h)]/(2h).

The full replay re-integrates the entire Nethra field under identical external input sequences.
Changing g therefore changes all subsequent activation trajectories, conductive flows, loading,
and F61 convergence.  L sees only completed-interval local Phi and receiver residual history.

Two consequence observables are kept separate:
- source:         target_j(k) = exact external source charge U_j(k)
- manifestation: target_j(k) = C * Delta a_j(k)

The field-carried predictor P_j(k) is the EXACT total internal field charge during interval k:

    P_j(k) = C*Delta a_j(k) - U_j(k) + leakage*A_j(k)

which includes ordinary conductive flow AND F61 convergence by integrated field balance.

Thus:
    epsilon_j(k+1) = target_j(k+1) - P_j(k)

This script is an audit only.  It installs no plasticity, matcher, selector, or learned kernel.

It measures:
- exact L_1 vs G;
- lagged local observables L_tau through HMAX;
- uniform accumulated horizons;
- fixed geometric eligibility kernels;
- a diagnostic linear kernel fitted on training seeds and evaluated on held-out seeds.

A fitted diagnostic kernel is NOT proposed as Nethra machinery.  It only answers whether a compact
temporal eligibility kernel could, in principle, account for the missing trajectory sensitivity.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
import os
import random
from statistics import median

from nethra import NethraField


INTERVALS=40
SUBSTEPS=14
DT_INTERVAL=.20
DT=DT_INTERVAL/SUBSTEPS
HMAX=8
FD_H=1e-4
DEPTHS=(1,3,6)
DELAYS=(1,3)
TOPOLOGIES=("chain","loop","skip")
CONVERGENCE=(0.0,1.0)
MODES=("source","manifestation")
SEEDS=(0,1)


@dataclass
class Spec:
    depth:int
    delay:int
    topology:str
    convergence:float
    mode:str
    seed:int


def topology_spec(depth,kind,seed):
    """Return node count, physical edges [i,j,g], with deterministic nondegenerate g."""
    rng=random.Random(60000+seed*97+depth*11+sum(map(ord,kind)))
    n=depth+3
    edges=[]
    for i in range(n-1):
        edges.append([i,i+1,0.32+0.65*rng.random()])
    if kind=="loop":
        edges.append([n-1,0,0.30+0.55*rng.random()])
    elif kind=="skip":
        for i in range(n-2):
            edges.append([i,i+2,0.22+0.42*rng.random()])
    return n,edges


def input_sequence(n,delay,seed,mode):
    rng=random.Random(70000+seed*131+delay*17+(0 if mode=="source" else 991))
    latent=[rng.uniform(-1.0,1.0) for _ in range(INTERVALS+delay+HMAX+5)]
    noise=[[rng.uniform(-1.0,1.0) for _ in range(INTERVALS)] for _ in range(n)]

    seq=[]
    for k in range(INTERVALS):
        row=[0.0]*n
        # Always excite the first node.
        row[0]=0.85*latent[k+delay]
        if mode=="source":
            # Every other node receives a delayed external trace plus small private disturbance.
            # No learner sees delay or latent identity; this is only the physical replay fixture.
            for j in range(1,n):
                idx=max(0,k+delay-delay)
                row[j]=(0.46+0.04*(j%3))*latent[idx] + 0.06*noise[j][k]
        else:
            # Recursive/internal manifestation fixture: only primitive boundary nodes are sourced.
            # A second primitive near the middle prevents the network from being a one-source toy.
            if n>3:
                row[n//2]=0.28*latent[max(0,k+delay-delay)] + 0.04*noise[n//2][k]
        seq.append(row)
    return seq


def build_field(n,edge_rows,convergence):
    f=NethraField(leakage=.55,capacitance=1.0,convergence_gain=convergence)
    nodes=[f.new() for _ in range(n)]
    def edge_fn():
        return tuple((nodes[i],nodes[j],g) for i,j,g in edge_rows)
    f._edges=edge_fn

    if convergence>0.0:
        # Controlled F61 fixture: every supplier pair has orthogonal prior residual history.
        # This makes convergence physically active wherever simultaneous positive suppliers meet.
        for i in range(n):
            for j in range(i+1,n):
                f.pair_stats[frozenset((nodes[i],nodes[j]))]=(0.0,1.0,1.0,20)
    return f,nodes


def integrate_interval(f,nodes,currents):
    for node,j in zip(nodes,currents):
        node.external=float(j)

    a_start={n:n.activation for n in nodes}
    A={n:0.0 for n in nodes}

    for _ in range(SUBSTEPS):
        a0={n:n.activation for n in nodes}
        k1=f._derivative_at(a0)
        a1={n:a0[n]+.5*DT*k1[n] for n in nodes}
        k2=f._derivative_at(a1)
        a2={n:a0[n]+.5*DT*k2[n] for n in nodes}
        k3=f._derivative_at(a2)
        a3={n:a0[n]+DT*k3[n] for n in nodes}
        k4=f._derivative_at(a3)

        for n in nodes:
            A[n]+=DT*(a0[n]+2*a1[n]+2*a2[n]+a3[n])/6.0
            n.activation=a0[n]+DT*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.0

    delta={n:n.activation-a_start[n] for n in nodes}
    U={n:DT_INTERVAL*float(j) for n,j in zip(nodes,currents)}
    P={
        n:f.capacitance*delta[n]-U[n]+f.leakage*A[n]
        for n in nodes
    }

    for n in nodes:
        n.external=0.0

    return A,delta,U,P


def replay(spec,edge_override=None):
    n,edges=topology_spec(spec.depth,spec.topology,spec.seed)
    if edge_override is not None:
        idx,newg=edge_override
        edges=[row[:] for row in edges]
        edges[idx][2]=newg

    f,nodes=build_field(n,edges,spec.convergence)
    seq=input_sequence(n,spec.delay,spec.seed,spec.mode)

    As=[]
    Ds=[]
    Us=[]
    Ps=[]
    for currents in seq:
        A,d,U,P=integrate_interval(f,nodes,currents)
        As.append([A[n] for n in nodes])
        Ds.append([d[n] for n in nodes])
        Us.append([U[n] for n in nodes])
        Ps.append([P[n] for n in nodes])

    target=Us if spec.mode=="source" else [
        [f.capacitance*x for x in row] for row in Ds
    ]

    eps=[[0.0]*INTERVALS for _ in range(n)]
    loss=[0.0]*n
    for j in range(n):
        for t in range(1,INTERVALS):
            e=target[t][j]-Ps[t-1][j]
            eps[j][t]=e
            loss[j]+=.5*e*e

    return {
        "n":n,"edges":edges,"A":As,"D":Ds,"U":Us,"P":Ps,
        "eps":eps,"loss":loss,
    }


def lag_features(base,edge_idx,src_idx,dst_idx):
    i,j,g=base["edges"][edge_idx]
    assert {i,j}=={src_idx,dst_idx}
    sign_phi=1.0 if (i==src_idx and j==dst_idx) else -1.0

    phi=[
        sign_phi*(base["A"][k][i]-base["A"][k][j])
        for k in range(INTERVALS)
    ]
    eps=base["eps"][dst_idx]

    features=[]
    for tau in range(1,HMAX+1):
        s=0.0
        for k in range(0,INTERVALS-tau):
            s+=phi[k]*eps[k+tau]
        features.append(s)
    return features


def scenario(spec):
    base=replay(spec)
    samples=[]

    for ei,(i,j,g) in enumerate(base["edges"]):
        plus=replay(spec,(ei,g+FD_H))
        minus=replay(spec,(ei,g-FD_H))

        # Both local orientations of the same physical incidence.
        for src,dst in ((i,j),(j,i)):
            G=-(plus["loss"][dst]-minus["loss"][dst])/(2*FD_H)
            X=lag_features(base,ei,src,dst)
            samples.append({
                "edge":ei,
                "src":src,
                "dst":dst,
                "g":g,
                "G":G,
                "L":X,
            })

    return {
        "spec":spec.__dict__,
        "samples":samples,
    }


def sign(x,tol=1e-11):
    if x>tol:return 1
    if x<-tol:return -1
    return 0


def pearson(xs,ys):
    if len(xs)<2:return float("nan")
    mx=sum(xs)/len(xs); my=sum(ys)/len(ys)
    xx=sum((x-mx)**2 for x in xs)
    yy=sum((y-my)**2 for y in ys)
    if xx<=0 or yy<=0:return float("nan")
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/math.sqrt(xx*yy)


def metrics(rows,predict):
    gs=[]; ps=[]; rel=[]; signs=[]
    for r in rows:
        g=r["G"]; p=predict(r)
        gs.append(g); ps.append(p)
        if abs(g)>1e-10:
            rel.append(abs(p-g)/abs(g))
        sg,sp=sign(g),sign(p)
        if sg and sp:
            signs.append(int(sg==sp))
    return {
        "n":len(rows),
        "corr":pearson(ps,gs),
        "sign_agree":sum(signs)/len(signs) if signs else None,
        "median_rel_error":median(rel) if rel else None,
        "mean_abs_G":sum(abs(x) for x in gs)/len(gs) if gs else None,
        "mean_abs_pred":sum(abs(x) for x in ps)/len(ps) if ps else None,
    }


def solve_ridge(X,y,lam=1e-8):
    p=len(X[0])
    A=[[0.0]*p for _ in range(p)]
    b=[0.0]*p
    for row,target in zip(X,y):
        for i in range(p):
            b[i]+=row[i]*target
            for j in range(p):
                A[i][j]+=row[i]*row[j]
    scale=sum(A[i][i] for i in range(p))/max(1,p)
    reg=lam*max(scale,1e-12)
    for i in range(p): A[i][i]+=reg

    # Gaussian elimination with partial pivot.
    aug=[A[i]+[b[i]] for i in range(p)]
    for col in range(p):
        pivot=max(range(col,p),key=lambda r:abs(aug[r][col]))
        aug[col],aug[pivot]=aug[pivot],aug[col]
        den=aug[col][col]
        if abs(den)<1e-20: continue
        inv=1.0/den
        aug[col]=[v*inv for v in aug[col]]
        for r in range(p):
            if r==col: continue
            fac=aug[r][col]
            if fac:
                aug[r]=[aug[r][c]-fac*aug[col][c] for c in range(p+1)]
    return [aug[i][-1] for i in range(p)]


def dot(a,b): return sum(x*y for x,y in zip(a,b))


def grouped(rows,keyfn):
    out={}
    for r in rows:
        out.setdefault(keyfn(r),[]).append(r)
    return out


def main():
    specs=[
        Spec(depth,delay,topology,conv,mode,seed)
        for depth in DEPTHS
        for delay in DELAYS
        for topology in TOPOLOGIES
        for conv in CONVERGENCE
        for mode in MODES
        for seed in SEEDS
    ]

    workers=min(16,max(1,os.cpu_count() or 1),len(specs))
    results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(scenario,s):s for s in specs}
        for fut in as_completed(futures):
            results.append(fut.result())

    rows=[]
    for result in results:
        spec=result["spec"]
        for sample in result["samples"]:
            rows.append({**spec,**sample})

    # Core metrics.
    summary={
        "contract":"NETHRA_LOCAL_OBSERVABILITY_COUNTERFACTUAL_1",
        "scenario_count":len(results),
        "sample_count":len(rows),
        "L1":metrics(rows,lambda r:r["L"][0]),
    }

    for h in (1,2,3,4,6,8):
        summary[f"uniform_H{h}"]=metrics(rows,lambda r,h=h:sum(r["L"][:h]))

    for rho in (.25,.5,.75,.9):
        weights=[rho**i for i in range(HMAX)]
        summary[f"geom_{rho}"]=metrics(rows,lambda r,w=weights:dot(r["L"],w))

    # Fit diagnostic eligibility kernels on seed 0, evaluate only on held-out seed 1.
    kernel={}
    for mode in ("all",)+MODES:
        train=[r for r in rows if r["seed"]==0 and (mode=="all" or r["mode"]==mode)]
        test=[r for r in rows if r["seed"]==1 and (mode=="all" or r["mode"]==mode)]
        K=solve_ridge([r["L"] for r in train],[r["G"] for r in train])
        kernel[mode]={
            "K":K,
            "train":metrics(train,lambda r,K=K:dot(r["L"],K)),
            "test":metrics(test,lambda r,K=K:dot(r["L"],K)),
        }
    summary["fitted_kernel"]=kernel

    # Horizon sweep: fit only the first H local lags on seed 0, evaluate on seed 1.
    horizon_fit={}
    for mode in MODES:
        horizon_fit[mode]={}
        train=[r for r in rows if r["seed"]==0 and r["mode"]==mode]
        test=[r for r in rows if r["seed"]==1 and r["mode"]==mode]
        for h in range(1,HMAX+1):
            K=solve_ridge([r["L"][:h] for r in train],[r["G"] for r in train])
            horizon_fit[mode][str(h)]={
                "K":K,
                "test":metrics(test,lambda r,K=K,h=h:dot(r["L"][:h],K)),
            }
    summary["horizon_fit"]=horizon_fit

    # Structural holdouts. Fit source-mode kernels on all remaining structure/seeds and evaluate
    # the unseen topology/depth/delay. This is diagnostic generalization, not model selection.
    structural={}
    source_rows=[r for r in rows if r["mode"]=="source"]
    holdouts=[
        ("topology","loop"),
        ("topology","skip"),
        ("depth",6),
        ("delay",3),
        ("convergence",1.0),
    ]
    for field,value in holdouts:
        train=[r for r in source_rows if r[field]!=value]
        test=[r for r in source_rows if r[field]==value]
        K=solve_ridge([r["L"] for r in train],[r["G"] for r in train])
        K4=solve_ridge([r["L"][:4] for r in train],[r["G"] for r in train])
        structural[f"{field}={value}"]={
            "K":K,
            "test":metrics(test,lambda r,K=K:dot(r["L"],K)),
            "K4":K4,
            "test_H4":metrics(test,lambda r,K4=K4:dot(r["L"][:4],K4)),
        }
    summary["source_structural_holdout"]=structural

    # Group the exact one-step observable by each requested stress axis.
    group_metrics={}
    for name,keyfn in (
        ("depth",lambda r:str(r["depth"])),
        ("delay",lambda r:str(r["delay"])),
        ("topology",lambda r:r["topology"]),
        ("convergence",lambda r:str(r["convergence"])),
        ("mode",lambda r:r["mode"]),
        ("depth_delay",lambda r:f"{r['depth']}/{r['delay']}"),
    ):
        group_metrics[name]={
            key:metrics(group,lambda r:r["L"][0])
            for key,group in grouped(rows,keyfn).items()
        }
    summary["L1_groups"]=group_metrics

    # Lag-by-lag diagnostic: which temporal offset carries the sensitivity?
    summary["lags"]={
        str(tau):metrics(rows,lambda r,t=tau:r["L"][t-1])
        for tau in range(1,HMAX+1)
    }

    # Save enough raw data to inspect failures without rerunning the field.
    with open("local_observability_counterfactual_results.json","w") as fh:
        json.dump({"summary":summary,"rows":rows},fh,indent=2)

    print("=== CORE ===")
    for key in ("L1","uniform_H2","uniform_H4","uniform_H8","geom_0.5","geom_0.9"):
        print(key,summary[key])

    print("=== GROUPS L1 ===")
    for name,groups in group_metrics.items():
        print(name)
        for key,val in sorted(groups.items()):
            print(" ",key,val)

    print("=== LAGS ===")
    for key,val in summary["lags"].items():
        print(key,val)

    print("=== FITTED KERNELS ===")
    for mode,val in kernel.items():
        print(mode,"K",val["K"])
        print(mode,"train",val["train"])
        print(mode,"test",val["test"])

    print("=== HORIZON FIT HELD SEED ===")
    for mode,hs in horizon_fit.items():
        for h,val in hs.items():
            print(mode,"H",h,"K",val["K"],"test",val["test"])

    print("=== SOURCE STRUCTURAL HOLDOUT ===")
    for name,val in structural.items():
        print(name,"K",val["K"],"test",val["test"])
        print(name,"K4",val["K4"],"test_H4",val["test_H4"])

    # Print the largest exact L1 sign failures for inspection.
    fails=[]
    for r in rows:
        if sign(r["G"]) and sign(r["L"][0]) and sign(r["G"])!=sign(r["L"][0]):
            fails.append((abs(r["G"]),r))
    fails.sort(reverse=True,key=lambda x:x[0])
    print("L1_sign_failures",len(fails))
    for _,r in fails[:20]:
        print({
            "depth":r["depth"],"delay":r["delay"],"topology":r["topology"],
            "convergence":r["convergence"],"mode":r["mode"],"seed":r["seed"],
            "edge":r["edge"],"src":r["src"],"dst":r["dst"],
            "G":r["G"],"L1":r["L"][0],"L":r["L"],
        })

    print("all_assertions_passed")


if __name__=="__main__":
    main()
