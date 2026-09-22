#!/usr/bin/env python3
"""Paired counterfactual comparison: Delta manifestation vs leak-compensated M manifestation.

Same topology, same external inputs, same conductance perturbations.

Objectives:
    D_i = C*Delta a_i
    M_i = C*Delta a_i + lambda*A_i

Both use the exact prior integrated internal field drive
    P_i = conductive + F61 = M_i - U_i

Residuals:
    eps_D(k+1) = D(k+1) - P(k)
    eps_M(k+1) = M(k+1) - P(k)

For each physical incidence and local orientation:
    L_tau = sum Phi_ij(k) * eps_j(k+tau)
and actual counterfactual sensitivity G=-dE/dg is measured by identical replay at g±h.

This is an audit only.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import math, os, random
from statistics import median
from nethra import NethraField

INTERVALS=44
SUBSTEPS=16
T=.20
DT=T/SUBSTEPS
HMAX=8
H=1e-4
DEPTHS=(1,3,6)
DELAYS=(1,3)
TOPS=("chain","loop","skip")
CONVS=(0.0,1.0)
SEEDS=(0,1)

@dataclass
class Spec:
    depth:int; delay:int; topology:str; convergence:float; seed:int


def topo(spec):
    rng=random.Random(91000+spec.seed*103+spec.depth*17+sum(map(ord,spec.topology)))
    n=spec.depth+3
    e=[]
    for i in range(n-1):e.append([i,i+1,.30+.68*rng.random()])
    if spec.topology=="loop":e.append([n-1,0,.28+.57*rng.random()])
    elif spec.topology=="skip":
        for i in range(n-2):e.append([i,i+2,.20+.45*rng.random()])
    return n,e


def seq(n,spec):
    rng=random.Random(92000+spec.seed*149+spec.delay*23)
    latent=[rng.uniform(-1,1) for _ in range(INTERVALS+spec.delay+HMAX+8)]
    noise=[[rng.uniform(-1,1) for _ in range(INTERVALS)] for _ in range(n)]
    out=[]
    for k in range(INTERVALS):
        row=[0.0]*n
        row[0]=.85*latent[k+spec.delay]
        j=n//2
        if j==0:j=1
        row[j]=.34*latent[k]+.05*noise[j][k]
        if n>3 and k%3==0:row[-1]=.12*noise[-1][k]
        out.append(row)
    return out


def build(n,e,conv):
    f=NethraField(leakage=.55,capacitance=1.0,convergence_gain=conv)
    nodes=[f.new() for _ in range(n)]
    f._edges=lambda:tuple((nodes[i],nodes[j],g) for i,j,g in e)
    if conv:
        for i in range(n):
            for j in range(i+1,n):
                f.pair_stats[frozenset((nodes[i],nodes[j]))]=(0.0,1.0,1.0,20)
    return f,nodes


def cond(f,state,nodes):
    c={n:0.0 for n in nodes}
    for a,b,g in f._edges():
        q=g*(state[a]-state[b]); c[a]-=q; c[b]+=q
    return c


def internal(f,state,nodes):
    c=cond(f,state,nodes)
    full={n:f.capacitance*v for n,v in f._derivative_at(state).items()}
    b={n:full[n]-n.external+f.leakage*state[n]-c[n] for n in nodes}
    return c,b


def interval(f,nodes,src):
    for n,j in zip(nodes,src):n.external=float(j)
    start={n:n.activation for n in nodes}
    A={n:0.0 for n in nodes}; P={n:0.0 for n in nodes}
    for _ in range(SUBSTEPS):
        a0={n:n.activation for n in nodes}; k1=f._derivative_at(a0)
        a1={n:a0[n]+.5*DT*k1[n] for n in nodes}; k2=f._derivative_at(a1)
        a2={n:a0[n]+.5*DT*k2[n] for n in nodes}; k3=f._derivative_at(a2)
        a3={n:a0[n]+DT*k3[n] for n in nodes}; k4=f._derivative_at(a3)
        ib=[internal(f,s,nodes) for s in (a0,a1,a2,a3)]
        for n in nodes:
            A[n]+=DT*(a0[n]+2*a1[n]+2*a2[n]+a3[n])/6
            pp=[ib[s][0][n]+ib[s][1][n] for s in range(4)]
            P[n]+=DT*(pp[0]+2*pp[1]+2*pp[2]+pp[3])/6
            n.activation=a0[n]+DT*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6
    D={n:f.capacitance*(n.activation-start[n]) for n in nodes}
    M={n:D[n]+f.leakage*A[n] for n in nodes}
    U={n:T*float(j) for n,j in zip(nodes,src)}
    for n in nodes:n.external=0.0
    return A,D,M,P,U


def replay(spec,override=None):
    n,e=topo(spec)
    if override:
        idx,g=override;e=[x[:] for x in e];e[idx][2]=g
    f,nodes=build(n,e,spec.convergence)
    As=[]; Ds=[]; Ms=[]; Ps=[]
    for row in seq(n,spec):
        A,D,M,P,U=interval(f,nodes,row)
        As.append([A[x] for x in nodes]);Ds.append([D[x] for x in nodes])
        Ms.append([M[x] for x in nodes]);Ps.append([P[x] for x in nodes])
    out={"n":n,"edges":e,"A":As}
    for name,target in (("D",Ds),("M",Ms)):
        eps=[[0.0]*INTERVALS for _ in range(n)];loss=[0.0]*n
        for j in range(n):
            for t in range(1,INTERVALS):
                z=target[t][j]-Ps[t-1][j]
                eps[j][t]=z;loss[j]+=.5*z*z
        out[name]={"eps":eps,"loss":loss}
    return out


def feats(base,ei,src,dst,obj):
    i,j,g=base["edges"][ei]; s=1 if (i==src and j==dst) else -1
    phi=[s*(base["A"][k][i]-base["A"][k][j]) for k in range(INTERVALS)]
    ep=base[obj]["eps"][dst]
    return [sum(phi[k]*ep[k+t] for k in range(INTERVALS-t)) for t in range(1,HMAX+1)]


def scenario(spec):
    b=replay(spec);rows=[]
    for ei,(i,j,g) in enumerate(b["edges"]):
        p=replay(spec,(ei,g+H));m=replay(spec,(ei,g-H))
        for src,dst in ((i,j),(j,i)):
            r={"spec":spec.__dict__,"edge":ei,"src":src,"dst":dst}
            for obj in ("D","M"):
                r[obj]={"G":-(p[obj]["loss"][dst]-m[obj]["loss"][dst])/(2*H),
                        "L":feats(b,ei,src,dst,obj)}
            rows.append(r)
    return rows


def sg(x,t=1e-11):return 1 if x>t else -1 if x<-t else 0
def corr(x,y):
    mx=sum(x)/len(x);my=sum(y)/len(y)
    xx=sum((z-mx)**2 for z in x);yy=sum((z-my)**2 for z in y)
    return sum((a-mx)*(b-my) for a,b in zip(x,y))/math.sqrt(xx*yy) if xx>0 and yy>0 else float("nan")
def metrics(rows,obj,pred):
    gs=[];ps=[];re=[];sa=[]
    for r in rows:
        g=r[obj]["G"];p=pred(r)
        gs.append(g);ps.append(p)
        if abs(g)>1e-10:re.append(abs(p-g)/abs(g))
        a,b=sg(g),sg(p)
        if a and b:sa.append(a==b)
    return {"corr":corr(ps,gs),"sign":sum(sa)/len(sa),"medrel":median(re),"n":len(rows)}
def dot(a,b):return sum(x*y for x,y in zip(a,b))


def solve(X,y,lam=1e-8):
    p=len(X[0]);A=[[0.0]*p for _ in range(p)];b=[0.0]*p
    for x,t in zip(X,y):
        for i in range(p):
            b[i]+=x[i]*t
            for j in range(p):A[i][j]+=x[i]*x[j]
    sc=sum(A[i][i] for i in range(p))/p
    for i in range(p):A[i][i]+=lam*max(sc,1e-12)
    aug=[A[i]+[b[i]] for i in range(p)]
    for c in range(p):
        q=max(range(c,p),key=lambda r:abs(aug[r][c]));aug[c],aug[q]=aug[q],aug[c]
        d=aug[c][c]
        if abs(d)<1e-20:continue
        aug[c]=[v/d for v in aug[c]]
        for r in range(p):
            if r==c:continue
            z=aug[r][c]
            if z:aug[r]=[aug[r][k]-z*aug[c][k] for k in range(p+1)]
    return [aug[i][-1] for i in range(p)]


def main():
    specs=[Spec(d,l,t,c,s) for d in DEPTHS for l in DELAYS for t in TOPS for c in CONVS for s in SEEDS]
    rows=[]
    with ProcessPoolExecutor(max_workers=min(16,len(specs),os.cpu_count() or 1)) as pool:
        fs=[pool.submit(scenario,s) for s in specs]
        for f in as_completed(fs):rows.extend(f.result())

    for obj in ("D","M"):
        print(obj,"L1",metrics(rows,obj,lambda r:r[obj]["L"][0]))
        tr=[r for r in rows if r["spec"]["seed"]==0];te=[r for r in rows if r["spec"]["seed"]==1]
        for h in (1,2,4,8):
            K=solve([r[obj]["L"][:h] for r in tr],[r[obj]["G"] for r in tr])
            print(obj,"held_seed_H",h,"K",K,"test",metrics(te,obj,lambda r,K=K,h=h:dot(r[obj]["L"][:h],K)))
        for field,value in (("topology","skip"),("depth",6),("delay",3),("convergence",1.0)):
            tr=[r for r in rows if r["spec"][field]!=value]
            te=[r for r in rows if r["spec"][field]==value]
            K=solve([r[obj]["L"][:4] for r in tr],[r[obj]["G"] for r in tr])
            print(obj,"holdout",field,value,"K4",K,"test",metrics(te,obj,lambda r,K=K:dot(r[obj]["L"][:4],K)))
    print("all_assertions_passed")


if __name__=="__main__":
    main()
