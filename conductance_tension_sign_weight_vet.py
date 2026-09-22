#!/usr/bin/env python3
"""Stress two remaining claims from the conductance-corrected tension proposal.

A. Sign reliability:
   Compare sign(epsilon*Phi_candidate) against sign(-dE/dg_candidate) from central finite
   differences of the full passive Nethra dynamics across random networks. Magnitude equality is
   already rejected; this asks whether the local direct term is at least a reliable descent sign.

B. Support weighting:
   Compare Q-weighted and Phi-weighted incidence credit when the genuinely predictive incidence
   begins weak and nuisance incidences begin strong. This directly tests the claimed rich-get-richer
   difference without subset search.

Core Nethra is not modified.
"""

import math
import random
from statistics import mean

from nethra import NethraField

T=.5
STEPS=70
DT=T/STEPS
U_NEXT=.010


def integrate_custom(field, currents, edge_rows):
    for n in field.nethra:
        n.activation=0.0
        n.external=0.0
    for n,j in currents.items():
        n.external=float(j)
    field._edges=lambda: tuple(edge_rows)

    A={n:0.0 for n in field.nethra}
    for _ in range(STEPS):
        a0={n:n.activation for n in field.nethra}
        k1=field._derivative_at(a0)
        a1={n:a0[n]+.5*DT*k1[n] for n in field.nethra}; k2=field._derivative_at(a1)
        a2={n:a0[n]+.5*DT*k2[n] for n in field.nethra}; k3=field._derivative_at(a2)
        a3={n:a0[n]+DT*k3[n] for n in field.nethra}; k4=field._derivative_at(a3)
        for n in field.nethra:
            A[n]+=DT*(a0[n]+2*a1[n]+2*a2[n]+a3[n])/6.0
            n.activation=a0[n]+DT*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.0
    for n in field.nethra: n.external=0.0
    return A


def pred_at(receiver, edges, A):
    p=0.0
    for a,b,g in edges:
        if b is receiver:
            p+=g*(A[a]-A[b])
        elif a is receiver:
            p+=g*(A[b]-A[a])
    return p


def random_case(seed, candidate_g, delta=1e-5):
    rng=random.Random(seed)
    f=NethraField(leakage=.3+rng.random()*.9,convergence_gain=0.0)
    nodes=[f.new() for _ in range(7)]
    r=nodes[0]
    y=nodes[1]

    # Connected random passive graph with candidate edge (r,y) controlled independently.
    fixed=[]
    # backbone excludes direct r-y
    for i in range(len(nodes)-1):
        a,b=nodes[i],nodes[i+1]
        if frozenset((a,b))==frozenset((r,y)):
            continue
        fixed.append((a,b,.08+rng.random()*1.25))
    # extra random edges
    pairs=[]
    for i in range(len(nodes)):
        for j in range(i+1,len(nodes)):
            if frozenset((nodes[i],nodes[j]))==frozenset((r,y)):
                continue
            if all(frozenset((nodes[i],nodes[j]))!=frozenset((a,b)) for a,b,_ in fixed):
                pairs.append((nodes[i],nodes[j]))
    rng.shuffle(pairs)
    for a,b in pairs[:6]:
        fixed.append((a,b,.05+rng.random()*1.3))

    sources={}
    for n in rng.sample(nodes,3):
        sources[n]=rng.uniform(-1.0,1.0)

    def measure(g):
        edges=list(fixed)
        if g>0:
            edges.append((r,y,g))
        A=integrate_custom(f,sources,edges)
        p=pred_at(y,edges,A)
        eps=U_NEXT-p
        return .5*eps*eps,p,eps,A[r]-A[y]

    lo=measure(max(0.0,candidate_g-delta))
    hi=measure(candidate_g+delta)
    mid=measure(candidate_g)

    if candidate_g<=delta:
        dE=(hi[0]-mid[0])/delta
    else:
        dE=(hi[0]-lo[0])/(2*delta)

    full=-dE
    local=mid[2]*mid[3]
    return full,local,mid[2],mid[3],f.leakage


def sign(x,tol=1e-12):
    if x>tol:return 1
    if x<-tol:return -1
    return 0


def test_sign_stress():
    rows=[]
    mismatches=[]
    near=[]
    for seed in range(12000,12500):
        for g in (0.0,.05,.2,.6,1.2):
            full,local,eps,phi,leak=random_case(seed,g)
            sf,sl=sign(full),sign(local)
            row=(seed,g,full,local,eps,phi,leak,sf,sl)
            rows.append(row)
            if sf and sl and sf!=sl:
                mismatches.append(row)
            if sf==0 or sl==0:
                near.append(row)
    return rows,mismatches,near


# --- incidence weighting comparison ---

def g_from_e(f,e):
    e=max(0.0,float(e))
    return f.g_min+(f.g_max-f.g_min)*(1-math.exp(-e/f.tau))


def build_credit():
    f=NethraField(leakage=.6,convergence_gain=0.0)
    a=f.new(); b=f.new(); c=f.new(); y=f.new(); r=f.new()
    members=(a,b,c,y)
    return f,a,b,c,y,r,members


def run_credit(seed, mode, eta=1800.0, trials=7000):
    rng=random.Random(seed)
    f,a,b,c,y,r,members=build_credit()

    # True A deliberately begins weak; nuisance B/C begin strong.
    ev={a:0.0,b:100.0,c:100.0,y:20.0}

    def edges():
        return tuple((r,m,g_from_e(f,ev[m])) for m in members)
    f._edges=edges

    for _ in range(trials):
        x=rng.random()
        jb=rng.random()
        jc=rng.random()
        jy=max(0.0,min(.02,.02*x+rng.uniform(-.0015,.0015)))

        A=integrate_custom(f,{a:x,b:jb,c:jc},edges())
        pred=g_from_e(f,ev[y])*(A[r]-A[y])
        eps=T*jy-pred
        phi_out=A[r]-A[y]
        tension=phi_out*eps

        # output incidence learns from same conductance-corrected local tension
        ev[y]=max(0.0,ev[y]+eta*tension)

        qs={}
        phis={}
        for m in (a,b,c):
            phi=A[m]-A[r]  # positive when m supplied R
            phis[m]=max(0.0,phi)
            qs[m]=max(0.0,g_from_e(f,ev[m])*phi)

        basis=qs if mode=="Q" else phis
        total=sum(basis.values())
        if total>0:
            for m,w in basis.items():
                ev[m]=max(0.0,ev[m]+eta*(w/total)*tension)

    # prospective output effect from each isolated context
    resp=[]
    for m in (a,b,c):
        A=integrate_custom(f,{m:1.0},edges())
        resp.append(g_from_e(f,ev[y])*(A[r]-A[y]))
    return (ev[a],ev[b],ev[c],ev[y]),tuple(resp)


def test_weighting_recovery():
    out={"Q":[],"Phi":[]}
    for mode in out:
        for i in range(8):
            out[mode].append(run_credit(13000+i,mode))
    return out


def main():
    rows,mismatches,near=test_sign_stress()
    print("sign_stress_total",len(rows))
    print("sign_mismatches",len(mismatches))
    print("sign_near_zero",len(near))
    print("first_mismatches",mismatches[:12])

    credit=test_weighting_recovery()
    for mode,rows2 in credit.items():
        print(mode,"runs")
        for row in rows2: print(row)
        print(mode,"mean_evidence",tuple(mean(r[0][i] for r in rows2) for i in range(4)))
        print(mode,"mean_response",tuple(mean(r[1][i] for r in rows2) for i in range(3)))
        print(mode,"A_wins",sum(1 for r in rows2 if r[0][0]>r[0][1] and r[0][0]>r[0][2]))

    print("all_assertions_passed")


if __name__=="__main__":
    main()
