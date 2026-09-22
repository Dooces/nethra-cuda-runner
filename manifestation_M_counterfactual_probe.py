#!/usr/bin/env python3
"""Vet M_i = C*Delta a_i + lambda*A_i as a universal completed-interval manifestation.

Core identity under the actual Nethra field:
    C*Delta a_i = U_i - lambda*A_i + Q_internal_i + Q_F61_i
therefore
    M_i = C*Delta a_i + lambda*A_i
        = U_i + P_i
where P_i is the exact integrated internal field drive (conductive + F61).

This probe tests:
1. identity closure against direct integration of conductive + F61 currents;
2. stationary manifestation: Delta~0 while A and M remain nonzero;
3. recursive internally manifested relation: U=0 but M>0;
4. counterfactual local observability using
       epsilon_i(k+1) = M_i(k+1) - P_i(k)
       L_tau = sum Phi_ij(k)*epsilon_j(k+tau)
   against the true replay sensitivity
       G = -d/dg 1/2 sum epsilon^2
   over topology, delay, depth, and F61 convergence;
5. held-seed and structural holdout temporal-kernel diagnostics.

No plasticity rule is installed.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json, math, os, random
from statistics import median

from nethra import NethraField

INTERVALS=44
SUBSTEPS=16
DT_INTERVAL=.20
DT=DT_INTERVAL/SUBSTEPS
HMAX=8
FD_H=1e-4
DEPTHS=(1,3,6)
DELAYS=(1,3)
TOPOLOGIES=("chain","loop","skip")
CONVERGENCE=(0.0,1.0)
SEEDS=(0,1)


@dataclass
class Spec:
    depth:int
    delay:int
    topology:str
    convergence:float
    seed:int


def topology_spec(depth,kind,seed):
    rng=random.Random(81000+seed*101+depth*13+sum(map(ord,kind)))
    n=depth+3
    edges=[]
    for i in range(n-1):
        edges.append([i,i+1,.30+.68*rng.random()])
    if kind=="loop":
        edges.append([n-1,0,.28+.57*rng.random()])
    elif kind=="skip":
        for i in range(n-2):
            edges.append([i,i+2,.20+.45*rng.random()])
    return n,edges


def input_sequence(n,delay,seed):
    rng=random.Random(82000+seed*137+delay*19)
    latent=[rng.uniform(-1,1) for _ in range(INTERVALS+delay+HMAX+8)]
    noise=[[rng.uniform(-1,1) for _ in range(INTERVALS)] for _ in range(n)]
    seq=[]
    for k in range(INTERVALS):
        row=[0.0]*n
        row[0]=.85*latent[k+delay]
        # A second physical source carries the delayed latent relation through another location.
        j=n//2
        if j==0:j=1
        row[j]=.34*latent[k]+.05*noise[j][k]
        # sparse nuisance source at far boundary
        if n>3 and k%3==0:
            row[-1]=.12*noise[-1][k]
        seq.append(row)
    return seq


def build_field(n,edges,conv):
    f=NethraField(leakage=.55,capacitance=1.0,convergence_gain=conv)
    nodes=[f.new() for _ in range(n)]
    f._edges=lambda: tuple((nodes[i],nodes[j],g) for i,j,g in edges)
    if conv>0:
        for i in range(n):
            for j in range(i+1,n):
                f.pair_stats[frozenset((nodes[i],nodes[j]))]=(0.0,1.0,1.0,20)
    return f,nodes


def conductive_current(field,state,nodes):
    cur={n:0.0 for n in nodes}
    for a,b,g in field._edges():
        q=g*(state[a]-state[b])
        cur[a]-=q
        cur[b]+=q
    return cur


def internal_components(field,state,nodes):
    """Return conductive current and exact F61 convergence current at one state."""
    cond=conductive_current(field,state,nodes)
    full={n:field.capacitance*v for n,v in field._derivative_at(state).items()}
    conv={}
    for n in nodes:
        # C da/dt = external - leak*a + conductive + convergence
        conv[n]=full[n]-n.external+field.leakage*state[n]-cond[n]
    return cond,conv


def integrate_interval(field,nodes,currents):
    for node,j in zip(nodes,currents):
        node.external=float(j)

    a_start={n:n.activation for n in nodes}
    A={n:0.0 for n in nodes}
    QC={n:0.0 for n in nodes}
    QB={n:0.0 for n in nodes}

    for _ in range(SUBSTEPS):
        a0={n:n.activation for n in nodes}
        k1=field._derivative_at(a0)
        a1={n:a0[n]+.5*DT*k1[n] for n in nodes}; k2=field._derivative_at(a1)
        a2={n:a0[n]+.5*DT*k2[n] for n in nodes}; k3=field._derivative_at(a2)
        a3={n:a0[n]+DT*k3[n] for n in nodes}; k4=field._derivative_at(a3)

        comps=[internal_components(field,s,nodes) for s in (a0,a1,a2,a3)]
        for n in nodes:
            A[n]+=DT*(a0[n]+2*a1[n]+2*a2[n]+a3[n])/6
            QC[n]+=DT*(comps[0][0][n]+2*comps[1][0][n]+2*comps[2][0][n]+comps[3][0][n])/6
            QB[n]+=DT*(comps[0][1][n]+2*comps[1][1][n]+2*comps[2][1][n]+comps[3][1][n])/6
            n.activation=a0[n]+DT*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6

    D={n:n.activation-a_start[n] for n in nodes}
    U={n:DT_INTERVAL*float(j) for n,j in zip(nodes,currents)}
    M={n:field.capacitance*D[n]+field.leakage*A[n] for n in nodes}
    P={n:QC[n]+QB[n] for n in nodes}

    for n in nodes:n.external=0.0
    return A,D,U,M,P,QC,QB


def replay(spec,edge_override=None):
    n,edges=topology_spec(spec.depth,spec.topology,spec.seed)
    if edge_override:
        idx,newg=edge_override
        edges=[r[:] for r in edges]
        edges[idx][2]=newg

    f,nodes=build_field(n,edges,spec.convergence)
    seq=input_sequence(n,spec.delay,spec.seed)
    As=[]; Ds=[]; Us=[]; Ms=[]; Ps=[]; QC=[]; QB=[]
    max_identity=0.0

    for cur in seq:
        A,D,U,M,P,qc,qb=integrate_interval(f,nodes,cur)
        As.append([A[n] for n in nodes])
        Ds.append([D[n] for n in nodes])
        Us.append([U[n] for n in nodes])
        Ms.append([M[n] for n in nodes])
        Ps.append([P[n] for n in nodes])
        QC.append([qc[n] for n in nodes])
        QB.append([qb[n] for n in nodes])
        for n0 in nodes:
            max_identity=max(max_identity,abs(M[n0]-(U[n0]+P[n0])))

    eps=[[0.0]*INTERVALS for _ in range(n)]
    loss=[0.0]*n
    for j in range(n):
        for t in range(1,INTERVALS):
            e=Ms[t][j]-Ps[t-1][j]
            eps[j][t]=e
            loss[j]+=.5*e*e

    return {
        "n":n,"edges":edges,"A":As,"D":Ds,"U":Us,"M":Ms,"P":Ps,
        "QC":QC,"QB":QB,"eps":eps,"loss":loss,"max_identity":max_identity,
    }


def lag_features(base,ei,src,dst):
    i,j,g=base["edges"][ei]
    signphi=1 if (i==src and j==dst) else -1
    phi=[signphi*(base["A"][k][i]-base["A"][k][j]) for k in range(INTERVALS)]
    eps=base["eps"][dst]
    out=[]
    for tau in range(1,HMAX+1):
        out.append(sum(phi[k]*eps[k+tau] for k in range(INTERVALS-tau)))
    return out


def scenario(spec):
    base=replay(spec)
    samples=[]
    for ei,(i,j,g) in enumerate(base["edges"]):
        plus=replay(spec,(ei,g+FD_H))
        minus=replay(spec,(ei,g-FD_H))
        for src,dst in ((i,j),(j,i)):
            G=-(plus["loss"][dst]-minus["loss"][dst])/(2*FD_H)
            samples.append({
                "edge":ei,"src":src,"dst":dst,"g":g,"G":G,
                "L":lag_features(base,ei,src,dst),
            })
    return {"spec":spec.__dict__,"identity":base["max_identity"],"samples":samples}


def sign(x,tol=1e-11):
    return 1 if x>tol else -1 if x<-tol else 0


def corr(xs,ys):
    if len(xs)<2:return float("nan")
    mx=sum(xs)/len(xs); my=sum(ys)/len(ys)
    xx=sum((x-mx)**2 for x in xs); yy=sum((y-my)**2 for y in ys)
    if xx<=0 or yy<=0:return float("nan")
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/math.sqrt(xx*yy)


def metrics(rows,predict):
    gs=[]; ps=[]; re=[]; sa=[]
    for r in rows:
        g=r["G"]; p=predict(r)
        gs.append(g); ps.append(p)
        if abs(g)>1e-10:re.append(abs(p-g)/abs(g))
        sg,sp=sign(g),sign(p)
        if sg and sp:sa.append(sg==sp)
    return {
        "n":len(rows),"corr":corr(ps,gs),
        "sign_agree":sum(sa)/len(sa) if sa else None,
        "median_rel_error":median(re) if re else None,
    }


def dot(a,b):return sum(x*y for x,y in zip(a,b))


def solve(X,y,lam=1e-8):
    p=len(X[0]); A=[[0.0]*p for _ in range(p)]; b=[0.0]*p
    for row,t in zip(X,y):
        for i in range(p):
            b[i]+=row[i]*t
            for j in range(p):A[i][j]+=row[i]*row[j]
    scale=sum(A[i][i] for i in range(p))/max(1,p)
    for i in range(p):A[i][i]+=lam*max(scale,1e-12)
    aug=[A[i]+[b[i]] for i in range(p)]
    for c in range(p):
        piv=max(range(c,p),key=lambda r:abs(aug[r][c]))
        aug[c],aug[piv]=aug[piv],aug[c]
        d=aug[c][c]
        if abs(d)<1e-20:continue
        aug[c]=[v/d for v in aug[c]]
        for r in range(p):
            if r==c:continue
            q=aug[r][c]
            if q:aug[r]=[aug[r][k]-q*aug[c][k] for k in range(p+1)]
    return [aug[i][-1] for i in range(p)]


def stationary_control():
    """Steady external hold reaches near-stationary delta while A and M remain positive."""
    f=NethraField(leakage=.55,capacitance=1.0,convergence_gain=0.0)
    n=f.new()
    # Long warmup under constant source.
    for _ in range(400):
        integrate_interval(f,[n],[1.0])
    A,D,U,M,P,_,_=integrate_interval(f,[n],[1.0])
    return {"delta":D[n],"A":A[n],"M":M[n],"U":U[n],"P":P[n]}


def recursive_control():
    f=NethraField(leakage=.55,capacitance=1.0,convergence_gain=0.0)
    q1,q2,y=[f.new() for _ in range(3)]
    f._edges=lambda: ((y,q1,.9),(y,q2,.9))
    # reset protocol: one interval with internally manifested y and no direct U_y
    A,D,U,M,P,_,_=integrate_interval(f,[q1,q2,y],[1.0,1.0,0.0])
    return {"Uy":U[y],"Dy":D[y],"Ay":A[y],"My":M[y],"Py":P[y],"identity":M[y]-(U[y]+P[y])}


def main():
    stat=stationary_control()
    rec=recursive_control()
    print("stationary_control",stat)
    print("recursive_control",rec)
    assert abs(stat["delta"]) < 1e-6
    assert abs(stat["A"]) > .1
    assert abs(stat["M"]) > .01
    assert rec["Uy"]==0.0 and rec["My"]>0.0
    assert abs(rec["identity"])<1e-12

    specs=[Spec(d,l,t,c,s) for d in DEPTHS for l in DELAYS for t in TOPOLOGIES for c in CONVERGENCE for s in SEEDS]
    results=[]
    with ProcessPoolExecutor(max_workers=min(16,len(specs),os.cpu_count() or 1)) as pool:
        futs=[pool.submit(scenario,s) for s in specs]
        for f in as_completed(futs):results.append(f.result())

    maxid=max(r["identity"] for r in results)
    rows=[]
    for result in results:
        for sample in result["samples"]:rows.append({**result["spec"],**sample})
    print("max_identity_error",maxid)
    assert maxid<2e-12

    print("L1",metrics(rows,lambda r:r["L"][0]))

    # held-seed horizon sweep
    train=[r for r in rows if r["seed"]==0]
    test=[r for r in rows if r["seed"]==1]
    for h in range(1,HMAX+1):
        K=solve([r["L"][:h] for r in train],[r["G"] for r in train])
        print("held_seed_H",h,"K",K,"test",metrics(test,lambda r,K=K,h=h:dot(r["L"][:h],K)))

    # structural H4 holdout
    for field,value in (("topology","loop"),("topology","skip"),("depth",6),("delay",3),("convergence",1.0)):
        tr=[r for r in rows if r[field]!=value]
        te=[r for r in rows if r[field]==value]
        K=solve([r["L"][:4] for r in tr],[r["G"] for r in tr])
        print("holdout",field,value,"K4",K,"test",metrics(te,lambda r,K=K:dot(r["L"][:4],K)))

    with open("manifestation_M_counterfactual_results.json","w") as fh:
        json.dump({"stationary":stat,"recursive":rec,"max_identity":maxid,"rows":rows},fh)
    print("all_assertions_passed")


if __name__=="__main__":
    main()
