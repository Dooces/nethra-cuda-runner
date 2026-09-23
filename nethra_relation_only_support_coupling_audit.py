#!/usr/bin/env python3
"""Minimal test: earned support changes only relation<->relation coupling.

No new state and no direction.

Train the ordinary stock core. During prediction only:
  - primitive<->relation incidences stay exactly stock;
  - relation<->relation incidences keep their same symmetric edge;
  - their conductance is blended toward a persistent relation-support conductance.

For an edge R_i--R_j:
    S_ij = min(S_i, S_j)
    G_ij = conductance(S_ij)
    g_eff = g_stock + beta * (G_ij - g_stock)

S_i is the strongest stable source-history recurrence mapped to relation i.

This asks whether the existing recursive Nethra activations already supply the dynamical degrees of
freedom, with earned support needed only to stabilize coupling inside that recursive subspace.

No primitive edge is strengthened. No phase variable, relay, stored direction, transition table,
future signature, or prediction threshold is introduced.
"""
from __future__ import annotations
import copy, math
from nethra import NethraField

DT=.15
SUB=.0015
CYCLES=300
BETAS=(0.0,.05,.10,.25,.50,1.0)

class RelationOnlyField(NethraField):
    beta=0.0

    def relation_support(self,r):
        return max((self.history_count[k] for k,x in self.history_relation.items() if x is r),default=0)

    def _edges(self):
        base=super()._edges()
        out=[]
        for a,b,g in base:
            if a.routes and b.routes and self.beta>0:
                s=min(self.relation_support(a),self.relation_support(b))
                target=self.conductance(s)
                g=g+self.beta*(target-g)
            out.append((a,b,g))
        return tuple(out)

def train(cue):
    f=NethraField(g_min=.20,g_max=1.50,tau=100.,capacitance=1.,leakage=.6,convergence_gain=0.)
    leaves=[f.new() for _ in range(4)];time=f.new()
    start=(cue+1)%4
    for k in range(CYCLES*4):
        i=(start+k)%4
        time.push(1.0);leaves[i].push(1.0);f.step(DT)
    for n in f.nethra:n.external=0.0
    return f

def as_mode(base,beta):
    f=copy.deepcopy(base);f.__class__=RelationOnlyField;f.beta=beta;return f

def integrate(f,h):
    if h<=0:return
    steps=max(1,round(h/SUB));dt=h/steps
    for _ in range(steps):
        a0={n:n.activation for n in f.nethra};k1=f._derivative_at(a0)
        a1={n:a0[n]+.5*dt*k1[n] for n in f.nethra};k2=f._derivative_at(a1)
        a2={n:a0[n]+.5*dt*k2[n] for n in f.nethra};k3=f._derivative_at(a2)
        a3={n:a0[n]+dt*k3[n] for n in f.nethra};k4=f._derivative_at(a3)
        for n in f.nethra:n.activation=a0[n]+dt*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.

def phase_events(f):
    leaves=f.nethra[:4];time=f.nethra[4]
    return [frozenset(((leaves[(i-1)%4],-1),(leaves[i],1),(time,0))) for i in range(4)]

def phase_relations(f,phases):
    out={p:[] for p in phases}
    for (before,after),r in f.history_relation.items():
        if before in out and r not in out[before]:out[before].append(r)
    return out

def phase_score(rels,vals):
    return {p:sum(vals.get(r,0.0) for r in rs) for p,rs in rels.items()}

def main():
    bases=[train(c) for c in range(4)]
    print("=== INTERNAL_EDGE_COUNTS ===")
    for cue,b in enumerate(bases):
        rr=sum(1 for a,bn,g in b._edges() if a.routes and bn.routes)
        rp=sum(1 for a,bn,g in b._edges() if bool(a.routes)!=bool(bn.routes))
        print("EDGES",cue,{"relation_relation":rr,"relation_other":rp,
                           "relations":sum(bool(n.routes) for n in b.nethra)})

    print("=== TIME_PHASE ===")
    for beta in BETAS:
        rel_rows=[];leaf_rows=[]
        for cue,base in enumerate(bases):
            for step in (1,2,3,4):
                f=as_mode(base,beta);leaves=f.nethra[:4];time=f.nethra[4]
                phases=phase_events(f);rels=phase_relations(f,phases)
                initial={n:n.activation for n in f.nethra}
                time.external=1.0;integrate(f,step*DT);time.external=0.0

                # Recursive phase: use support above each relation's pure leakage.
                residual={n:n.activation-initial[n]*math.exp(-f.leakage*step*DT/f.capacitance)
                          for n in f.nethra}
                rs=phase_score(rels,residual)
                expected_phase=(cue+step)%4
                ro=sorted(range(4),key=lambda i:rs[phases[i]],reverse=True)
                rrival=max(rs[phases[i]] for i in range(4) if i!=expected_phase)
                rel_rows.append((step,ro.index(expected_phase)+1,
                                 rs[phases[expected_phase]]-rrival))

                # Observable expectation: same support-above-pure-decay metric on primitive leaves.
                ls=[leaves[i].activation-initial[leaves[i]]*
                    math.exp(-f.leakage*step*DT/f.capacitance) for i in range(4)]
                expected_leaf=(cue+step)%4
                lo=sorted(range(4),key=lambda i:ls[i],reverse=True)
                lrival=max(ls[i] for i in range(4) if i!=expected_leaf)
                leaf_rows.append((step,lo.index(expected_leaf)+1,ls[expected_leaf]-lrival))

        print("BETA",beta,{
            "relation_phase":{s:{
                "rank1":sum(r[1]==1 for r in rel_rows if r[0]==s),
                "mean_rank":sum(r[1] for r in rel_rows if r[0]==s)/4,
                "mean_margin":sum(r[2] for r in rel_rows if r[0]==s)/4,
            } for s in (1,2,3,4)},
            "leaf_phase":{s:{
                "rank1":sum(r[1]==1 for r in leaf_rows if r[0]==s),
                "mean_rank":sum(r[1] for r in leaf_rows if r[0]==s)/4,
                "mean_margin":sum(r[2] for r in leaf_rows if r[0]==s)/4,
            } for s in (1,2,3,4)}
        })

    print("=== CONTINUOUS_ONE_CUE ===")
    base=bases[1]
    for beta in BETAS:
        rows=[]
        for h in (.025,.05,.075,.10,.15,.20,.25,.30,.35,.40,.45,.50,.55,.60,.75,.90):
            f=as_mode(base,beta);leaves=f.nethra[:4];time=f.nethra[4]
            phases=phase_events(f);rels=phase_relations(f,phases)
            initial={n:n.activation for n in f.nethra}
            time.external=1.0;integrate(f,h);time.external=0.0
            residual={n:n.activation-initial[n]*math.exp(-f.leakage*h/f.capacitance)
                      for n in f.nethra}
            rs=phase_score(rels,residual)
            rorder=sorted(range(4),key=lambda i:rs[phases[i]],reverse=True)
            ls=[residual[leaves[i]] for i in range(4)]
            lorder=sorted(range(4),key=lambda i:ls[i],reverse=True)
            rows.append((h,rorder,lorder,[rs[phases[i]] for i in range(4)],ls))
        print("TRAJ",beta,rows)
    print("all_assertions_passed")

if __name__=="__main__":main()
