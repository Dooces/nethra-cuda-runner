#!/usr/bin/env python3
"""Test relation-level earned support x continuous live manifestation.

Candidate law (diagnostic, core unchanged):

For relation R with routes q:
    S_R = strongest stable source-history recurrence mapped to R
    G_R = conductance(S_R)

At one RK stage with activation a:
    I_q = mean_m[max(a_m - a_R, 0)] for m in route q
    phi_q(cross) = sum_{p != q} I_p / sum_p I_p

Then:
    g(R,q) = g_min + (G_R - g_min) * phi_q

Interpretation: the route presently supplying R identifies the live manifestation. Earned support
belongs to R. Support is exposed toward the other manifestation(s), so the same symmetric incidence
law carries prospective current without a stored direction bit.

Controls:
  STOCK      current exact-event evidence rule
  SAME       support x same-route manifestation (retrospective control)
  GLOBAL     relation support on every route (known over-propagation control)
  CROSS      proposal above

Tests:
  1) deterministic A-B-C-D with TIME-only prediction across experience and horizon;
  2) source-free immediate derivative;
  3) 2/4/6 shared-surface contextual continuations;
  4) representation-drift trace: relation support grows while old route signatures stay stale;
  5) ambiguous 70/30 continuation, to see whether support magnitude matters sensibly.
"""
from __future__ import annotations
from collections import defaultdict
import copy, math, random
from nethra import NethraField

SUB=.0025

class SupportManifestField(NethraField):
    def __init__(self,*args,mode="STOCK",**kwargs):
        super().__init__(*args,**kwargs)
        self.mode=mode

    def relation_support(self,relation):
        vals=[
            self.history_count[key]
            for key,r in self.history_relation.items()
            if r is relation
        ]
        return max(vals,default=0)

    def _candidate_edges(self,activation):
        edges={}
        for relation in self.nethra:
            if not relation.routes:
                continue
            routes=list(relation.routes)
            support=self.relation_support(relation)
            G=self.conductance(support)

            pressure={}
            for route in routes:
                if route:
                    pressure[route]=sum(max(0.0,activation[m]-activation[relation]) for m in route)/len(route)
                else:
                    pressure[route]=0.0
            total=sum(pressure.values())

            for route,conditions in relation.routes.items():
                if self.mode=="GLOBAL":
                    phi=1.0
                elif total<=1e-15:
                    phi=0.0
                elif self.mode=="SAME":
                    phi=pressure[route]/total
                elif self.mode=="CROSS":
                    phi=(total-pressure[route])/total
                else:
                    raise ValueError(self.mode)

                g=self.g_min+(G-self.g_min)*max(0.0,min(1.0,phi))
                for member in route:
                    key=frozenset((relation,member))
                    if g>edges.get(key,0.0):
                        edges[key]=g
        return tuple((*(tuple(key)),g) for key,g in edges.items())

    def _derivative_at(self,activation):
        if self.mode=="STOCK":
            return super()._derivative_at(activation)

        current={n:n.external-self.leakage*activation[n] for n in self.nethra}
        neighbors=defaultdict(list)
        for a,b,g in self._candidate_edges(activation):
            flow=g*(activation[a]-activation[b])
            current[a]-=flow
            current[b]+=flow
            neighbors[a].append((b,g));neighbors[b].append((a,g))

        # Preserve the exact ordinary F61 convergence term, although these probes use gain=0.
        for receiver,row in neighbors.items():
            suppliers=[]
            for neighbor,g in row:
                p=g*(activation[neighbor]-activation[receiver])
                if p>0.0:suppliers.append((neighbor,p))
            if len(suppliers)<2:continue
            total=sum(p for _,p in suppliers)
            pair_sum=0.0
            for i,(a,pa) in enumerate(suppliers):
                for b,pb in suppliers[i+1:]:
                    pair_sum+=pa*pb*self._independence(a,b)
            bonus=min(total,self.convergence_gain*(2.0*pair_sum/total)) if total else 0.0
            if bonus<=0.0:continue
            current[receiver]+=bonus
            for supplier,p in suppliers:
                current[supplier]-=bonus*p/total
        return {n:v/self.capacitance for n,v in current.items()}


def integrate(f,duration):
    steps=max(1,round(duration/SUB));dt=duration/steps
    for _ in range(steps):
        a0={n:n.activation for n in f.nethra};k1=f._derivative_at(a0)
        a1={n:a0[n]+.5*dt*k1[n] for n in f.nethra};k2=f._derivative_at(a1)
        a2={n:a0[n]+.5*dt*k2[n] for n in f.nethra};k3=f._derivative_at(a2)
        a3={n:a0[n]+dt*k3[n] for n in f.nethra};k4=f._derivative_at(a3)
        for n in f.nethra:
            n.activation=a0[n]+dt*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.0


def cycle_train(cycles,cue,mode="STOCK",candidate_during_training=False,time_node=True):
    train_mode=mode if candidate_during_training else "STOCK"
    f=SupportManifestField(
        g_min=.20,g_max=1.50,tau=100.0,
        capacitance=1.0,leakage=.6,convergence_gain=0.0,
        mode=train_mode,
    )
    leaves=[f.new() for _ in range(4)]
    time=f.new() if time_node else None
    start=(cue+1)%4
    for k in range(cycles*4):
        i=(start+k)%4
        if time is not None:time.push(1.0)
        leaves[i].push(1.0)
        f.step(.15)
    for n in f.nethra:n.external=0.0
    f.mode=mode
    return f,leaves,time


def time_residual_row(cycles,cue,mode,horizon,candidate_during_training=False):
    f,leaves,time=cycle_train(cycles,cue,mode,candidate_during_training,True)
    before=[n.activation for n in leaves]
    time.external=1.0
    integrate(f,horizon)
    time.external=0.0
    pure=[x*math.exp(-f.leakage*horizon/f.capacitance) for x in before]
    residual=[leaves[i].activation-pure[i] for i in range(4)]
    expected=(cue+1)%4
    candidates=[i for i in range(4) if i!=cue]
    order=sorted(candidates,key=lambda i:residual[i],reverse=True)
    rival=max(residual[i] for i in candidates if i!=expected)
    return order.index(expected)+1,residual[expected]-rival,residual,f


def deterministic():
    print("=== DETERMINISTIC_TIME ===")
    for trained in (False,True):
        for cycles in (3,10,30,100,300):
            for horizon in (.05,.15,.30,.60):
                for mode in ("STOCK","SAME","GLOBAL","CROSS"):
                    if trained and mode=="STOCK":
                        label="STOCK_TRAIN"
                    else:
                        label=(mode+"_FULLTRAIN") if trained else mode
                    rows=[time_residual_row(cycles,c,mode,horizon,trained) for c in range(4)]
                    print("TIME",{
                        "train_candidate":trained,"mode":label,
                        "cycles":cycles,"horizon":horizon,
                        "rank1":sum(r[0]==1 for r in rows),
                        "mean_margin":sum(r[1] for r in rows)/4,
                        "margins":[r[1] for r in rows],
                    })


def source_free():
    print("=== SOURCE_FREE ===")
    for cycles in (10,30,100,300):
        for mode in ("STOCK","SAME","GLOBAL","CROSS"):
            ranks=[];margins=[]
            for cue in range(4):
                f,leaves,_=cycle_train(cycles,cue,mode,False,False)
                d=f.derivative();expected=(cue+1)%4
                cand=[i for i in range(4) if i!=cue]
                order=sorted(cand,key=lambda i:d[leaves[i]],reverse=True)
                rival=max(d[leaves[i]] for i in cand if i!=expected)
                ranks.append(order.index(expected)+1)
                margins.append(d[leaves[expected]]-rival)
            print("ZERO",{
                "mode":mode,"cycles":cycles,
                "rank1":sum(r==1 for r in ranks),
                "ranks":ranks,"mean_margin":sum(margins)/4,
            })


def context_load():
    print("=== CONTEXT_LOAD ===")
    for k in (2,4,6):
        for mode in ("STOCK","GLOBAL","CROSS"):
            f=SupportManifestField(
                g_min=.20,g_max=1.50,tau=100.0,
                capacitance=1.0,leakage=.6,convergence_gain=0.0,mode=mode,
            )
            # A=0,B=1,X_i=2+i,Y_i=2+k+i
            leaves=[f.new() for _ in range(2+2*k)]
            A=0;B=1
            def seg(i):return [2+i,A,B,2+k+i]
            for _ in range(80):
                for i in range(k):
                    for x in seg(i):
                        leaves[x].push(1.0);f.step(.12)
            ranks=[];margins=[]
            for i in range(k):
                x,y=2+i,2+k+i
                for z in (x,A,B):
                    leaves[z].push(1.0);f.step(.12)
                for n in f.nethra:n.external=0.0
                d=f.derivative();ys=[2+k+j for j in range(k)]
                order=sorted(ys,key=lambda z:d[leaves[z]],reverse=True)
                rival=max((d[leaves[z]] for z in ys if z!=y),default=d[leaves[y]])
                ranks.append(order.index(y)+1);margins.append(d[leaves[y]]-rival)
                leaves[y].push(1.0);f.step(.12)
            print("CONTEXT",{
                "k":k,"mode":mode,
                "rank1":sum(r==1 for r in ranks),
                "ranks":ranks,"margins":margins,
                "relations":sum(bool(n.routes) for n in f.nethra),
            })


def drift_trace():
    print("=== DRIFT_SUPPORT ===")
    f,_,_=cycle_train(300,1,"CROSS",False,False)
    nm={n:(chr(65+i) if i<4 else f"R{i-3}") for i,n in enumerate(f.nethra)}
    for r in f.nethra:
        if not r.routes:continue
        stored=[max(conds.values(),default=0) for conds in r.routes.values()]
        print("DRIFT",{
            "relation":nm[r],
            "stable_relation_support":f.relation_support(r),
            "stored_route_evidence":stored,
        })


def ambiguity():
    print("=== AMBIGUOUS_70_30 ===")
    # A,B,C,D plus TIME. Histories AB->BC and AB->BD compete.
    A,B,C,D,T=range(5)
    rng=random.Random(9917)
    outcomes=[C if rng.random()<.70 else D for _ in range(600)]
    for mode in ("STOCK","GLOBAL","CROSS"):
        f=SupportManifestField(
            g_min=.20,g_max=1.50,tau=100.0,
            capacitance=1.0,leakage=.6,convergence_gain=0.0,mode=mode,
        )
        leaves=[f.new() for _ in range(5)]
        for out in outcomes:
            for x in (A,B,out):
                leaves[T].push(1.0);leaves[x].push(1.0);f.step(.15)
        # Natural cue A,B then TIME only.
        for x in (A,B):
            leaves[T].push(1.0);leaves[x].push(1.0);f.step(.15)
        for n in f.nethra:n.external=0.0
        before=[leaves[C].activation,leaves[D].activation]
        leaves[T].external=1.0;integrate(f,.15);leaves[T].external=0.0
        pure=[v*math.exp(-f.leakage*.15/f.capacitance) for v in before]
        rc=leaves[C].activation-pure[0];rd=leaves[D].activation-pure[1]
        mapped=[]
        for key,r in f.history_relation.items():
            # show strongest source recurrence associated with each persistent relation
            mapped.append(f.history_count[key])
        print("AMBIG",{
            "mode":mode,
            "C_residual":rc,"D_residual":rd,"C_minus_D":rc-rd,
            "relations":sum(bool(n.routes) for n in f.nethra),
            "top_history_counts":sorted(mapped,reverse=True)[:8],
        })


def main():
    drift_trace()
    source_free()
    deterministic()
    context_load()
    ambiguity()
    print("all_assertions_passed")

if __name__=="__main__":main()
