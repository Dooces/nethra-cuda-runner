#!/usr/bin/env python3
"""Test differential manifestation: relation support x live common-mode-cancelled phase.

For each relation R:
  common = intersection(all routes)
  differential route q = q - common
  V_q = mean activation of differential members
  mu = mean_q V_q
  M_q = max(0, V_q - mu)
  w_q = M_q / sum M

w identifies the currently dominant manifestation from the live field after common-mode cancellation.
Prospective exposure of route q is:
  phi_q = 1 - w_q

and:
  g_q = g_min + (conductance(S_R)-g_min)*phi_q

For a two-route relation, the field-dominant manifestation remains baseline-coupled while earned
support is exposed toward the other manifestation. As the other side rises, the differential can
flip without storing a direction bit or an event gate.

Core unchanged; diagnostic subclass only.
"""
from __future__ import annotations
from collections import defaultdict
import copy,math
from nethra import NethraField

DT=.15
SUB=.0015

class DifferentialField(NethraField):
    def relation_support(self,r):
        return max((self.history_count[k] for k,x in self.history_relation.items() if x is r),default=0)

    def _diff_edges(self,a):
        edges={}
        for r in self.nethra:
            if not r.routes:continue
            routes=list(r.routes)
            common=set(routes[0])
            for q in routes[1:]:common.intersection_update(q)

            V={}
            for q in routes:
                diff=[m for m in q if m not in common]
                members=diff if diff else list(q)
                V[q]=sum(a[m] for m in members)/len(members) if members else a[r]
            mu=sum(V.values())/len(V)
            M={q:max(0.0,V[q]-mu) for q in routes}
            total=sum(M.values())
            G=self.conductance(self.relation_support(r))
            for q,conds in r.routes.items():
                w=M[q]/total if total>1e-15 else 0.0
                phi=1.0-w
                g=self.g_min+(G-self.g_min)*phi
                for m in q:
                    key=frozenset((r,m))
                    if g>edges.get(key,0.0):edges[key]=g
        return tuple((*(tuple(k)),g) for k,g in edges.items())

    def _derivative_at(self,a):
        current={n:n.external-self.leakage*a[n] for n in self.nethra}
        for x,y,g in self._diff_edges(a):
            flow=g*(a[x]-a[y]);current[x]-=flow;current[y]+=flow
        return {n:v/self.capacitance for n,v in current.items()}


def integrate(f,duration):
    steps=max(1,round(duration/SUB));dt=duration/steps
    for _ in range(steps):
        a0={n:n.activation for n in f.nethra};k1=f._derivative_at(a0)
        a1={n:a0[n]+.5*dt*k1[n] for n in f.nethra};k2=f._derivative_at(a1)
        a2={n:a0[n]+.5*dt*k2[n] for n in f.nethra};k3=f._derivative_at(a2)
        a3={n:a0[n]+dt*k3[n] for n in f.nethra};k4=f._derivative_at(a3)
        for n in f.nethra:n.activation=a0[n]+dt*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.


def train_cycle(cycles,cue,fulltrain=False):
    if fulltrain:
        f=DifferentialField(g_min=.20,g_max=1.50,tau=100.,capacitance=1.,leakage=.6,convergence_gain=0.)
    else:
        # train stock, then copy persistent/transient state into DifferentialField
        s=NethraField(g_min=.20,g_max=1.50,tau=100.,capacitance=1.,leakage=.6,convergence_gain=0.)
        leaves=[s.new() for _ in range(4)];time=s.new()
        start=(cue+1)%4
        for k in range(cycles*4):
            i=(start+k)%4;time.push(1.);leaves[i].push(1.);s.step(DT)
        for n in s.nethra:n.external=0.
        f=copy.deepcopy(s)
        f.__class__=DifferentialField
        return f

    leaves=[f.new() for _ in range(4)];time=f.new();start=(cue+1)%4
    for k in range(cycles*4):
        i=(start+k)%4;time.push(1.);leaves[i].push(1.);f.step(DT)
    for n in f.nethra:n.external=0.
    return f


def residual_at(base,h):
    f=copy.deepcopy(base);leaves=f.nethra[:4];time=f.nethra[4]
    before=[n.activation for n in leaves]
    time.external=1.;integrate(f,h);time.external=0.
    pure=[v*math.exp(-f.leakage*h/f.capacitance) for v in before]
    return [leaves[i].activation-pure[i] for i in range(4)]


def phase_test():
    print("=== PHASE ===")
    for full in (False,True):
        for cycles in (30,100,300):
            rows=[]
            for cue in range(4):
                base=train_cycle(cycles,cue,full)
                for steps in (1,2,3,4):
                    h=steps*DT;res=residual_at(base,h);expected=(cue+steps)%4
                    order=sorted(range(4),key=lambda i:res[i],reverse=True)
                    rival=max(res[i] for i in range(4) if i!=expected)
                    rows.append((steps,order.index(expected)+1,res[expected]-rival))
            print("PHASE",{
                "fulltrain":full,"cycles":cycles,
                "by_step":{s:{
                    "rank1":sum(x[1]==1 for x in rows if x[0]==s),
                    "mean_rank":sum(x[1] for x in rows if x[0]==s)/4,
                    "mean_margin":sum(x[2] for x in rows if x[0]==s)/4,
                } for s in (1,2,3,4)}
            })


def trajectory():
    print("=== TRAJECTORY ===")
    for cycles in (30,100,300):
        base=train_cycle(cycles,1,False)
        rows=[]
        for h in (.025,.05,.075,.10,.15,.20,.25,.30,.35,.40,.45,.50,.55,.60,.75,.90):
            res=residual_at(base,h)
            order=sorted(range(4),key=lambda i:res[i],reverse=True)
            rows.append((h,order,res))
        print("TRAJ",cycles,rows)


def context_test():
    print("=== CONTEXT ===")
    for k in (2,4,6):
        f=DifferentialField(g_min=.20,g_max=1.50,tau=100.,capacitance=1.,leakage=.6,convergence_gain=0.)
        leaves=[f.new() for _ in range(2+2*k)];A=0;B=1
        def seg(i):return [2+i,A,B,2+k+i]
        for _ in range(80):
            for i in range(k):
                for x in seg(i):leaves[x].push(1.);f.step(.12)
        ranks=[];margins=[]
        for i in range(k):
            x,y=2+i,2+k+i
            for z in (x,A,B):leaves[z].push(1.);f.step(.12)
            for n in f.nethra:n.external=0.
            d=f.derivative();ys=[2+k+j for j in range(k)]
            order=sorted(ys,key=lambda z:d[leaves[z]],reverse=True)
            rival=max((d[leaves[z]] for z in ys if z!=y),default=d[leaves[y]])
            ranks.append(order.index(y)+1);margins.append(d[leaves[y]]-rival)
            leaves[y].push(1.);f.step(.12)
        print("CONTEXT",{"k":k,"rank1":sum(x==1 for x in ranks),"ranks":ranks,"margins":margins})


def phase_weights():
    print("=== WEIGHTS ===")
    f=train_cycle(300,1,False);a={n:n.activation for n in f.nethra}
    nm={n:(chr(65+i) if i<4 else ("TIME" if i==4 else f"R{i-4}")) for i,n in enumerate(f.nethra)}
    for r in f.nethra:
        if not r.routes:continue
        routes=list(r.routes);common=set(routes[0])
        for q in routes[1:]:common.intersection_update(q)
        vals=[]
        for q in routes:
            diff=[m for m in q if m not in common];members=diff if diff else list(q)
            vals.append((q,sum(a[m] for m in members)/len(members)))
        mu=sum(v for _,v in vals)/len(vals);M=[max(0,v-mu) for _,v in vals];tot=sum(M)
        print("WEIGHT",{
            "relation":nm[r],"support":f.relation_support(r),
            "common":tuple(sorted(nm[m] for m in common)),
            "routes":[{
                "diff":tuple(sorted(nm[m] for m in q if m not in common)),
                "V":v,"current_w":M[i]/tot if tot>1e-15 else 0.,
                "prospective_phi":1-(M[i]/tot if tot>1e-15 else 0.),
            } for i,(q,v) in enumerate(vals)]
        })


def main():
    phase_weights();phase_test();trajectory();context_test();print("all_assertions_passed")
if __name__=="__main__":main()
