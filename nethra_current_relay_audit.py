#!/usr/bin/env python3
"""Diagnostic conserved current-relay candidate.

Keep stock symmetric Nethra incidences unchanged.

For each learned relation R at one derivative evaluation:
  S_R = stable source-history support mapped to R
  eta_R = 1 - exp(-S_R/tau)

  common = members common to every route
  V_q = mean activation of route-exclusive members
  current manifestation q* is the positive differential above the route mean

  P_R = total positive ordinary field current currently entering R from all neighbors

  relay_R = eta_R * P_R

Redistribute relay_R out of R toward the non-current manifestation(s), only through route-exclusive
members, and subtract exactly the same current from R. No current is created. No direction bit,
transition table, event signature, prediction target, or threshold is used by the relay.

The relay therefore tests a stronger interpretation of:
    effective prospective influence = earned support x live manifestation x available field current

Core unchanged.
"""
from __future__ import annotations
from collections import defaultdict
import copy,math
from nethra import NethraField

DT=.15
SUB=.0015

class RelayField(NethraField):
    def relation_support(self,r):
        return max((self.history_count[k] for k,x in self.history_relation.items() if x is r),default=0)

    def _derivative_at(self,a):
        # Ordinary stock field first.
        current={n:n.external-self.leakage*a[n] for n in self.nethra}
        neighbors=defaultdict(list)
        for x,y,g in self._edges():
            flow=g*(a[x]-a[y])
            current[x]-=flow;current[y]+=flow
            neighbors[x].append((y,g));neighbors[y].append((x,g))

        # Conserved relation-local relay.
        for r in self.nethra:
            if len(r.routes)<2:continue
            routes=list(r.routes)
            common=set(routes[0])
            for q in routes[1:]:common.intersection_update(q)

            exclusive={}
            V={}
            for q in routes:
                ex=[m for m in q if m not in common]
                if not ex:continue
                exclusive[q]=ex
                V[q]=sum(a[m] for m in ex)/len(ex)
            if len(V)<2:continue

            mu=sum(V.values())/len(V)
            M={q:max(0.0,V[q]-mu) for q in V}
            mt=sum(M.values())
            if mt<=1e-15:continue
            w={q:M[q]/mt for q in V}

            incoming=[]
            for n,g in neighbors[r]:
                p=g*(a[n]-a[r])
                if p>0.0:incoming.append((n,p))
            P=sum(p for _,p in incoming)
            if P<=0.0:continue

            S=self.relation_support(r)
            eta=1.0-math.exp(-max(0.0,float(S))/self.tau)
            relay=eta*P
            if relay<=0.0:continue

            exposure={q:max(0.0,1.0-w[q]) for q in V}
            et=sum(exposure.values())
            if et<=1e-15:continue

            delivered=0.0
            for q,ex in exclusive.items():
                frac=exposure.get(q,0.0)/et
                amount=relay*frac
                if amount<=0.0:continue
                share=amount/len(ex)
                for m in ex:
                    current[m]+=share
                delivered+=amount
            current[r]-=delivered

        return {n:v/self.capacitance for n,v in current.items()}


def integrate(f,duration):
    steps=max(1,round(duration/SUB));dt=duration/steps
    for _ in range(steps):
        a0={n:n.activation for n in f.nethra};k1=f._derivative_at(a0)
        a1={n:a0[n]+.5*dt*k1[n] for n in f.nethra};k2=f._derivative_at(a1)
        a2={n:a0[n]+.5*dt*k2[n] for n in f.nethra};k3=f._derivative_at(a2)
        a3={n:a0[n]+dt*k3[n] for n in f.nethra};k4=f._derivative_at(a3)
        for n in f.nethra:n.activation=a0[n]+dt*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.


def train_stock(cycles,cue):
    s=NethraField(g_min=.20,g_max=1.50,tau=100.,capacitance=1.,leakage=.6,convergence_gain=0.)
    leaves=[s.new() for _ in range(4)];time=s.new();start=(cue+1)%4
    for k in range(cycles*4):
        i=(start+k)%4;time.push(1.);leaves[i].push(1.);s.step(DT)
    for n in s.nethra:n.external=0.
    return s

def as_relay(s):
    f=copy.deepcopy(s);f.__class__=RelayField;return f

def residual(base,h,relay):
    f=as_relay(base) if relay else copy.deepcopy(base)
    leaves=f.nethra[:4];time=f.nethra[4]
    before=[n.activation for n in leaves]
    time.external=1.;integrate(f,h);time.external=0.
    pure=[v*math.exp(-f.leakage*h/f.capacitance) for v in before]
    return [leaves[i].activation-pure[i] for i in range(4)]

def phase():
    print("=== PHASE ===")
    for cycles in (30,100,300):
        bases=[train_stock(cycles,c) for c in range(4)]
        for relay in (False,True):
            rows=[]
            for cue,b in enumerate(bases):
                for steps in (1,2,3,4):
                    res=residual(b,steps*DT,relay);expected=(cue+steps)%4
                    order=sorted(range(4),key=lambda i:res[i],reverse=True)
                    rival=max(res[i] for i in range(4) if i!=expected)
                    rows.append((steps,order.index(expected)+1,res[expected]-rival))
            print("PHASE",{
                "cycles":cycles,"mode":"RELAY" if relay else "STOCK",
                "by_step":{s:{
                    "rank1":sum(x[1]==1 for x in rows if x[0]==s),
                    "mean_rank":sum(x[1] for x in rows if x[0]==s)/4,
                    "mean_margin":sum(x[2] for x in rows if x[0]==s)/4,
                } for s in (1,2,3,4)}
            })

def trajectory():
    print("=== TRAJECTORY ===")
    base=train_stock(300,1)
    for relay in (False,True):
        rows=[]
        for h in (.025,.05,.075,.10,.15,.20,.25,.30,.35,.40,.45,.50,.55,.60,.75,.90,1.20):
            res=residual(base,h,relay);order=sorted(range(4),key=lambda i:res[i],reverse=True)
            rows.append((h,order,res))
        print("TRAJ","RELAY" if relay else "STOCK",rows)

def context():
    print("=== CONTEXT ===")
    for k in (2,4,6):
        # Train stock naturally; use relay only for derivative readout by cloning.
        s=NethraField(g_min=.20,g_max=1.50,tau=100.,capacitance=1.,leakage=.6,convergence_gain=0.)
        leaves=[s.new() for _ in range(2+2*k)];A=0;B=1
        def seg(i):return [2+i,A,B,2+k+i]
        for _ in range(80):
            for i in range(k):
                for x in seg(i):leaves[x].push(1.);s.step(.12)
        for relay in (False,True):
            # Need a separate natural continuation copy per mode.
            f=as_relay(s) if relay else copy.deepcopy(s);ls=f.nethra[:2+2*k]
            ranks=[];margins=[]
            for i in range(k):
                x,y=2+i,2+k+i
                for z in (x,A,B):ls[z].push(1.);f.step(.12)
                for n in f.nethra:n.external=0.
                d=f.derivative();ys=[2+k+j for j in range(k)]
                order=sorted(ys,key=lambda z:d[ls[z]],reverse=True)
                rival=max((d[ls[z]] for z in ys if z!=y),default=d[ls[y]])
                ranks.append(order.index(y)+1);margins.append(d[ls[y]]-rival)
                ls[y].push(1.);f.step(.12)
            print("CONTEXT",{
                "k":k,"mode":"RELAY" if relay else "STOCK",
                "rank1":sum(x==1 for x in ranks),"ranks":ranks,"margins":margins
            })

def conservation():
    print("=== CONSERVATION ===")
    base=as_relay(train_stock(300,1));a={n:n.activation for n in base.nethra}
    # With leakage and external current zeroed, internal terms should sum to zero.
    old_ext={n:n.external for n in base.nethra}
    for n in base.nethra:n.external=0.
    d=base._derivative_at(a)
    internal=sum(base.capacitance*d[n]+base.leakage*a[n] for n in base.nethra)
    for n,v in old_ext.items():n.external=v
    print("internal_current_sum",internal)
    assert abs(internal)<1e-10

def main():
    conservation();phase();trajectory();context();print("all_assertions_passed")
if __name__=="__main__":main()
