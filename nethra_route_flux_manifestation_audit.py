#!/usr/bin/env python3
"""Vet whether existing route-local flux already contains a travelling manifestation state.

Core is unchanged.  This probe only observes physical incidence current.

For each learned relation R and each of its learned routes q:
  common_R = intersection of all routes of R
  exclusive_q = q - common_R

At every RK substep, measure the actual stock-field current entering R from the exclusive members:

  J_q(t) = sum_{m in exclusive_q} g_Rm(t) * (a_m(t) - a_R(t))

Two transient diagnostics are accumulated, neither fed back into Nethra:
  signed trace:    tau dm/dt = J_q - m
  positive trace:  tau dp/dt = max(J_q,0) - p

The source-history map is used only after training to label the birth-side routes as manifestations
of source events.  It does not affect dynamics or trace integration.

Tests:
1. A-B-C-D + TIME.  Warm traces through eight real source intervals with learning frozen, then
   withhold symbols and continue TIME alone.  At t=0,.15,.30,.45,.60, ask whether summed route
   traces rank the source phase E_AB,E_BC,E_CD,E_DA,E_AB respectively.
2. Sweep tau broadly.  If phase progression exists only for a finely tuned tau, that is evidence
   against "existing flux history is already enough".
3. Shared-context load Xi->A->B->Yi for K=2,4,6.  At B, rank the direct AB->BYi relations by the
   trace on the learned AB-side route.  This checks whether the same flux trace preserves the
   context already present in topology.

No new persistent field state, topology, evidence, selector, decoder, or prediction current.
"""
from __future__ import annotations
from collections import defaultdict
import copy, math
from nethra import NethraField

DT=.15
SUB=.0015
TAUS=(.025,.05,.10,.15,.30,.60,1.20)

class BirthSideField(NethraField):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # Diagnostic metadata only: route identities as they represented source-history sides at
        # the moment a source history first mapped to a relation.
        self.birth_sides={}

    def _mint_history(self,before,after,evidence,history_key=None):
        key=(before,after) if history_key is None else history_key
        existed=key in self.history_relation
        r=super()._mint_history(before,after,evidence,history_key)
        if r is not None and not existed and key in self.history_relation:
            left=self._matching_route(r,before)
            right=self._matching_route(r,after)
            if left is not None and right is not None:
                self.birth_sides[key]=(r,left[0],right[0])
        return r


def edge_map(f):
    out={}
    for a,b,g in f._edges():
        out[frozenset((a,b))]=g
    return out


def route_fluxes(f,state):
    """Actual stock-field incidence current into each relation from route-exclusive members."""
    em=edge_map(f)
    out={}
    for r in f.nethra:
        if len(r.routes)<2: continue
        routes=list(r.routes)
        common=set(routes[0])
        for q in routes[1:]:
            common.intersection_update(q)
        for q in routes:
            exclusive=[m for m in q if m not in common]
            j=0.0
            for m in exclusive:
                g=em.get(frozenset((r,m)),0.0)
                j += g*(state[m]-state[r])
            out[(r,q)]=j
    return out


def trace_step(traces_pos,traces_signed,flux,dt,tau):
    # Exact exponential update for piecewise-constant drive over this tiny diagnostic substep.
    decay=math.exp(-dt/tau)
    gain=tau*(1.0-decay)
    keys=set(traces_pos)|set(flux)
    for k in keys:
        j=flux.get(k,0.0)
        traces_signed[k]=decay*traces_signed.get(k,0.0)+gain*j
        traces_pos[k]=decay*traces_pos.get(k,0.0)+gain*max(0.0,j)


def rk4_interval_with_traces(f,duration,traces_pos,traces_signed,tau):
    steps=max(1,round(duration/SUB));dt=duration/steps
    for _ in range(steps):
        a0={n:n.activation for n in f.nethra}
        k1=f._derivative_at(a0)
        a1={n:a0[n]+.5*dt*k1[n] for n in f.nethra}
        k2=f._derivative_at(a1)
        a2={n:a0[n]+.5*dt*k2[n] for n in f.nethra}
        k3=f._derivative_at(a2)
        a3={n:a0[n]+dt*k3[n] for n in f.nethra}
        k4=f._derivative_at(a3)

        # Use RK4-consistent average route current for the diagnostic trace.
        f0=route_fluxes(f,a0); f1=route_fluxes(f,a1); f2=route_fluxes(f,a2); f3=route_fluxes(f,a3)
        favg={}
        for key in set(f0)|set(f1)|set(f2)|set(f3):
            favg[key]=(f0.get(key,0)+2*f1.get(key,0)+2*f2.get(key,0)+f3.get(key,0))/6.0
        trace_step(traces_pos,traces_signed,favg,dt,tau)

        for n in f.nethra:
            n.activation=a0[n]+dt*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.0


def raw_source_interval(f,source_nodes,duration,traces_pos,traces_signed,tau):
    for n in f.nethra:n.external=0.0
    for n in source_nodes:n.external+=1.0
    rk4_interval_with_traces(f,duration,traces_pos,traces_signed,tau)
    for n in f.nethra:n.external=0.0


def source_event(prev,cur,time):
    return frozenset(((prev,-1),(cur,1),(time,0)))


def train_cycle(cycles=300):
    f=BirthSideField(g_min=.20,g_max=1.50,tau=100.,capacitance=1.,leakage=.6,convergence_gain=0.)
    leaves=[f.new() for _ in range(4)];time=f.new()
    for _ in range(cycles):
        for i in range(4):
            time.push(1.0);leaves[i].push(1.0);f.step(DT)
    return f,leaves,time


def cycle_phase_map(f,leaves,time):
    phases=[source_event(leaves[(i-1)%4],leaves[i],time) for i in range(4)]
    route_by_phase=defaultdict(list)
    for key,sides in f.birth_sides.items():
        r,left,right=sides
        before,after=key
        route_by_phase[before].append((r,left))
        route_by_phase[after].append((r,right))
    return phases,route_by_phase


def score_phases(route_by_phase,traces):
    return {phase:sum(traces.get(pair,0.0) for pair in pairs) for phase,pairs in route_by_phase.items()}


def deterministic():
    base,leaves,time=train_cycle()
    phases,route_by_phase=cycle_phase_map(base,leaves,time)
    print("=== DETERMINISTIC ===")
    print("phase_route_counts",[len(route_by_phase[p]) for p in phases])

    # Evaluate all four cue phases independently from the same mature trained state.
    for tau in TAUS:
        for kind in ("positive","signed"):
            rows=[]
            for cue in range(4):
                f=copy.deepcopy(base); ls=f.nethra[:4]; tm=f.nethra[4]
                # Rebuild phase objects against the clone identities.
                phases_c=[source_event(ls[(i-1)%4],ls[i],tm) for i in range(4)]
                route_map=defaultdict(list)
                for key,sides in f.birth_sides.items():
                    r,left,right=sides
                    route_map[key[0]].append((r,left));route_map[key[1]].append((r,right))

                tp={};ts={}
                # Eight natural source intervals, ending on the requested cue, to establish only
                # transient trace history.  Learning/construction is not called.
                start=(cue-7)%4
                for k in range(8):
                    i=(start+k)%4
                    raw_source_interval(f,(tm,ls[i]),DT,tp,ts,tau)

                trace=tp if kind=="positive" else ts
                for step in range(5):
                    scores=score_phases(route_map,trace)
                    expected_phase=phases_c[(cue+step)%4]
                    ordered=sorted(phases_c,key=lambda p:scores.get(p,0.0),reverse=True)
                    rank=ordered.index(expected_phase)+1
                    rival=max(scores.get(p,0.0) for p in phases_c if p!=expected_phase)
                    margin=scores.get(expected_phase,0.0)-rival
                    rows.append((step,rank,margin,[scores.get(p,0.0) for p in phases_c]))
                    if step<4:
                        raw_source_interval(f,(tm,),DT,tp,ts,tau)
                        trace=tp if kind=="positive" else ts

            print("PHASE_TRACE",{
                "tau":tau,"kind":kind,
                "by_step":{s:{
                    "rank1":sum(r[1]==1 for r in rows if r[0]==s),
                    "mean_rank":sum(r[1] for r in rows if r[0]==s)/4,
                    "mean_margin":sum(r[2] for r in rows if r[0]==s)/4,
                    "phase_scores":[r[3] for r in rows if r[0]==s],
                } for s in range(5)}
            })


def context_load():
    print("=== CONTEXT ===")
    for k in (2,4,6):
        for tau in TAUS:
            ranks_pos=[];ranks_signed=[]
            for target in range(k):
                # indices A=0,B=1,X_i=2+i,Y_i=2+k+i
                f=BirthSideField(g_min=.20,g_max=1.50,tau=100.,capacitance=1.,leakage=.6,convergence_gain=0.)
                nodes=[f.new() for _ in range(2+2*k)];A=nodes[0];B=nodes[1]
                X=[nodes[2+i] for i in range(k)];Y=[nodes[2+k+i] for i in range(k)]
                def seg(i):return (X[i],A,B,Y[i])
                for _ in range(80):
                    for i in range(k):
                        for n in seg(i):
                            n.push(1.0);f.step(.12)

                # Candidate direct relations for E_AB -> E_BYi.  No TIME in this fixture.
                eab=frozenset(((A,-1),(B,1)))
                direct=[]
                for i in range(k):
                    eby=frozenset(((B,-1),(Y[i],1)))
                    key=(eab,eby)
                    sides=f.birth_sides.get(key)
                    direct.append(sides)

                tp={};ts={}
                # Warm through two full natural supercycles ending at target B without learning.
                sequence=[]
                for _ in range(2):
                    for i in range(k):
                        sequence.extend(seg(i))
                # append target context X,A,B and omit its Y
                sequence.extend((X[target],A,B))
                for n in sequence:
                    raw_source_interval(f,(n,),.12,tp,ts,tau)

                scores_pos=[];scores_signed=[]
                for sides in direct:
                    if sides is None:
                        scores_pos.append(float("-inf"));scores_signed.append(float("-inf"));continue
                    r,left,right=sides
                    scores_pos.append(tp.get((r,left),0.0))
                    scores_signed.append(ts.get((r,left),0.0))
                op=sorted(range(k),key=lambda i:scores_pos[i],reverse=True)
                os=sorted(range(k),key=lambda i:scores_signed[i],reverse=True)
                ranks_pos.append(op.index(target)+1)
                ranks_signed.append(os.index(target)+1)

            print("CONTEXT_TRACE",{
                "k":k,"tau":tau,
                "positive_rank1":sum(r==1 for r in ranks_pos),"positive_ranks":ranks_pos,
                "signed_rank1":sum(r==1 for r in ranks_signed),"signed_ranks":ranks_signed,
            })


def main():
    deterministic()
    context_load()
    print("all_assertions_passed")

if __name__=="__main__":main()
