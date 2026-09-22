#!/usr/bin/env python3
"""Shadow plasticity using only an ordinary Nethra's interval-integrated field tension.

No global history/support/outcome counters and no probability baseline are used.

For each trial:
  1. context current drives an existing ordinary relation during interval k;
  2. integrate positive incidence flow over that interval;
  3. next external source charge defines receiver residual against all positive incoming field flow;
  4. relation-local tension T_r = sum p_rm * epsilon_m;
  5. shadow evidence changes only by e_r <- max(0, e_r + eta*T_r).

This probe does not solve construction. Relations are supplied as fixtures so only the local
strength law is tested.
"""

import random
from nethra import NethraField

DURATION=.9
STEPS=30
DT=DURATION/STEPS
TARGET_CURRENT=.02


def set_evidence(r,e):
    for c in r.routes.values():
        c.clear(); c[frozenset()]=max(0.0,float(e))


def get_evidence(r):
    vals=[]
    for c in r.routes.values(): vals.extend(c.values())
    return max((float(v) for v in vals),default=0.0)


def integrate(field,currents):
    for n in field.nethra:
        n.activation=0.0; n.external=0.0
    for n,j in currents.items(): n.external=float(j)
    es=field._edges(); charges=[0.0]*len(es)
    def flows(state):
        return [g*(state[a]-state[b]) for a,b,g in es]
    for _ in range(STEPS):
        a0={n:n.activation for n in field.nethra}
        k1=field._derivative_at(a0)
        a1={n:a0[n]+.5*DT*k1[n] for n in field.nethra}; k2=field._derivative_at(a1)
        a2={n:a0[n]+.5*DT*k2[n] for n in field.nethra}; k3=field._derivative_at(a2)
        a3={n:a0[n]+DT*k3[n] for n in field.nethra}; k4=field._derivative_at(a3)
        fs=(flows(a0),flows(a1),flows(a2),flows(a3))
        for i in range(len(es)):
            charges[i]+=DT*(fs[0][i]+2*fs[1][i]+2*fs[2][i]+fs[3][i])/6.0
        for n in field.nethra:
            n.activation=a0[n]+DT*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.0
    for n in field.nethra: n.external=0.0
    return es,charges


def positive_outflows(field,es,charges):
    out={n:{} for n in field.nethra}
    inc={n:0.0 for n in field.nethra}
    for (a,b,g),q in zip(es,charges):
        if q>0:
            out[a][b]=out[a].get(b,0.0)+q; inc[b]+=q
        elif q<0:
            z=-q; out[b][a]=out[b].get(a,0.0)+z; inc[a]+=z
    return out,inc


def tensions(field,out,inc,next_currents):
    src={n:0.0 for n in field.nethra}
    for n,j in next_currents.items(): src[n]=DURATION*float(j)
    eps={n:src[n]-inc[n] for n in field.nethra}
    t={r:sum(q*eps[m] for m,q in out[r].items()) for r in field.nethra}
    return t,eps


def add_relation(f,*members,evidence=0.0):
    r=f.new(); f._route(r,members,frozenset(),1); set_evidence(r,evidence); return r


def train_single(prob,eta,seed,trials=5000):
    rng=random.Random(seed)
    f=NethraField(leakage=.6,convergence_gain=0.0)
    a=f.new(); y=f.new(); r=add_relation(f,a,y,evidence=0.0)
    tail=[]
    for k in range(trials):
        es,q=integrate(f,{a:1.0}); out,inc=positive_outflows(f,es,q)
        nxt={y:TARGET_CURRENT} if rng.random()<prob else {}
        t,eps=tensions(f,out,inc,nxt)
        set_evidence(r,get_evidence(r)+eta*t[r])
        if k>=trials-500: tail.append((get_evidence(r),t[r]))
    es,q=integrate(f,{a:1.0}); out,inc=positive_outflows(f,es,q)
    return get_evidence(r),out[r].get(y,0.0),sum(x for x,_ in tail)/len(tail),sum(t for _,t in tail)/len(tail)


def test_frequency_order_across_step_sizes():
    all_rows={}
    for eta in (50.0,200.0,1000.0,5000.0):
        rows=[]
        for i,p in enumerate((.1,.5,.9)):
            rows.append((p,)+train_single(p,eta,9000+i))
        all_rows[eta]=rows
        # .1 is pinned at the minimum because even g_min overpredicts its expected source.
        assert rows[0][0]==.1 and rows[0][1] < 1e-9
        assert rows[0][2] < rows[1][2] < rows[2][2]
        assert rows[1][1] < rows[2][1]
    return all_rows


def test_regime_change_reverses_strength():
    rng=random.Random(9100)
    eta=1000.0
    f=NethraField(leakage=.6,convergence_gain=0.0)
    a=f.new(); y=f.new(); r=add_relation(f,a,y,evidence=0.0)
    def phase(prob,n):
        for _ in range(n):
            es,q=integrate(f,{a:1.0}); out,inc=positive_outflows(f,es,q)
            nxt={y:TARGET_CURRENT} if rng.random()<prob else {}
            t,_=tensions(f,out,inc,nxt)
            set_evidence(r,get_evidence(r)+eta*t[r])
    phase(.9,4000); high=get_evidence(r)
    phase(.1,4000); low=get_evidence(r)
    assert high>5.0
    assert low < high*.1
    return high,low


def test_established_predictor_starves_late_redundant_relation():
    rng=random.Random(9200)
    eta=1000.0
    f=NethraField(leakage=.6,convergence_gain=0.0)
    c=f.new(); a=f.new(); y=f.new()
    base=add_relation(f,c,y,evidence=0.0)

    # Establish background predictor.
    for _ in range(5000):
        es,q=integrate(f,{c:1.0}); out,inc=positive_outflows(f,es,q)
        nxt={y:TARGET_CURRENT} if rng.random()<.9 else {}
        t,_=tensions(f,out,inc,nxt)
        set_evidence(base,get_evidence(base)+eta*t[base])
    base_before=get_evidence(base)

    cand=add_relation(f,a,y,evidence=0.0)
    for _ in range(5000):
        aon=rng.random()<.5
        cur={c:1.0}
        if aon: cur[a]=1.0
        es,q=integrate(f,cur); out,inc=positive_outflows(f,es,q)
        nxt={y:TARGET_CURRENT} if rng.random()<.9 else {}
        t,_=tensions(f,out,inc,nxt)
        set_evidence(base,get_evidence(base)+eta*t[base])
        if aon:
            set_evidence(cand,get_evidence(cand)+eta*t[cand])

    assert get_evidence(cand) < max(1.0,base_before*.1)
    return base_before,get_evidence(base),get_evidence(cand)


def test_exact_symmetric_duplicates_remain_symmetric():
    eta=1000.0
    rng=random.Random(9300)
    f=NethraField(leakage=.6,convergence_gain=0.0)
    a=f.new(); b=f.new(); y=f.new()
    r1=add_relation(f,a,y,evidence=0.0)
    r2=add_relation(f,b,y,evidence=0.0)
    for _ in range(4000):
        es,q=integrate(f,{a:1.0,b:1.0}); out,inc=positive_outflows(f,es,q)
        nxt={y:TARGET_CURRENT} if rng.random()<.9 else {}
        t,_=tensions(f,out,inc,nxt)
        set_evidence(r1,get_evidence(r1)+eta*t[r1])
        set_evidence(r2,get_evidence(r2)+eta*t[r2])
    assert abs(get_evidence(r1)-get_evidence(r2))<1e-12
    return get_evidence(r1),get_evidence(r2)


def test_two_different_predictors_can_share_residual():
    # Different initial strengths; both are active every time. Check whether local tension moves the
    # total toward an equilibrium instead of requiring a winner.
    eta=1000.0
    f=NethraField(leakage=.6,convergence_gain=0.0)
    a=f.new(); b=f.new(); y=f.new()
    r1=add_relation(f,a,y,evidence=0.0)
    r2=add_relation(f,b,y,evidence=20.0)
    before=(get_evidence(r1),get_evidence(r2))
    tail=[]
    for _ in range(5000):
        es,q=integrate(f,{a:1.0,b:1.0}); out,inc=positive_outflows(f,es,q)
        t,eps=tensions(f,out,inc,{y:TARGET_CURRENT})
        set_evidence(r1,get_evidence(r1)+eta*t[r1])
        set_evidence(r2,get_evidence(r2)+eta*t[r2])
        tail.append(abs(eps[y]))
    after=(get_evidence(r1),get_evidence(r2))
    assert sum(tail[-500:])/500 < sum(tail[:500])/500
    return before,after,sum(tail[:500])/500,sum(tail[-500:])/500


def main():
    print("frequency_order",test_frequency_order_across_step_sizes())
    print("regime_change",test_regime_change_reverses_strength())
    print("late_redundant",test_established_predictor_starves_late_redundant_relation())
    print("symmetric_duplicates",test_exact_symmetric_duplicates_remain_symmetric())
    print("shared_residual",test_two_different_predictors_can_share_residual())
    print("all_assertions_passed")


if __name__=="__main__":
    main()
