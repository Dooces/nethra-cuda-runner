#!/usr/bin/env python3
"""Interval-integrated relation perspective and residual probe.

No instantaneous endpoint state is used as evidence.

For each completed interval k, integrate every symmetric incidence current over the interval using
the same RK4 stage quadrature used for activation integration.  For ordinary relation Nethra R:

    Q_Rm[k] = integral g_Rm * (a_R(t)-a_m(t)) dt

is signed charge/current-flow from R toward member m over the completed interval.

The next completed interval has exact external source charge:

    S_m[k+1] = integral J_m(t) dt

For a relation-local perspective receipt:

    q_R[k->k+1] = sum_m Q_Rm[k] * S_m[k+1]

No member role, direction label, state matcher, probability ledger or endpoint snapshot is used.

For a receiver-local residual:

    P_m[k] = sum incident integrated current into m during k
    epsilon_m[k+1] = S_m[k+1] - P_m[k]

These are audit quantities only; no plasticity rule is installed.
"""

from nethra import NethraField


def set_strength(r,e):
    for c in r.routes.values():
        c.clear(); c[frozenset()]=float(e)


def edges(field):
    return field._edges()


def flows_at(edge_rows,state):
    out=[]
    for a,b,g in edge_rows:
        out.append(g*(state[a]-state[b]))  # oriented a -> b
    return out


def integrate_interval(field,currents,steps=30,dt=.03,reset=True):
    if reset:
        for n in field.nethra:
            n.activation=0.0
    for n in field.nethra:
        n.external=0.0
    for n,j in currents.items():
        n.external=float(j)

    edge_rows=edges(field)
    edge_charge=[0.0]*len(edge_rows)
    source_charge={n:0.0 for n in field.nethra}
    duration=steps*dt

    for _ in range(steps):
        a0={n:n.activation for n in field.nethra}
        k1=field._derivative_at(a0)
        a1={n:a0[n]+.5*dt*k1[n] for n in field.nethra}
        k2=field._derivative_at(a1)
        a2={n:a0[n]+.5*dt*k2[n] for n in field.nethra}
        k3=field._derivative_at(a2)
        a3={n:a0[n]+dt*k3[n] for n in field.nethra}
        k4=field._derivative_at(a3)

        f0=flows_at(edge_rows,a0)
        f1=flows_at(edge_rows,a1)
        f2=flows_at(edge_rows,a2)
        f3=flows_at(edge_rows,a3)
        for i in range(len(edge_rows)):
            edge_charge[i] += dt*(f0[i]+2*f1[i]+2*f2[i]+f3[i])/6.0

        for n in field.nethra:
            n.activation=a0[n]+dt*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.0

    for n,j in currents.items():
        source_charge[n]=float(j)*duration

    for n in field.nethra:
        n.external=0.0

    return edge_rows,edge_charge,source_charge


def relation_flow(field,r,edge_rows,edge_charge):
    out={}
    for (a,b,g),q in zip(edge_rows,edge_charge):
        if a is r:
            out[b]=out.get(b,0.0)+q
        elif b is r:
            out[a]=out.get(a,0.0)-q
    return out


def net_conductive_charge(field,edge_rows,edge_charge):
    out={n:0.0 for n in field.nethra}
    for (a,b,g),q in zip(edge_rows,edge_charge):
        out[a]-=q
        out[b]+=q
    return out


def receipt(local_flow,next_source_charge):
    return sum(local_flow.get(n,0.0)*q for n,q in next_source_charge.items())


def source_charge_for(field,currents,steps=30,dt=.03):
    duration=steps*dt
    return {n:float(j)*duration for n,j in currents.items() if float(j)!=0.0}


def make_pair(evidence=1.0):
    f=NethraField(leakage=.6,convergence_gain=0.0)
    a=f.new(); y=f.new(); z=f.new(); r=f.new()
    f._route(r,(a,y),frozenset(),1)
    set_strength(r,evidence)
    return f,a,y,z,r


def test_temporal_orientation_from_integrated_relation_flow():
    f,a,y,z,r=make_pair()
    er,eq,sc=integrate_interval(f,{a:1.0})
    c=relation_flow(f,r,er,eq)
    ay=receipt(c,source_charge_for(f,{y:.02}))
    aa=receipt(c,source_charge_for(f,{a:.02}))
    az=receipt(c,source_charge_for(f,{z:.02}))

    f2,a2,y2,z2,r2=make_pair()
    er2,eq2,sc2=integrate_interval(f2,{y2:1.0})
    c2=relation_flow(f2,r2,er2,eq2)
    ya=receipt(c2,source_charge_for(f2,{a2:.02}))

    assert ay>0.0 and ya>0.0
    assert aa<0.0
    assert az==0.0
    return ay,aa,az,ya,c,c2


def test_multi_member_integrated_support():
    def one(use_a,use_b):
        f=NethraField(leakage=.6,convergence_gain=0.0)
        a=f.new(); b=f.new(); y=f.new(); r=f.new()
        f._route(r,(a,b,y),frozenset(),1); set_strength(r,1.0)
        cur={}
        if use_a: cur[a]=1.0
        if use_b: cur[b]=1.0
        er,eq,_=integrate_interval(f,cur)
        c=relation_flow(f,r,er,eq)
        q=receipt(c,source_charge_for(f,{y:.02}))
        return q,c[y]
    ao=one(True,False)
    bo=one(False,True)
    both=one(True,True)
    assert both[0] > ao[0] > 0.0
    assert both[0] > bo[0] > 0.0
    return ao,bo,both


def test_competing_integrated_perspectives():
    f=NethraField(leakage=.6,convergence_gain=0.0)
    a=f.new(); b=f.new(); y=f.new()
    r1=f.new(); r2=f.new()
    f._route(r1,(a,y),frozenset(),1); set_strength(r1,1.0)
    f._route(r2,(b,y),frozenset(),1); set_strength(r2,1.0)
    er,eq,_=integrate_interval(f,{a:1.0,b:1.0})
    c1=relation_flow(f,r1,er,eq)
    c2=relation_flow(f,r2,er,eq)
    nextq=source_charge_for(f,{y:.02})
    q1=receipt(c1,nextq); q2=receipt(c2,nextq)
    assert q1>0.0 and q2>0.0
    assert abs(q1-q2)<1e-15
    return q1,q2,q1+q2,c1[y],c2[y]


def test_integrated_loading_suppresses_weak_relation():
    rows=[]
    for strong in (1.0,5.0,20.0,100.0,1000.0):
        f=NethraField(leakage=.6,convergence_gain=0.0)
        a=f.new(); c=f.new(); y=f.new()
        weak=f.new(); base=f.new()
        f._route(weak,(a,y),frozenset(),1); set_strength(weak,1.0)
        f._route(base,(c,y),frozenset(),1); set_strength(base,strong)
        er,eq,_=integrate_interval(f,{a:1.0,c:1.0})
        cw=relation_flow(f,weak,er,eq)
        cb=relation_flow(f,base,er,eq)
        nq=source_charge_for(f,{y:.02})
        qw=receipt(cw,nq); qb=receipt(cb,nq)
        rows.append((strong,qw,qb,cw[y],cb[y]))
    assert all(rows[i][1] >= rows[i+1][1]-1e-15 for i in range(len(rows)-1))
    return rows


def test_integrated_receiver_residual_combines_predictors():
    target_current=.018
    rows=[]
    for count in (0,1,2,3):
        f=NethraField(leakage=.6,convergence_gain=0.0)
        y=f.new(); contexts=[]
        for _ in range(count):
            c=f.new(); r=f.new()
            f._route(r,(c,y),frozenset(),1); set_strength(r,1.0)
            contexts.append(c)
        er,eq,_=integrate_interval(f,{c:1.0 for c in contexts})
        p=net_conductive_charge(f,er,eq)[y]
        next_source=source_charge_for(f,{y:target_current})[y]
        eps=next_source-p
        rows.append((count,p,next_source,eps,abs(eps)))
    assert rows[1][4] < rows[0][4]
    # Do not require every added predictor to improve: overprediction must be allowed.
    assert min(r[4] for r in rows[1:]) < rows[0][4]
    return rows


def test_no_endpoint_dependence():
    # Different numerical step counts over the same duration should converge on the same interval
    # receipt; this guards against accidentally depending on a sampled endpoint current.
    vals=[]
    for steps,dt in ((15,.06),(30,.03),(60,.015),(120,.0075)):
        f,a,y,z,r=make_pair()
        er,eq,_=integrate_interval(f,{a:1.0},steps=steps,dt=dt)
        c=relation_flow(f,r,er,eq)
        nextq=source_charge_for(f,{y:.02},steps=steps,dt=dt)
        vals.append((steps,receipt(c,nextq),c[y]))
    ref=vals[-1][1]
    assert max(abs(v[1]-ref) for v in vals) < 2e-9
    return vals


def main():
    print("integrated_orientation",test_temporal_orientation_from_integrated_relation_flow())
    print("integrated_multi_member",test_multi_member_integrated_support())
    print("integrated_competing",test_competing_integrated_perspectives())
    print("integrated_loading",test_integrated_loading_suppresses_weak_relation())
    print("integrated_receiver_residual",test_integrated_receiver_residual_combines_predictors())
    print("step_refinement",test_no_endpoint_dependence())
    print("all_assertions_passed")


if __name__=="__main__":
    main()
