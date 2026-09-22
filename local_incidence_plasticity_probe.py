#!/usr/bin/env python3
"""Orientation-invariant local incidence plasticity sign probe.

No conductance update is installed.

For one symmetric incidence i--j, choose either algebraic orientation and define

    f_ij(k) = g_ij * (a_i(k) - a_j(k))

as current from i toward j at the end of interval k.

For every Nethra:
    P_j(k) = net conductive current into j at end of k
    epsilon_j(k+1) = J_j(k+1) - P_j(k)

The purely local symmetric edge receipt tested here is

    q_ij = f_ij(k) * (epsilon_j(k+1) - epsilon_i(k+1))

Swapping i and j changes the sign of both factors, so q_ij is unchanged.  Positive/negative q is
only an audit signal here; this file does not define a conductance update law.
"""

from nethra import NethraField


def rk4(field,currents,steps=30,dt=.03):
    for n in field.nethra: n.external=0.0
    for n,j in currents.items(): n.external=float(j)
    for _ in range(steps):
        a0={n:n.activation for n in field.nethra}
        k1=field._derivative_at(a0)
        a1={n:a0[n]+.5*dt*k1[n] for n in field.nethra}; k2=field._derivative_at(a1)
        a2={n:a0[n]+.5*dt*k2[n] for n in field.nethra}; k3=field._derivative_at(a2)
        a3={n:a0[n]+dt*k3[n] for n in field.nethra}; k4=field._derivative_at(a3)
        for n in field.nethra:
            n.activation=a0[n]+dt*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.0
    for n in field.nethra: n.external=0.0


def set_strength(r,e):
    for c in r.routes.values():
        c.clear(); c[frozenset()]=float(e)


def edge_rows(field):
    return [(a,b,g,g*(a.activation-b.activation)) for a,b,g in field._edges()]


def net_conductive(field):
    p={n:0.0 for n in field.nethra}
    for a,b,g,f in edge_rows(field):
        p[a]-=f; p[b]+=f
    return p


def credits(field,next_source):
    p=net_conductive(field)
    eps={n:float(next_source.get(n,0.0))-p[n] for n in field.nethra}
    rows=[]
    for a,b,g,f in edge_rows(field):
        q=f*(eps[b]-eps[a])
        # orientation-invariance assertion
        q_rev=(-f)*(eps[a]-eps[b])
        assert abs(q-q_rev)<1e-15
        rows.append((a,b,f,eps[a],eps[b],q))
    return p,eps,rows


def build_chain(extra_wrong=True):
    f=NethraField(leakage=.6,convergence_gain=0.0)
    a=f.new(); y=f.new(); z=f.new()
    r=f.new()
    f._route(r,(a,y),frozenset(),1); set_strength(r,1.0)
    # z is deliberately disconnected unless requested by another path later.
    return f,a,y,z,r


def edge_credit(rows,x,y):
    for a,b,flow,ea,eb,q in rows:
        if frozenset((a,b))==frozenset((x,y)):
            return q,flow,ea,eb
    raise AssertionError("edge absent")


def test_correct_vs_wrong_path_receipts():
    f,a,y,z,r=build_chain()
    rk4(f,{a:1.0})

    _,_,correct=credits(f,{y:.02})
    _,_,wrong=credits(f,{z:.02})
    c_ar=edge_credit(correct,a,r)
    c_ry=edge_credit(correct,r,y)
    w_ar=edge_credit(wrong,a,r)
    w_ry=edge_credit(wrong,r,y)

    # Audit what the local law actually does rather than assuming it.
    return {
        "correct_A_R":c_ar,
        "correct_R_Y":c_ry,
        "wrong_A_R":w_ar,
        "wrong_R_Y":w_ry,
    }


def test_reverse_sequence_is_not_hardcoded_direction():
    f,a,y,z,r=build_chain()
    rk4(f,{y:1.0})
    _,_,rows=credits(f,{a:.02})
    return {
        "Y_R":edge_credit(rows,y,r),
        "R_A":edge_credit(rows,r,a),
    }


def test_simultaneous_has_no_prior_temporal_receipt_from_zero():
    f,a,y,z,r=build_chain()
    # Before a zero-origin interval evolves, all branch flow is zero.
    p,eps,rows=credits(f,{a:1.0,y:1.0})
    vals=[abs(row[-1]) for row in rows]
    assert max(vals,default=0.0)==0.0
    return vals


def test_combined_predictors_share_local_receipts():
    f=NethraField(leakage=.6,convergence_gain=0.0)
    y=f.new(); a=f.new(); b=f.new()
    r1=f.new(); r2=f.new()
    f._route(r1,(a,y),frozenset(),1); set_strength(r1,1.0)
    f._route(r2,(b,y),frozenset(),1); set_strength(r2,1.0)
    rk4(f,{a:1.0,b:1.0})
    _,eps,rows=credits(f,{y:.02})
    return (
        eps[y],
        edge_credit(rows,r1,y),
        edge_credit(rows,r2,y),
        edge_credit(rows,a,r1),
        edge_credit(rows,b,r2),
    )


def main():
    result=test_correct_vs_wrong_path_receipts()
    print("correct_wrong_path_receipts",result)
    print("reverse_sequence",test_reverse_sequence_is_not_hardcoded_direction())
    print("simultaneous_zero_origin",test_simultaneous_has_no_prior_temporal_receipt_from_zero())
    print("combined_predictors",test_combined_predictors_share_local_receipts())

    # Promotion gate for this candidate: a correct A then Y sequence must give positive receipt to
    # both incidences of the ordinary A--R--Y path, while an unrelated next Z must not.
    assert result["correct_A_R"][0] > 0.0
    assert result["correct_R_Y"][0] > 0.0
    assert result["wrong_A_R"][0] <= 0.0
    assert result["wrong_R_Y"][0] <= 0.0
    print("candidate_sign_gate_passed")


if __name__=="__main__":
    main()
