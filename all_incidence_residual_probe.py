#!/usr/bin/env python3
"""Test whether prospective residual survives removal of semantic 'perspective branch' selection.

Every symmetric incidence is treated identically.  For Nethra j after interval k:

    P_j(k) = sum over every incidence (m,j) of g_mj * (a_m - a_j)

This is exactly the ordinary conductive term already present in F61, before leakage/external source
and before convergence redistribution.  At the next externally observed interval:

    epsilon_j(k+1) = J_j(k+1) - P_j(k)

No edge is tagged predictive/structural/contextual.  The probe tests whether this purely local
quantity retains the useful behavior of the manually selected perspective-current experiment.
"""

from nethra import NethraField


def rk4(field, currents, steps=30, dt=0.03):
    for n in field.nethra:
        n.external=0.0
    for n,j in currents.items():
        n.external=float(j)
    for _ in range(steps):
        a0={n:n.activation for n in field.nethra}
        k1=field._derivative_at(a0)
        a1={n:a0[n]+.5*dt*k1[n] for n in field.nethra}
        k2=field._derivative_at(a1)
        a2={n:a0[n]+.5*dt*k2[n] for n in field.nethra}
        k3=field._derivative_at(a2)
        a3={n:a0[n]+dt*k3[n] for n in field.nethra}
        k4=field._derivative_at(a3)
        for n in field.nethra:
            n.activation=a0[n]+dt*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.0
    for n in field.nethra:
        n.external=0.0


def set_strength(relation,strength):
    for conditions in relation.routes.values():
        conditions.clear()
        conditions[frozenset()]=float(strength)


def net_conductive(field):
    out={n:0.0 for n in field.nethra}
    for a,b,g in field._edges():
        flow=g*(a.activation-b.activation)
        out[a]-=flow
        out[b]+=flow
    return out


def make_mixed(base_strength=None, candidate_strength=1.0, distractor_edges=0):
    f=NethraField(leakage=.6,convergence_gain=0.0)
    a=f.new(); y=f.new(); z=f.new()
    cand=f.new()
    f._route(cand,(a,y),frozenset(),1)
    set_strength(cand,candidate_strength)

    c=None; base=None
    if base_strength is not None:
        c=f.new(); base=f.new()
        f._route(base,(c,y),frozenset(),1)
        set_strength(base,base_strength)

    # Ordinary same-interval structure touching Y.  It receives no source current in the probe.
    distractors=[]
    for _ in range(distractor_edges):
        d=f.new(); rd=f.new()
        f._route(rd,(y,d),frozenset(),1)
        set_strength(rd,10.0)
        distractors.extend((d,rd))

    return f,a,y,z,c,cand,base,distractors


def test_all_edge_residual_correct_vs_wrong():
    f,a,y,z,c,cand,base,d=make_mixed(distractor_edges=3)
    rk4(f,{a:1.0})
    p=net_conductive(f)

    J=.01
    correct={n:-p[n] for n in f.nethra}
    correct[y]=J-p[y]

    wrong={n:-p[n] for n in f.nethra}
    wrong[z]=J-p[z]

    # Score only externally observed source coordinates; no semantic target list is needed.
    correct_source_abs=abs(correct[y])
    wrong_source_abs=abs(wrong[z])
    assert abs(p[y])>0.0
    assert correct_source_abs < J
    assert wrong_source_abs == J
    return p[y],p[z],correct_source_abs,wrong_source_abs


def test_existing_predictor_suppresses_net_residual_contribution():
    rows=[]
    prior=None
    for strength in (None,1.0,5.0,20.0,100.0,1000.0):
        f,a,y,z,c,cand,base,d=make_mixed(base_strength=strength)
        currents={a:1.0}
        if c is not None:
            currents[c]=1.0
        rk4(f,currents)
        p=net_conductive(f)
        # Candidate branch is not selected here; only total local conductive prediction at Y.
        rows.append((strength,y.activation,p[y]))
        if prior is not None:
            assert p[y] >= prior-1e-12
        prior=p[y]
    return rows


def test_distractor_loading_changes_prediction_physically_not_semantically():
    rows=[]
    for count in (0,1,2,4,8):
        f,a,y,z,c,cand,base,d=make_mixed(distractor_edges=count)
        rk4(f,{a:1.0})
        p=net_conductive(f)
        rows.append((count,y.activation,p[y]))
    # More ordinary conductive legs load Y; the field itself should reflect that.
    # We require only finite, deterministic variation, not a preferred direction.
    assert len({round(x[2],15) for x in rows})>1
    return rows


def test_combined_predictors_reduce_source_residual():
    J=.018
    rows=[]
    for count in (0,1,2,3):
        f=NethraField(leakage=.6,convergence_gain=0.0)
        y=f.new()
        contexts=[]
        for _ in range(count):
            c=f.new(); r=f.new()
            f._route(r,(c,y),frozenset(),1)
            set_strength(r,1.0)
            contexts.append(c)
        rk4(f,{c:1.0 for c in contexts})
        p=net_conductive(f)[y]
        rows.append((count,p,J-p,abs(J-p)))
    assert rows[1][3] < rows[0][3]
    assert rows[2][3] < rows[1][3]
    return rows


def test_signed_symmetry_all_edges():
    def one(j):
        f,a,y,z,c,cand,base,d=make_mixed()
        rk4(f,{a:j})
        return net_conductive(f)[y]
    pos=one(1.0)
    neg=one(-1.0)
    assert abs(pos+neg)<1e-12
    return pos,neg


def test_local_edge_credit_sign_needs_no_role_label():
    # For every edge incident on Y, compute current into Y at k and multiply by the
    # next local residual at Y.  This does NOT update conductance; it only checks sign.
    f,a,y,z,c,cand,base,d=make_mixed(base_strength=20.0,distractor_edges=2)
    rk4(f,{a:1.0,c:1.0})
    p=net_conductive(f)
    J=.05
    eps=J-p[y]

    rows=[]
    for x1,x2,g in f._edges():
        if y not in (x1,x2):
            continue
        other=x2 if x1 is y else x1
        current=g*(other.activation-y.activation)
        rows.append((current,current*eps))
    assert rows
    # Both supportive and sink branches may coexist; credit is branch-local.
    return eps,rows


def main():
    print("correct_wrong_all_edges",test_all_edge_residual_correct_vs_wrong())
    print("existing_predictor_total_prediction",test_existing_predictor_suppresses_net_residual_contribution())
    print("distractor_loading",test_distractor_loading_changes_prediction_physically_not_semantically())
    print("combined_predictors",test_combined_predictors_reduce_source_residual())
    print("signed_symmetry",test_signed_symmetry_all_edges())
    print("local_edge_credit",test_local_edge_credit_sign_needs_no_role_label())
    print("all_assertions_passed")


if __name__=="__main__":
    main()
