#!/usr/bin/env python3
"""Vet the proposed residual-squared conductance tension against actual Nethra field dynamics.

Questions tested:
1. Is Q = g*Phi exactly recoverable from one per-Nethra activation integral?
2. Is epsilon*Phi the actual derivative of residual squared loss w.r.t. conductance, or only the
   direct/frozen-trajectory term?
3. Does the finite recruitment formula match an actual newly added incidence?
4. If persistent plasticity updates evidence e and g=g(e), what chain factor is required?
5. Does signed conductive charge capture all field-carried influence when F61 convergence is active?

No learning rule is installed and no core file is changed.
"""

import math
from nethra import NethraField

T=.6
STEPS=120
DT=T/STEPS
U_NEXT=.012


def integrate(field, currents):
    for n in field.nethra:
        n.activation=0.0
        n.external=0.0
    for n,j in currents.items():
        n.external=float(j)

    es=field._edges()
    direct_q=[0.0]*len(es)
    activation_integral={n:0.0 for n in field.nethra}
    convergence_charge={n:0.0 for n in field.nethra}

    def edge_flow(state):
        return [g*(state[a]-state[b]) for a,b,g in es]

    def conductive_current(state):
        cur={n:0.0 for n in field.nethra}
        for a,b,g in es:
            q=g*(state[a]-state[b])
            cur[a]-=q
            cur[b]+=q
        return cur

    def conv_current(state):
        # Exact difference between full F61 current*C and external-leak-conductive components.
        full={n:field.capacitance*v for n,v in field._derivative_at(state).items()}
        cond=conductive_current(state)
        out={}
        for n in field.nethra:
            out[n]=full[n] - n.external + field.leakage*state[n] - cond[n]
        return out

    for _ in range(STEPS):
        a0={n:n.activation for n in field.nethra}
        k1=field._derivative_at(a0)
        a1={n:a0[n]+.5*DT*k1[n] for n in field.nethra}
        k2=field._derivative_at(a1)
        a2={n:a0[n]+.5*DT*k2[n] for n in field.nethra}
        k3=field._derivative_at(a2)
        a3={n:a0[n]+DT*k3[n] for n in field.nethra}
        k4=field._derivative_at(a3)

        fs=(edge_flow(a0),edge_flow(a1),edge_flow(a2),edge_flow(a3))
        cs=(conv_current(a0),conv_current(a1),conv_current(a2),conv_current(a3))

        for i in range(len(es)):
            direct_q[i]+=DT*(fs[0][i]+2*fs[1][i]+2*fs[2][i]+fs[3][i])/6.0

        for n in field.nethra:
            activation_integral[n]+=DT*(a0[n]+2*a1[n]+2*a2[n]+a3[n])/6.0
            convergence_charge[n]+=DT*(cs[0][n]+2*cs[1][n]+2*cs[2][n]+cs[3][n])/6.0
            n.activation=a0[n]+DT*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.0

    for n in field.nethra:
        n.external=0.0

    return es,direct_q,activation_integral,convergence_charge


def make_passive(g_ry, extra=False):
    f=NethraField(leakage=.6,convergence_gain=0.0)
    a=f.new(); r=f.new(); y=f.new()
    z=None
    g_ar=.9
    g_yz=.55 if extra else None
    if extra:
        z=f.new()

    def edges():
        rows=[(a,r,g_ar)]
        if g_ry>0.0:
            rows.append((r,y,g_ry))
        if z is not None:
            rows.append((y,z,g_yz))
        return tuple(rows)
    f._edges=edges
    return f,a,r,y,z


def incoming_q(es,q,receiver):
    total=0.0
    for (a,b,g),v in zip(es,q):
        if b is receiver:
            total+=v
        elif a is receiver:
            total-=v
    return total


def phi_from_integral(A,a,b):
    return A[a]-A[b]


def passive_measure(g,extra=False):
    f,a,r,y,z=make_passive(g,extra)
    es,q,A,C=integrate(f,{a:1.0})
    pred=incoming_q(es,q,y)
    eps=U_NEXT-pred
    E=.5*eps*eps
    phi=phi_from_integral(A,r,y)
    return {
        "E":E,"pred":pred,"eps":eps,"phi":phi,
        "es":es,"q":q,"A":A,"nodes":(f,a,r,y,z),
    }


def test_node_integral_reconstructs_edge_charge():
    row=passive_measure(.73,extra=True)
    worst=0.0
    for (a,b,g),q in zip(row["es"],row["q"]):
        reconstructed=g*(row["A"][a]-row["A"][b])
        worst=max(worst,abs(q-reconstructed))
    assert worst < 2e-15
    return worst,len(row["A"]),len(row["es"])


def finite_gradient(g,h,extra=False):
    lo=passive_measure(g-h,extra)
    hi=passive_measure(g+h,extra)
    mid=passive_measure(g,extra)
    dE=(hi["E"]-lo["E"])/(2*h)
    dP=(hi["pred"]-lo["pred"])/(2*h)
    candidate_grad=mid["eps"]*mid["phi"]  # - direct dE/dg if trajectories frozen
    actual_minus_dE=-dE
    return {
        "g":g,"h":h,
        "phi":mid["phi"],"eps":mid["eps"],
        "dP_full":dP,
        "candidate_dP_direct":mid["phi"],
        "minus_dE_full":actual_minus_dE,
        "candidate_tension":candidate_grad,
        "relerr_dP":abs(dP-mid["phi"])/max(1e-18,abs(dP)),
        "relerr_T":abs(actual_minus_dE-candidate_grad)/max(1e-18,abs(actual_minus_dE)),
    }


def test_full_gradient_vs_frozen_trajectory_term():
    rows=[]
    for extra in (False,True):
        for g in (.05,.2,.5,1.0,1.4):
            for h in (1e-5,1e-4):
                rows.append((extra,finite_gradient(g,h,extra)))
    # We expect the direct term to have the same sign in this simple topology, but it must NOT be
    # claimed exact unless the trajectory term is numerically negligible.
    assert any(row[1]["relerr_T"] > .01 for row in rows)
    return rows


def recruitment_actual(delta_g,extra=False):
    base=passive_measure(0.0,extra)
    pert=passive_measure(delta_g,extra)
    phi0=base["phi"]
    dq_virtual=delta_g*phi0
    predicted_collapse=base["eps"]*dq_virtual-.5*dq_virtual*dq_virtual
    actual_collapse=base["E"]-pert["E"]
    return {
        "dg":delta_g,
        "phi0":phi0,
        "predicted":predicted_collapse,
        "actual":actual_collapse,
        "abs_error":abs(actual_collapse-predicted_collapse),
        "rel_error":abs(actual_collapse-predicted_collapse)/max(1e-18,abs(actual_collapse)),
    }


def test_virtual_recruitment_is_local_approximation_not_exact_finite_change():
    rows=[recruitment_actual(dg,extra=True) for dg in (1e-6,1e-5,1e-4,1e-3,.01,.05,.2)]
    # It should converge as dg -> 0 and deviate for finite dg due to changed trajectories.
    assert rows[0]["rel_error"] < rows[-1]["rel_error"]
    assert rows[0]["rel_error"] < 1e-4
    assert rows[-1]["rel_error"] > 1e-3
    return rows


def g_of_e(f,e):
    return f.g_min+(f.g_max-f.g_min)*(1-math.exp(-e/f.tau))


def dg_de(f,e):
    return (f.g_max-f.g_min)/f.tau*math.exp(-e/f.tau)


def E_from_e(e):
    f0=NethraField()
    g=g_of_e(f0,e)
    return passive_measure(g,extra=True)


def test_evidence_chain_rule():
    f0=NethraField()
    rows=[]
    for e in (0.0,10.0,50.0,100.0,300.0,800.0):
        h=1e-4
        lo=E_from_e(max(0.0,e-h))
        hi=E_from_e(e+h)
        mid=E_from_e(e)
        if e==0.0:
            # one-sided at boundary
            dE=(hi["E"]-mid["E"])/h
        else:
            dE=(hi["E"]-lo["E"])/(2*h)
        tension_g=mid["eps"]*mid["phi"]
        direct_chain=tension_g*dg_de(f0,e)
        rows.append((e,g_of_e(f0,e),-dE,tension_g,dg_de(f0,e),direct_chain))
    # dg/de collapses with saturation; using Phi*eps directly as delta-e would discard this.
    assert rows[-1][4] < rows[0][4]*.001
    return rows


def make_convergence():
    f=NethraField(leakage=.6,convergence_gain=1.0)
    s1=f.new(); s2=f.new(); y=f.new()
    # fixed symmetric incidences
    def edges():
        return ((s1,y,.8),(s2,y,.8))
    f._edges=edges
    # Orthogonal residual-history fixture => independence 1.
    f.pair_stats[frozenset((s1,s2))]=(0.0,1.0,1.0,20)
    return f,s1,s2,y


def test_convergence_is_field_carried_but_missing_from_signed_Q_prediction():
    f,s1,s2,y=make_convergence()
    es,q,A,C=integrate(f,{s1:1.0,s2:1.0})
    conductive=incoming_q(es,q,y)
    conv=C[y]
    full_internal=conductive+conv
    assert conv>0.0
    assert full_internal>conductive
    return conductive,conv,full_internal,conv/full_internal


def main():
    print("q_equals_gphi",test_node_integral_reconstructs_edge_charge())
    print("full_gradient_vs_direct")
    for row in test_full_gradient_vs_frozen_trajectory_term():
        print(row)
    print("virtual_recruitment")
    for row in test_virtual_recruitment_is_local_approximation_not_exact_finite_change():
        print(row)
    print("evidence_chain_rule")
    for row in test_evidence_chain_rule():
        print(row)
    print("convergence_omission",test_convergence_is_field_carried_but_missing_from_signed_Q_prediction())
    print("all_assertions_passed")


if __name__=="__main__":
    main()
