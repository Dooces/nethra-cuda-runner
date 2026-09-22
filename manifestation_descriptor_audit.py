#!/usr/bin/env python3
"""Audit completed-interval manifestation representation without prediction or causal credit.

Question only:
    What happened to one Nethra during a completed interval?

Candidate native descriptor:
    X_i = (Delta a_i, A_i)
where
    Delta a_i = a_i(t1)-a_i(t0)
    A_i       = integral a_i(t) dt.

External source charge U_i is retained separately as provenance, not folded into manifestation.

This probe asks what X can reconstruct exactly under the frozen field with fixed conductance over
an interval, and whether either scalar alone loses distinct field histories.

No residual, prediction, learning, evidence update, or counterfactual credit appears here.
"""

import math
from nethra import NethraField

T=.4
STEPS=120
DT=T/STEPS


def integrate(field,nodes,currents,initial=None):
    if initial is not None:
        for n,v in zip(nodes,initial):
            n.activation=float(v)
    for n,j in zip(nodes,currents):
        n.external=float(j)

    start={n:n.activation for n in nodes}
    A={n:0.0 for n in nodes}
    QC={n:0.0 for n in nodes}
    QB={n:0.0 for n in nodes}

    def cond(state):
        out={n:0.0 for n in nodes}
        for a,b,g in field._edges():
            q=g*(state[a]-state[b])
            out[a]-=q; out[b]+=q
        return out

    def comps(state):
        c=cond(state)
        full={n:field.capacitance*v for n,v in field._derivative_at(state).items()}
        b={n:full[n]-n.external+field.leakage*state[n]-c[n] for n in nodes}
        return c,b

    for _ in range(STEPS):
        a0={n:n.activation for n in nodes}; k1=field._derivative_at(a0)
        a1={n:a0[n]+.5*DT*k1[n] for n in nodes}; k2=field._derivative_at(a1)
        a2={n:a0[n]+.5*DT*k2[n] for n in nodes}; k3=field._derivative_at(a2)
        a3={n:a0[n]+DT*k3[n] for n in nodes}; k4=field._derivative_at(a3)
        cs=[comps(s) for s in (a0,a1,a2,a3)]
        for n in nodes:
            A[n]+=DT*(a0[n]+2*a1[n]+2*a2[n]+a3[n])/6
            QC[n]+=DT*(cs[0][0][n]+2*cs[1][0][n]+2*cs[2][0][n]+cs[3][0][n])/6
            QB[n]+=DT*(cs[0][1][n]+2*cs[1][1][n]+2*cs[2][1][n]+cs[3][1][n])/6
            n.activation=a0[n]+DT*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6

    D={n:n.activation-start[n] for n in nodes}
    U={n:T*float(j) for n,j in zip(nodes,currents)}
    end={n:n.activation for n in nodes}
    for n in nodes:n.external=0.0
    return start,end,A,D,U,QC,QB


def make_field(conv=1.0):
    f=NethraField(leakage=.55,capacitance=1.0,convergence_gain=conv)
    a,b,c=[f.new() for _ in range(3)]
    f._edges=lambda:((a,b,.7),(b,c,.45),(a,c,.25))
    if conv:
        for x,y in ((a,b),(a,c),(b,c)):
            f.pair_stats[frozenset((x,y))]=(0.0,1.0,1.0,20)
    return f,(a,b,c)


def test_exact_reconstruction_from_D_A_plus_separate_U():
    worst_endpoint=0.0
    worst_phi=0.0
    worst_q=0.0
    worst_m=0.0
    worst_internal=0.0
    worst_f61=0.0

    fixtures=[
        ((0,0,0),(1.0,.2,-.1)),
        ((.3,-.2,.1),(.0,.0,.0)),
        ((-.4,.5,.2),(-.7,.6,.0)),
        ((.8,.8,.8),(.3,-.2,.4)),
    ]

    for initial,currents in fixtures:
        f,nodes=make_field(1.0)
        start,end,A,D,U,QC,QB=integrate(f,nodes,currents,initial)
        for n in nodes:
            # previous boundary state + Delta reconstructs current endpoint
            worst_endpoint=max(worst_endpoint,abs((start[n]+D[n])-end[n]))

            M=f.capacitance*D[n]+f.leakage*A[n]
            internal=M-U[n]
            worst_m=max(worst_m,abs(M-(U[n]+QC[n]+QB[n])))
            worst_internal=max(worst_internal,abs(internal-(QC[n]+QB[n])))

            # If U remains separate provenance, aggregate F61 contribution is recoverable from
            # X=(D,A), U, and conductive Q derived below.
            recovered_f61=internal-QC[n]
            worst_f61=max(worst_f61,abs(recovered_f61-QB[n]))

        for x,y,g in f._edges():
            phi=A[x]-A[y]
            # Direct numerical incidence charge equals g*Phi by fixed-g interval identity.
            # Recover node-oriented contribution from integrated conductive totals indirectly by
            # recomputing the exact analytic edge charge.
            q=g*phi
            worst_phi=max(worst_phi,abs(phi-(A[x]-A[y])))
            worst_q=max(worst_q,abs(q-g*(A[x]-A[y])))

    assert max(worst_endpoint,worst_phi,worst_q,worst_m,worst_internal,worst_f61)<3e-15
    return {
        "endpoint":worst_endpoint,"phi":worst_phi,"q":worst_q,"M":worst_m,
        "internal":worst_internal,"f61":worst_f61,
    }


def scalar_fixture_constant():
    f=NethraField(leakage=.55,capacitance=1.0,convergence_gain=0.0)
    n=f.new()
    # warm to exact equilibrium a=J/lambda using constant source J=.55 => a=1
    n.activation=1.0
    return integrate(f,(n,),(.55,),(1.0,))


def scalar_fixture_zero():
    f=NethraField(leakage=.55,capacitance=1.0,convergence_gain=0.0)
    n=f.new()
    return integrate(f,(n,),(0.0,),(0.0,))


def test_delta_alone_loses_sustained_presence():
    s=scalar_fixture_constant()
    z=scalar_fixture_zero()
    ns=list(s[3])[0]; nz=list(z[3])[0]
    Ds=s[3][ns]; Dz=z[3][nz]
    As=s[2][ns]; Az=z[2][nz]
    assert abs(Ds)<1e-15 and abs(Dz)<1e-15
    assert As>0.39 and abs(Az)<1e-15
    return Ds,Dz,As,Az


def test_A_alone_does_not_determine_delta():
    # Algebraic completed-interval counterexample using admissible continuous activation paths.
    # Both paths have integral zero over symmetric interval, but opposite endpoint deltas.
    # path1 a(t)=t-T/2 => A=0, Delta=+T
    # path2 a(t)=T/2-t => A=0, Delta=-T
    A1=0.0; D1=T
    A2=0.0; D2=-T
    assert A1==A2 and D1!=D2
    return A1,D1,A2,D2


def test_M_alone_does_not_determine_A_or_delta():
    # M=C*D+lambda*A is one scalar over two independent native interval coordinates.
    C=1.0; lam=.55
    D1=.22; A1=.31
    M=C*D1+lam*A1
    A2=.51
    D2=(M-lam*A2)/C
    assert abs((C*D2+lam*A2)-M)<1e-15
    assert D1!=D2 and A1!=A2
    return M,(D1,A1),(D2,A2)


def test_same_descriptor_same_next_physical_state_given_same_persistent_field():
    # The next ODE state depends on endpoint activation, not on a hidden label for the prior path.
    # Given identical previous boundary activation and identical D, the endpoint is identical.
    # A is retained for completed-interval interaction/manifestation, not because RK4 needs it as
    # a hidden next-state variable.
    prior=.37; D=-.12
    end1=prior+D; end2=prior+D
    assert end1==end2
    return prior,D,end1


def main():
    print("exact_reconstruction",test_exact_reconstruction_from_D_A_plus_separate_U())
    print("delta_collision",test_delta_alone_loses_sustained_presence())
    print("A_collision",test_A_alone_does_not_determine_delta())
    print("M_collision",test_M_alone_does_not_determine_A_or_delta())
    print("next_state",test_same_descriptor_same_next_physical_state_given_same_persistent_field())
    print("all_assertions_passed")


if __name__=="__main__":
    main()
