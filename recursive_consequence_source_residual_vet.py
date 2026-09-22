#!/usr/bin/env python3
"""Test whether future external source charge U can be the universal Nethra consequence.

Fixture:
- P is an ordinary predictor relation driven by context A and connected to future relation Y.
- Y is itself an ordinary Nethra relation over primitive q1/q2.
- During the future interval q1/q2 are externally sourced; Y has ZERO external source.
- Y nevertheless acquires positive activation change through the Nethra field.

If prior field flow P->Y is positive and next Y delta is positive, a universal prospective law
should be able to represent that recurrence. The proposed external-source residual
epsilon_Y = U_Y(next)-prediction_Y(prior) has U_Y=0 and therefore assigns a negative residual.

No learning update is installed.
"""

from nethra import NethraField

T=.5
STEPS=80
DT=T/STEPS


def integrate(field,currents):
    for n in field.nethra:
        n.activation=0.0
        n.external=0.0
    for n,j in currents.items(): n.external=float(j)

    a_start={n:n.activation for n in field.nethra}
    A={n:0.0 for n in field.nethra}

    for _ in range(STEPS):
        a0={n:n.activation for n in field.nethra}
        k1=field._derivative_at(a0)
        a1={n:a0[n]+.5*DT*k1[n] for n in field.nethra}; k2=field._derivative_at(a1)
        a2={n:a0[n]+.5*DT*k2[n] for n in field.nethra}; k3=field._derivative_at(a2)
        a3={n:a0[n]+DT*k3[n] for n in field.nethra}; k4=field._derivative_at(a3)
        for n in field.nethra:
            A[n]+=DT*(a0[n]+2*a1[n]+2*a2[n]+a3[n])/6
            n.activation=a0[n]+DT*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6

    delta={n:n.activation-a_start[n] for n in field.nethra}
    U={n:T*float(currents.get(n,0.0)) for n in field.nethra}
    for n in field.nethra: n.external=0.0
    return A,delta,U


def main():
    f=NethraField(leakage=.6,convergence_gain=0.0)

    a=f.new()
    q1=f.new(); q2=f.new()
    y=f.new()      # future relation Nethra
    p=f.new()      # predictor relation Nethra

    # Y is relation over q1/q2. P links context A to Y.
    edges=((y,q1,.9),(y,q2,.9),(p,a,.9),(p,y,.45))
    f._edges=lambda: edges

    A0,d0,U0=integrate(f,{a:1.0})
    prior_prediction=.45*(A0[p]-A0[y])

    # New physical interval; primitive q1/q2 arrive. Y itself is not externally sourced.
    A1,d1,U1=integrate(f,{q1:1.0,q2:1.0})

    source_residual=U1[y]-prior_prediction
    manifestation_charge=f.capacitance*d1[y]
    manifestation_residual=manifestation_charge-prior_prediction

    print("prior_prediction_charge",prior_prediction)
    print("future_external_source_Y",U1[y])
    print("future_delta_Y",d1[y])
    print("future_manifestation_charge_Y",manifestation_charge)
    print("source_residual",source_residual)
    print("manifestation_residual",manifestation_residual)

    assert prior_prediction>0.0
    assert U1[y]==0.0
    assert d1[y]>0.0
    assert source_residual<0.0
    # This is the contradiction under test: source-only residual cannot represent that Y did
    # manifest positively through the field.
    print("source_only_consequence_is_not_universal")
    print("all_assertions_passed")


if __name__=="__main__":
    main()
