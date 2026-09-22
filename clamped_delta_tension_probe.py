#!/usr/bin/env python3
"""Clamped-observation boundary probe for literal field prediction tension.

This tests a physical boundary model allowed by the old frozen V53 contract: externally observed
ordinary Nethra are held to the measured trajectory instead of receiving an additive semantic
"error" current.

For a clamped Nethra i with prescribed derivative v_i(t), the passive/free Nethra field would
produce derivative d_i^free.  The external reaction current required to enforce the observation is

    J_i^reaction = C * (v_i - d_i^free).

That is not a learned score.  It is the physical current the boundary must supply (positive) or
remove (negative) to keep the observed Nethra on its externally imposed trajectory.

The probe uses two finite intervals:
- context interval: context Nethra held at 1, future observation Nethra held at 0;
- consequence interval: future observation either remains at 0 or ramps 0 -> 1.

No probability ledger, target label, threshold, matcher, or conductance update is used.
"""

from nethra import NethraField

T=0.6
STEPS=120
DT=T/STEPS


def set_evidence(r,e):
    for c in r.routes.values():
        c.clear(); c[frozenset()]=float(e)


def add_relation(f,*members,evidence=1.0):
    r=f.new()
    f._route(r,members,frozenset(),1)
    set_evidence(r,evidence)
    return r


def stage_boundary(spec,t):
    """Return prescribed activation and derivative for each externally held ordinary Nethra."""
    out={}
    for n,(kind,a0,a1) in spec.items():
        if kind=="constant":
            out[n]=(float(a0),0.0)
        elif kind=="ramp":
            frac=max(0.0,min(1.0,t/T))
            out[n]=(float(a0)+(float(a1)-float(a0))*frac,(float(a1)-float(a0))/T)
        else:
            raise ValueError(kind)
    return out


def with_boundary(state,bound):
    s=dict(state)
    for n,(value,velocity) in bound.items():
        s[n]=value
    return s


def interval(field,spec):
    # No additive external current is used in this boundary model.
    for n in field.nethra: n.external=0.0

    reaction={n:0.0 for n in spec}
    for step in range(STEPS):
        t=step*DT
        b0=stage_boundary(spec,t)
        bh=stage_boundary(spec,t+.5*DT)
        b1=stage_boundary(spec,t+DT)

        a0=with_boundary({n:n.activation for n in field.nethra},b0)
        d0=field._derivative_at(a0)
        k1=dict(d0)
        for n,(value,v) in b0.items(): k1[n]=v

        a_half={n:a0[n]+.5*DT*k1[n] for n in field.nethra}
        a_half=with_boundary(a_half,bh)
        d1=field._derivative_at(a_half)
        k2=dict(d1)
        for n,(value,v) in bh.items(): k2[n]=v

        a_half2={n:a0[n]+.5*DT*k2[n] for n in field.nethra}
        a_half2=with_boundary(a_half2,bh)
        d2=field._derivative_at(a_half2)
        k3=dict(d2)
        for n,(value,v) in bh.items(): k3[n]=v

        a_end={n:a0[n]+DT*k3[n] for n in field.nethra}
        a_end=with_boundary(a_end,b1)
        d3=field._derivative_at(a_end)
        k4=dict(d3)
        for n,(value,v) in b1.items(): k4[n]=v

        # RK4 quadrature of the literal boundary reaction current C*(prescribed derivative -
        # free derivative) at the same four stage states.
        for n in spec:
            r0=field.capacitance*(b0[n][1]-d0[n])
            r1=field.capacitance*(bh[n][1]-d1[n])
            r2=field.capacitance*(bh[n][1]-d2[n])
            r3=field.capacitance*(b1[n][1]-d3[n])
            reaction[n]+=DT*(r0+2*r1+2*r2+r3)/6.0

        for n in field.nethra:
            n.activation=a0[n]+DT*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.0
        # Enforce exact endpoint boundary.
        for n,(value,v) in b1.items():
            n.activation=value

    return reaction


def build(evidence=1.0,predictors=1):
    f=NethraField(leakage=.6,convergence_gain=0.0)
    y=f.new()
    contexts=[]
    rels=[]
    for _ in range(predictors):
        a=f.new()
        r=add_relation(f,a,y,evidence=evidence)
        contexts.append(a); rels.append(r)
    return f,y,contexts,rels


def establish_context(f,y,contexts):
    spec={y:("constant",0.0,0.0)}
    for a in contexts: spec[a]=("constant",1.0,1.0)
    return interval(f,spec)


def consequence_reaction(evidence,present=True,predictors=1):
    f,y,contexts,rels=build(evidence,predictors)
    r0=establish_context(f,y,contexts)
    spec={}
    for a in contexts: spec[a]=("constant",1.0,1.0)
    spec[y]=("ramp",0.0,1.0) if present else ("constant",0.0,0.0)
    r1=interval(f,spec)
    return r0[y],r1[y],f,y,contexts,rels


def test_prediction_reduces_external_work_for_observed_transition():
    rows=[]
    # Isolated Y baseline.
    f=NethraField(leakage=.6,convergence_gain=0.0)
    y=f.new()
    base=interval(f,{y:("ramp",0.0,1.0)})[y]

    for e in (0.0,1.0,5.0,20.0,100.0,1000.0):
        pre,react,*_=consequence_reaction(e,True,1)
        rows.append((e,pre,react,base-react))

    # Stronger established context relation should require less positive reaction to realize Y.
    assert rows[-1][2] < rows[0][2]
    assert rows[-1][3] > rows[0][3]
    return base,rows


def test_false_prediction_requires_opposite_boundary_reaction():
    rows=[]
    for e in (0.0,1.0,5.0,20.0,100.0,1000.0):
        pre,react,*_=consequence_reaction(e,False,1)
        rows.append((e,pre,react))
    # While Y is physically held at zero, relation-driven prediction must be countered by
    # negative boundary reaction. Greater relation strength should increase that burden.
    assert all(x[1] < 0.0 for x in rows)
    assert all(x[2] < 0.0 for x in rows)
    assert abs(rows[-1][2]) > abs(rows[0][2])
    return rows


def test_combined_predictors_reduce_transition_reaction():
    rows=[]
    for count in (0,1,2,3):
        if count==0:
            f=NethraField(leakage=.6,convergence_gain=0.0)
            y=f.new()
            react=interval(f,{y:("ramp",0.0,1.0)})[y]
            rows.append((count,react))
        else:
            pre,react,*_=consequence_reaction(1.0,True,count)
            rows.append((count,react))
    assert rows[1][1] < rows[0][1]
    assert rows[2][1] < rows[1][1]
    assert rows[3][1] < rows[2][1]
    return rows


def test_wrong_receiver_not_helped():
    # A relation prepares Y. The world instead ramps unrelated Z.
    f=NethraField(leakage=.6,convergence_gain=0.0)
    y=f.new(); z=f.new(); a=f.new()
    r=add_relation(f,a,y,evidence=20.0)
    interval(f,{a:("constant",1.0,1.0),y:("constant",0.0,0.0),z:("constant",0.0,0.0)})
    wrong=interval(f,{a:("constant",1.0,1.0),y:("constant",0.0,0.0),z:("ramp",0.0,1.0)})

    cold=NethraField(leakage=.6,convergence_gain=0.0)
    zc=cold.new()
    cold_z=interval(cold,{zc:("ramp",0.0,1.0)})[zc]

    assert abs(wrong[z]-cold_z)<1e-12
    assert wrong[y]<0.0
    return wrong[y],wrong[z],cold_z


def test_reverse_order_uses_same_symmetric_relation():
    # Swap which member is context and which is future observation.
    f=NethraField(leakage=.6,convergence_gain=0.0)
    a=f.new(); y=f.new(); r=add_relation(f,a,y,evidence=20.0)
    interval(f,{y:("constant",1.0,1.0),a:("constant",0.0,0.0)})
    reverse=interval(f,{y:("constant",1.0,1.0),a:("ramp",0.0,1.0)})[a]

    f2=NethraField(leakage=.6,convergence_gain=0.0)
    a2=f2.new(); y2=f2.new(); r2=add_relation(f2,a2,y2,evidence=20.0)
    interval(f2,{a2:("constant",1.0,1.0),y2:("constant",0.0,0.0)})
    forward=interval(f2,{a2:("constant",1.0,1.0),y2:("ramp",0.0,1.0)})[y2]

    assert abs(reverse-forward)<1e-12
    return forward,reverse


def test_step_refinement():
    # Reimplement same protocol at several step counts by changing globals locally through a small
    # nested integrator would duplicate code. The main 120-step result is compared to a 60-step
    # run by temporarily changing module globals.
    global STEPS,DT
    original=(STEPS,DT)
    vals=[]
    try:
        for steps in (30,60,120,240):
            STEPS=steps; DT=T/STEPS
            pre,react,*_=consequence_reaction(20.0,True,1)
            vals.append((steps,pre,react))
    finally:
        STEPS,DT=original
    ref=vals[-1][2]
    assert max(abs(v[2]-ref) for v in vals)<1e-8
    return vals


def main():
    print("transition_work",test_prediction_reduces_external_work_for_observed_transition())
    print("false_prediction_reaction",test_false_prediction_requires_opposite_boundary_reaction())
    print("combined_predictors",test_combined_predictors_reduce_transition_reaction())
    print("wrong_receiver",test_wrong_receiver_not_helped())
    print("reverse_symmetric",test_reverse_order_uses_same_symmetric_relation())
    print("step_refinement",test_step_refinement())
    print("all_assertions_passed")


if __name__=="__main__":
    main()
