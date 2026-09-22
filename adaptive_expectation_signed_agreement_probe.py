#!/usr/bin/env python3
"""Minimal adaptive-expectation loop.

Purpose:
- no probability table;
- no marginal baseline;
- no counterfactual gradient;
- no autonomous replay of historical delay;
- no privileged causal relation.

R is an already-earned ordinary Nethra representing a refound context such as AA.
Each observed continuation gets its own ordinary relation P_x connecting R to continuation x.
The context incidence R<->P_x is structural support and remains earned.
The continuation incidence P_x<->x carries empirical success/failure evidence.

On each recurrence of R:
    observed continuation: e_x += 1
    other already-known continuations: e_y = max(0, e_y - 1)

A never-before-seen continuation creates a new weak ordinary relation with evidence 1.

Expectation is read only from the Nethra field:
    source R;
    source no continuation;
    integrate;
    compare continuation activations.

This tests whether signed empirical agreement stored as per-incidence evidence is sufficient to
produce adaptive resonance in the intended examples.
"""

from nethra import NethraField

DT=.01
EXPECT_SECONDS=.75
SUPPORT_EVIDENCE=600.0


class AdaptiveExpectation:
    def __init__(self):
        self.f=NethraField(leakage=1.0,capacitance=1.0,convergence_gain=0.0)
        self.R=self.f.new()
        self.continuations={}
        self.relations={}
        self.evidence={}

    def add_continuation(self,name):
        if name in self.continuations:
            return
        x=self.f.new()
        p=self.f.new()
        self.continuations[name]=x
        self.relations[name]=p
        self.evidence[name]=0.0

    def observe(self,name):
        if name not in self.continuations:
            self.add_continuation(name)
        for key in list(self.evidence):
            if key==name:
                self.evidence[key]+=1.0
            else:
                self.evidence[key]=max(0.0,self.evidence[key]-1.0)

    def edges(self):
        rows=[]
        gs=self.f.conductance(SUPPORT_EVIDENCE)
        for name,p in self.relations.items():
            rows.append((self.R,p,gs))
            rows.append((p,self.continuations[name],self.f.conductance(self.evidence[name])))
        return tuple(rows)

    def expectation(self):
        self.f._edges=self.edges
        for n in self.f.nethra:
            n.activation=0.0
            n.external=0.0
        self.R.external=1.0
        steps=round(EXPECT_SECONDS/DT)
        for _ in range(steps):
            a0={n:n.activation for n in self.f.nethra}
            k1=self.f._derivative_at(a0)
            a1={n:a0[n]+.5*DT*k1[n] for n in self.f.nethra}; k2=self.f._derivative_at(a1)
            a2={n:a0[n]+.5*DT*k2[n] for n in self.f.nethra}; k3=self.f._derivative_at(a2)
            a3={n:a0[n]+DT*k3[n] for n in self.f.nethra}; k4=self.f._derivative_at(a3)
            for n in self.f.nethra:
                n.activation=a0[n]+DT*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.0
        self.R.external=0.0
        return {name:self.continuations[name].activation for name in self.continuations}

    def state(self):
        return {
            name:{
                "evidence":self.evidence[name],
                "g":self.f.conductance(self.evidence[name]),
                "activation":self.expectation()[name],
            }
            for name in self.continuations
        }


def snapshot(m,label):
    exp=m.expectation()
    print(label,{
        "evidence":dict(m.evidence),
        "conductance":{k:m.f.conductance(v) for k,v in m.evidence.items()},
        "expectation":exp,
    })
    return exp


def test_repeated_success_builds_expectation():
    m=AdaptiveExpectation()
    before=snapshot(m,"initial")
    vals=[]
    for i in range(120):
        m.observe("B")
        if i in (0,1,4,19,59,119):
            vals.append((i+1,m.evidence["B"],m.expectation()["B"]))
    print("B_learning_curve",vals)
    assert all(vals[i][2] < vals[i+1][2] for i in range(len(vals)-1))
    return m,vals


def test_single_surprise_does_not_erase_reliable_expectation():
    m,_=test_repeated_success_builds_expectation()
    pre=m.expectation()["B"]
    m.observe("D")
    post=m.expectation()
    print("single_D",{
        "B_before":pre,
        "B_after":post["B"],
        "D_after":post["D"],
        "evidence":dict(m.evidence),
    })
    assert post["B"]>post["D"]
    assert m.evidence["B"]==119.0
    assert m.evidence["D"]==1.0
    return m


def test_alternation_produces_ambiguity_not_forced_winner():
    m=AdaptiveExpectation()
    # establish both from scratch under symmetric alternating recurrence
    for _ in range(80):
        m.observe("B")
        m.observe("D")
    # After an even number of perfectly alternating outcomes, D has one more net unit only because
    # it was observed last; both remain near the same empirical support.
    exp=m.expectation()
    print("alternating",{"evidence":dict(m.evidence),"expectation":exp})
    assert abs(m.evidence["B"]-m.evidence["D"])<=1.0
    assert abs(exp["B"]-exp["D"])<.002
    return m


def test_persistent_reversal_moves_expectation():
    m,_=test_repeated_success_builds_expectation()
    # Introduce D once, then sustained reversal.
    m.observe("D")
    crossing=None
    rows=[]
    for i in range(1,151):
        m.observe("D")
        if i in (1,10,30,60,90,120,150):
            e=m.expectation()
            rows.append((i,dict(m.evidence),e))
        e=m.expectation()
        if crossing is None and e["D"]>e["B"]:
            crossing=i
    print("reversal_curve",rows)
    print("reversal_crossing",crossing)
    assert crossing is not None
    final=m.expectation()
    assert final["D"]>final["B"]
    assert m.evidence["B"]==0.0
    return m,crossing


def test_field_is_the_expectation_readout():
    m,_=test_repeated_success_builds_expectation()
    exp=m.expectation()
    # no B source was supplied; B activation exists only because R resonated through learned structure.
    print("field_expectation_without_B_source",exp["B"])
    assert exp["B"]>0.0
    return exp["B"]


def main():
    test_single_surprise_does_not_erase_reliable_expectation()
    test_alternation_produces_ambiguity_not_forced_winner()
    test_persistent_reversal_moves_expectation()
    test_field_is_the_expectation_readout()
    print("all_assertions_passed")


if __name__=="__main__":
    main()
