#!/usr/bin/env python3
"""Probe the proposed Nethra-native completed-interval boundary.

The boundary is deliberately non-learning:
    source_current: exact external current by Nethra for the completed interval
    delta:          exact activation change by Nethra over the completed interval

No sign projection, membership projection, equality test, recurrence test, or construction
permission occurs here.
"""

from nethra import NethraField


def integrate_capture(field, currents, dt=0.1):
    for n in field.nethra:
        n.external=0.0
    for n,j in currents.items():
        n.external=float(j)

    source={n:n.external for n in field.nethra if n.external != 0.0}
    a0={n:n.activation for n in field.nethra}

    k1=field._derivative_at(a0)
    a1={n:a0[n]+0.5*dt*k1[n] for n in field.nethra}
    k2=field._derivative_at(a1)
    a2={n:a0[n]+0.5*dt*k2[n] for n in field.nethra}
    k3=field._derivative_at(a2)
    a3={n:a0[n]+dt*k3[n] for n in field.nethra}
    k4=field._derivative_at(a3)

    delta={}
    for n in field.nethra:
        old=a0[n]
        n.activation=old+dt*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.0
        d=n.activation-old
        if d != 0.0:
            delta[n]=d

    for n in field.nethra:
        n.external=0.0
    return source,delta,a0,{n:n.activation for n in field.nethra}


def test_magnitude_is_preserved_without_identity_multiplication():
    def once(j):
        f=NethraField()
        a=f.new()
        source,delta,a0,a1=integrate_capture(f,{a:j})
        return list(source.values())[0],delta[a],len(source),len(f.nethra)

    lo=once(0.1)
    hi=once(0.9)
    assert lo[0] == 0.1 and hi[0] == 0.9
    assert hi[1] > lo[1] > 0.0
    assert lo[2:] == hi[2:] == (1,1)
    return lo,hi


def test_internal_field_change_is_not_relabelled_external_source():
    f=NethraField()
    a=f.new(); b=f.new(); r=f.new()
    f._route(r,(a,b),frozenset(),40)

    source,delta,a0,a1=integrate_capture(f,{a:1.0})

    assert set(source)=={a}
    assert a in delta
    assert r in delta
    assert b in delta
    assert r not in source and b not in source
    return len(source),len(delta),delta[a],delta[r],delta[b]


def test_before_state_reconstructs_exactly_from_after_minus_delta():
    f=NethraField()
    a=f.new(); b=f.new(); r=f.new()
    f._route(r,(a,b),frozenset(),20)
    # Seed a nonzero field so this is not a zero-origin special case.
    a.activation=.31; b.activation=.07; r.activation=.19

    source,delta,a0,a1=integrate_capture(f,{a:.37,b:-.11})
    worst=0.0
    for n in f.nethra:
        reconstructed=a1[n]-delta.get(n,0.0)
        worst=max(worst,abs(reconstructed-a0[n]))
    assert worst < 1e-15
    return worst


def test_same_source_ids_can_carry_different_continuous_instantiations():
    f1=NethraField(); a1=f1.new(); b1=f1.new()
    f2=NethraField(); a2=f2.new(); b2=f2.new()

    s1,d1,_,_=integrate_capture(f1,{a1:.2,b1:.8})
    s2,d2,_,_=integrate_capture(f2,{a2:.8,b2:.2})

    assert len(s1)==len(s2)==2
    vals1=tuple(sorted(s1.values()))
    vals2=tuple(sorted(s2.values()))
    # Same magnitudes as a multiset but attached oppositely; preserve attachment.
    assert vals1==vals2==(0.2,0.8)
    assert s1[a1] != s2[a2]
    assert d1[a1] != d2[a2]
    return (s1[a1],s1[b1],d1[a1],d1[b1]),(s2[a2],s2[b2],d2[a2],d2[b2])



def test_shadow_boundary_is_filled_by_real_step():
    f=NethraField()
    a=f.new(); b=f.new(); r=f.new()
    f._route(r,(a,b),frozenset(),20)

    a.push(.37)
    before={n:n.activation for n in f.nethra}
    returned=f.step(.1)

    assert set(f.current_interval_source)=={a}
    assert f.current_interval_source[a] == .37
    assert set(f.current_interval_delta)==set(n for n,v in returned.items() if v != 0.0)
    for n,v in f.current_interval_delta.items():
        assert v == returned[n]
    worst=max(abs((n.activation-f.current_interval_delta.get(n,0.0))-before[n]) for n in f.nethra)
    assert worst < 1e-15
    return len(f.current_interval_source),len(f.current_interval_delta),worst


def main():
    print("magnitude_preserved",test_magnitude_is_preserved_without_identity_multiplication())
    print("internal_vs_external",test_internal_field_change_is_not_relabelled_external_source())
    print("reconstruction_worst_error",test_before_state_reconstructs_exactly_from_after_minus_delta())
    print("same_ids_different_instantiation",test_same_source_ids_can_carry_different_continuous_instantiations())
    print("real_step_shadow_boundary",test_shadow_boundary_is_filled_by_real_step())
    print("all_assertions_passed")


if __name__=="__main__":
    main()
