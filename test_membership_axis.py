from nethra import Field


def test_same_members_can_hold_distinct_modes():
    f = Field(recurrence=2)
    a, b = f.add(), f.add()
    f.consider({a: 1.0, b: 1.0})
    r1 = f.consider({a: 2.0, b: 2.0})
    f.consider({a: 1.0, b: 2.0})
    r2 = f.consider({a: 2.0, b: 4.0})
    assert r1 is not None and r2 is not None and r1 is not r2
    assert r1.members == r2.members == frozenset((a, b))
    assert r1.incidence != r2.incidence


def test_global_reversal_refinds_same_relation():
    f = Field(recurrence=2)
    a, b = f.add(), f.add()
    f.consider({a: 1.0, b: 2.0})
    r = f.consider({a: 2.0, b: 4.0})
    before = r.evidence
    assert f.consider({a: -3.0, b: -6.0}) is r
    assert r.evidence == before + 1


def test_mixed_sign_mode_is_not_forced_into_passive_topology():
    f = Field(recurrence=2)
    a, b = f.add(), f.add()
    f.consider({a: 1.0, b: -1.0})
    assert f.consider({a: 2.0, b: -2.0}) is None
    assert len(f.nethra) == 2


def test_pending_evidence_has_no_behavior():
    f = Field(recurrence=3)
    a, b = f.add(), f.add()
    f.consider({a: 1.0, b: 1.0})
    a.push(1.0)
    for _ in range(4):
        f.step()
    assert b.activation == 0.0


def test_constructed_relation_changes_behavior():
    f = Field(recurrence=2)
    a, b = f.add(), f.add()
    f.consider({a: 1.0, b: 1.0})
    r = f.consider({a: 2.0, b: 2.0})
    a.push(1.0)
    for _ in range(4):
        f.step()
    assert r is not None
    assert b.activation > 0.0


def test_arbitrary_arity_and_recursion():
    f = Field(recurrence=2)
    a, b, c = f.add(), f.add(), f.add()
    f.consider({a: 1.0, b: 2.0, c: 3.0})
    r = f.consider({a: -2.0, b: -4.0, c: -6.0})
    assert r is not None and len(r.members) == 3
    f.consider({r: 1.0, a: 1.0})
    q = f.consider({r: 2.0, a: 2.0})
    assert q is not None and r in q.members and a in q.members
