#!/usr/bin/env python3
import json, random
from nethra import NethraField

TRAIN_N = 4000
TEST_N = 12000
P_VALUES = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)
FIELD_STEPS = 20
FIELD_DT = 0.05

def independent_episode(field, context, outcome):
    field.previous_closure = frozenset()
    field.previous_event = frozenset()
    field.current_event = frozenset()
    field.observe(tuple(context))
    field.observe(tuple(context) + (outcome,))

def field_margin(field, context, b, d):
    for n in field.nethra:
        n.activation = 0.0
        n.external = 0.0
    for n in context:
        n.external = 1.0
    field.current_event = frozenset((n, +1) for n in context)
    for _ in range(FIELD_STEPS):
        a0 = {n:n.activation for n in field.nethra}
        k1 = field._derivative_at(a0)
        a1 = {n:a0[n]+0.5*FIELD_DT*k1[n] for n in field.nethra}
        k2 = field._derivative_at(a1)
        a2 = {n:a0[n]+0.5*FIELD_DT*k2[n] for n in field.nethra}
        k3 = field._derivative_at(a2)
        a3 = {n:a0[n]+FIELD_DT*k3[n] for n in field.nethra}
        k4 = field._derivative_at(a3)
        for n in field.nethra:
            n.activation = a0[n] + FIELD_DT*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.0
    for n in context:
        n.external = 0.0
    return b.activation - d.activation

def make_rows(p, n, seed):
    rng = random.Random(seed)
    rows = []
    # balanced cue present/absent by construction, randomized order
    for i in range(n):
        cue = bool(i & 1)
        pb = p if cue else (1.0-p)
        b = rng.random() < pb
        rows.append((cue, b))
    rng.shuffle(rows)
    return rows

def train_model(p, n, seed):
    f = NethraField()
    a = f.new()
    c = f.new()
    b = f.new()
    d = f.new()
    for cue, is_b in make_rows(p, n, seed):
        context = (a,c) if cue else (a,)
        independent_episode(f, context, b if is_b else d)
    return f,a,c,b,d

def score_model(f,a,c,b,d,p,seed):
    m_cue = field_margin(f,(a,c),b,d)
    m_no = field_margin(f,(a,),b,d)
    rows = make_rows(p, TEST_N, seed)
    correct = 0.0
    for cue,is_b in rows:
        margin = m_cue if cue else m_no
        if abs(margin) <= 1e-15:
            correct += 0.5
        else:
            pred_b = margin > 0
            correct += float(pred_b == is_b)
    acc = correct / len(rows)
    optimum = max(p,1.0-p)
    return {
        "p":p,
        "accuracy":acc,
        "bayes_optimum":optimum,
        "gap":optimum-acc,
        "cue_margin":m_cue,
        "no_cue_margin":m_no,
        "relations":len(f.nethra)-4,
        "history_relations":len(f.history_relation),
        "history_rows":len(f.history_count),
    }

def cue_sweep():
    out=[]
    for ix,p in enumerate(P_VALUES):
        f,a,c,b,d = train_model(p, TRAIN_N, 91000+ix)
        row = score_model(f,a,c,b,d,p,191000+ix)
        out.append(row)
        print("cue_sweep", json.dumps(row, sort_keys=True))
    return out

def perfect_confirmation_20():
    f=NethraField()
    a=f.new(); c=f.new(); b=f.new(); d=f.new()
    rows=[]
    first_both_correct=None
    for cycle in range(1,21):
        independent_episode(f,(a,c),b)
        independent_episode(f,(a,),d)
        mc=field_margin(f,(a,c),b,d)
        mn=field_margin(f,(a,),b,d)
        cue_correct = mc > 0
        no_correct = mn < 0
        if first_both_correct is None and cue_correct and no_correct:
            first_both_correct=cycle
        rows.append({
            "cycle":cycle,
            "cue_margin":mc,
            "no_cue_margin":mn,
            "both_correct":bool(cue_correct and no_correct),
            "relations":len(f.nethra)-4,
            "histories":len(f.history_relation),
        })
    print("confirmation_curve", json.dumps(rows, sort_keys=True))
    final=rows[-1]
    return {
        "first_both_correct":first_both_correct,
        "cycle20":final,
        "cycle20_accuracy": 1.0 if final["both_correct"] else 0.5,
    }

def main():
    sweep=cue_sweep()
    confirm=perfect_confirmation_20()
    print("confirmation20",json.dumps(confirm,sort_keys=True))
    # Claims under test:
    # 1) learned field should track the attainable statistical accuracy closely.
    # 2) perfect differentiated cue should be expressed correctly by cycle 20.
    max_gap=max(abs(r["gap"]) for r in sweep)
    print("max_optimality_gap",max_gap)
    assert max_gap <= 0.04, sweep
    assert confirm["cycle20"]["both_correct"], confirm
    print("VERIFY_PASS")

if __name__=="__main__":
    main()
