"""One stream of 500 numbers built from different cycles, one seed.  What does the field carry
toward the number that comes next?

Stream: the seed picks a cycle, repeats it 4 to 8 times, picks another cycle, and so on, until 500
numbers.  Cycles share some numbers (2, 3, 5, 6, 9), so after those the next number depends on which
cycle is running.  Each cycle repeats several times in a row, so its transitions are built before
anything larger (CLAUDE.md section 4).

Feeding: one Nethra per number (the numbers are cycle members, not graded magnitudes).  Interval t:
push 1.0 onto the Nethra of x[t], step(1.0).  Nothing else is pushed.

Read points (the "output" Nethra): the number Nethra themselves.  A separate Nethra that is never
pushed is never in a closure, so it never joins a route, has no incidence, and stays at zero
(g(0) = 0).  This script creates one such Nethra per number (UNPUSHED) and reports the largest
|activation| any of them reaches, so this stays checked rather than assumed.

What is read after step t, before x[t+1] is pushed (the expectation for interval t+1):
  act   live activation of every number Nethra.
  P     prior flow toward every number Nethra: P_m = sum over incidences of max(0, g (A_R - A_m)),
        with A = the activation integrals of interval t and g = the conductances step t+1 will use.
        This is the P that step t+1 subtracts from the manifestation M (nethra.py,
        _move_evidence_and_construct).  It is checked against the field every CHECK_EVERY intervals:
        M - P as the field computes it (captured from update_residuals) minus M measured on
        frozen copies (actual source vs zero source) must reproduce this P.
Shares are taken over the number Nethra other than x[t] (x[t] was just pushed).  Where a block
boundary repeats a number (x[t+1] == x[t]) there is no share; those intervals are left out.  "next" is the
share on x[t+1]; "prev" is the share on x[t-1], i.e. what is left over from the interval before.

Reference: the same stream into a field with topology_and_evidence_change=False.  It never builds
anything, so its activation is leftover decay only; the difference is what construction added.

Alignment checks every interval (the script stops if one fails):
  current_interval_source  == {x[t]: 1.0}      the interval that just completed is t
  previous_interval_source == {x[t-1]: 1.0}    the one before it is t-1
  previous_explicit        == {x[t]}           closure reads use interval t
Closure after step t is read under current topology: f.closure(f.previous_explicit,
f.current_source_event), because f.previous_closure was computed before step t's construction.

usage: python3 cycle_stream.py [seed] [per-interval tsv path]"""
import sys, random
from collections import defaultdict
import nethra_presence as core

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
TSV = sys.argv[2] if len(sys.argv) > 2 else None
LENGTH, NUMBERS, CHECK_EVERY = 500, 10, 25
CYCLES = {"c3": (1, 2, 3), "c4": (4, 5, 6, 7), "c5": (8, 9, 0, 2, 5), "c3b": (3, 6, 9)}


def build_stream(rng):
    xs, tags, last = [], [], None
    while len(xs) < LENGTH:
        name = rng.choice([c for c in CYCLES if c != last]); last = name
        for _ in range(rng.randint(4, 8)):
            xs += CYCLES[name]; tags += [name] * len(CYCLES[name])
    return xs[:LENGTH], tags[:LENGTH]


def prior_flow(f):
    """P toward each Nethra for the NEXT step, computed exactly as nethra.py does it."""
    physical = f._physical_incidences(f.current_event)
    A = f.current_interval_integral
    P = defaultdict(float)
    for (relation, member), row in physical.items():
        g = float(row["g"])
        if g > 0.0:
            q = g * (float(A.get(relation, 0.0)) - float(A.get(member, 0.0)))
            if q > 0.0:
                P[member] += q
    return P


def measured_prior_flow(f, push_nethra_index):
    """P for the next step as the field itself uses it: (M - P) from update_residuals, M from copies."""
    captured = {}
    def frozen_copy():
        g = core.NethraField.from_checkpoint_dict(f.checkpoint_dict()); g.topology_and_evidence_change = False
        return g
    live = core.NethraField.from_checkpoint_dict(f.checkpoint_dict())
    original = live.update_residuals
    def capture(residual, neighbors=None):
        captured.update({live._order[n]: v for n, v in residual.items()})
        return original(residual, neighbors)
    live.update_residuals = capture
    live.nethra[push_nethra_index].push(1.0); live.step(1.0)
    with_source, zero_source = frozen_copy(), frozen_copy()
    with_source.nethra[push_nethra_index].push(1.0); with_source.step(1.0); zero_source.step(1.0)
    M = [max(0.0, a.activation - z.activation) for a, z in zip(with_source.nethra, zero_source.nethra)]
    return {i: M[i] - captured.get(i, 0.0) for i in range(len(M))}


def shares(values, x_now, x_next, x_prev):
    if x_next == x_now:                                   # next is the number just pushed: no share
        return float("nan"), float("nan")
    pool = {v: max(0.0, values[v]) for v in range(NUMBERS) if v != x_now}
    total = sum(pool.values())
    nxt = pool[x_next] / total if total > 0 else 0.0
    prv = pool[x_prev] / total if total > 0 and x_prev is not None and x_prev != x_now else 0.0
    return nxt, prv


def run(xs, build):
    f = core.NethraField(topology_and_evidence_change=build)
    num = [f.new() for _ in range(NUMBERS)]               # pushed; also the read points
    unpushed = [f.new() for _ in range(NUMBERS)]          # never pushed
    rows, worst_unpushed, worst_p_check = [], 0.0, 0.0
    for t, x in enumerate(xs):
        before = len(f.nethra)
        num[x].push(1.0); f.step(1.0)
        # alignment: the field's own record says the completed interval is t, the one before t-1
        assert f.current_interval_source == {num[x]: 1.0}, t
        assert t == 0 or f.previous_interval_source == {num[xs[t - 1]]: 1.0}, t
        assert f.previous_explicit == frozenset((num[x],)), t
        worst_unpushed = max(worst_unpushed, max(abs(n.activation) for n in unpushed))
        if t + 1 == len(xs):
            break
        x_next, x_prev = xs[t + 1], (xs[t - 1] if t > 0 else None)
        act = [n.activation for n in num]
        P = prior_flow(f)
        pv = [P.get(n, 0.0) for n in num]
        if build and t % CHECK_EVERY == 0:
            field_P = measured_prior_flow(f, x_next)
            worst_p_check = max(worst_p_check, max(abs(field_P[i] - pv[i]) for i in range(NUMBERS)))
        closed = f.closure(f.previous_explicit, f.current_source_event)
        rows.append(dict(t=t, x=x, next=x_next, new=len(f.nethra) - before, nethra=len(f.nethra),
                         refound=sum(1 for n in closed if n.routes),
                         act=shares(act, x, x_next, x_prev), P=shares(pv, x, x_next, x_prev),
                         act_next=act[x_next], P_next=pv[x_next]))
    return f, rows, worst_unpushed, worst_p_check


def mean(v):
    v = [x for x in v if x == x]; return sum(v) / len(v) if v else float("nan")


if __name__ == "__main__":
    rng = random.Random(SEED)
    xs, tags = build_stream(rng)
    succ = defaultdict(set)
    for a, b in zip(xs, xs[1:]):
        succ[a].add(b)
    transitions = {(a, b) for a, b in zip(xs, xs[1:])}
    blocks = [(tags[0], 0)]
    for i in range(1, LENGTH):
        if tags[i] != tags[i - 1]: blocks.append((tags[i], i))
    print(f"seed {SEED}, {LENGTH} numbers, cycles {CYCLES}")
    print("blocks (cycle@start):", " ".join(f"{c}@{s}" for c, s in blocks))
    print(f"distinct transitions in the stream: {len(transitions)}  "
          f"(prediction from the code: one constructed Nethra per distinct transition)")
    print("numbers with more than one successor in this stream:",
          {a: sorted(b) for a, b in sorted(succ.items()) if len(b) > 1})

    f, rows, unp, pchk = run(xs, True)
    g, ref, unp_ref, _ = run(xs, False)
    print(f"\nconstructed Nethra: {len(f.nethra) - 2 * NUMBERS}   (reference field: {len(g.nethra) - 2 * NUMBERS})")
    print(f"never-pushed Nethra, largest |activation| over the run: {unp:.3g} (reference {unp_ref:.3g})")
    print(f"P check against the field's own M - P, largest difference: {pchk:.3g}")
    print("alignment checks: passed on every interval")
    print(f"intervals where x[t+1] == x[t] (left out of shares): {sum(1 for r in rows if r['next'] == r['x'])}")

    def report(label, pick):
        sel = [i for i, r in enumerate(rows) if pick(r)]
        if not sel: return
        a = mean(rows[i]["act"][0] for i in sel); ap = mean(rows[i]["act"][1] for i in sel)
        p = mean(rows[i]["P"][0] for i in sel)
        ra = mean(ref[i]["act"][0] for i in sel); rp = mean(ref[i]["act"][1] for i in sel)
        print(f"  {label:34s} n={len(sel):3d}  act next {a:.3f} (ref {ra:.3f})  "
              f"act prev {ap:.3f} (ref {rp:.3f})  P next {p:.3f}")

    print("\nshares over the other 9 number Nethra after step t (chance would be 0.111):")
    report("all intervals", lambda r: True)
    report("next determined by current number", lambda r: len(succ[r["x"]]) == 1)
    report("next depends on cycle", lambda r: len(succ[r["x"]]) > 1)
    for name in CYCLES:
        report(f"inside {name}", lambda r, name=name: tags[r["t"]] == name and tags[r["t"] + 1] == name)
    for lo, hi in ((0, 100), (100, 200), (200, 300), (300, 400), (400, 499)):
        report(f"t {lo}-{hi}", lambda r, lo=lo, hi=hi: lo <= r["t"] < hi)

    print("\nper number, next depends on cycle: share of each successor after that number")
    for a, bs in sorted(succ.items()):
        if len(bs) < 2: continue
        for b in sorted(bs):
            sel = [i for i, r in enumerate(rows) if r["x"] == a and r["next"] == b]
            print(f"  {a} -> {b}: n={len(sel):3d}  act share {mean(rows[i]['act'][0] for i in sel):.3f} "
                  f"(ref {mean(ref[i]['act'][0] for i in sel):.3f})  P share {mean(rows[i]['P'][0] for i in sel):.3f}")

    new_at = [r["t"] for r in rows if r["new"]]
    print(f"\nintervals where construction added Nethra: {len(new_at)}; last at t={max(new_at) if new_at else None}")
    print(f"constructed Nethra refound after step t (mean): {mean(r['refound'] for r in rows):.1f}")

    if TSV:
        with open(TSV, "w") as out:
            out.write("t\tcycle\tx\tnext\tnew\tnethra\trefound\tact_next\tact_share_next\tact_share_prev"
                      "\tP_next\tP_share_next\tref_act_share_next\tref_act_share_prev\n")
            for r, q in zip(rows, ref):
                out.write(f"{r['t']}\t{tags[r['t']]}\t{r['x']}\t{r['next']}\t{r['new']}\t{r['nethra']}\t{r['refound']}"
                          f"\t{r['act_next']:.6g}\t{r['act'][0]:.4f}\t{r['act'][1]:.4f}\t{r['P_next']:.6g}"
                          f"\t{r['P'][0]:.4f}\t{q['act'][0]:.4f}\t{q['act'][1]:.4f}\n")
        print(f"per-interval rows written to {TSV}")
