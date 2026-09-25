"""Symbolic focus test.  Nethra: periphery Pc[l][c] (4 locations x colour food/other), gaze G[l],
fovea V[c], F.  Trial: object at l colour c appears with gaze elsewhere; next interval the gaze is
at l and the fovea sees c (scripted saccade); held 3 intervals; food trials then push F; 2 silent.
Test on frozen copies: food at la and other at lb appear together, gaze at g0.  Read P toward
G[la] and G[lb] (and live activation)."""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nethra_presence as core
def build(with_F, seed, n_trials=80):
    rng = random.Random(seed)
    f = core.NethraField()
    Pc = [[f.new() for c in range(2)] for l in range(4)]; G = [f.new() for _ in range(4)]; V = [f.new(), f.new()]; F = f.new()
    def show(xs):
        for x in xs: x.push(1.0)
        f.step(1.0)
    for t in range(n_trials):
        l = rng.randrange(4); c = t % 2; g0 = rng.choice([x for x in range(4) if x != l])
        show([Pc[l][c], G[g0]])
        for _ in range(3): show([Pc[l][c], G[l], V[c]])
        if with_F and c == 0: show([F])
        f.step(1.0); f.step(1.0)
    return f, Pc, G, V, F
def P_toward(g, reads):
    A = g.current_interval_integral; out = {t: 0.0 for t in reads}
    for (rel, mem), row in g._physical_incidences(g.current_event).items():
        if mem in out:
            q = row["g"] * (A.get(rel, 0.0) - A.get(mem, 0.0))
            if q > 0: out[mem] += q
    return out
for with_F in (True, False):
    shares_P, shares_a = [], []
    for seed in range(6):
        f, Pc, G, V, F = build(with_F, seed)
        for la in range(4):
            for lb in range(4):
                if la == lb: continue
                for g0 in range(4):
                    if g0 in (la, lb): continue
                    g = core.NethraField.from_checkpoint_dict(f.checkpoint_dict()); g.topology_and_evidence_change = False
                    n = g.nethra; idx = f._order
                    for _ in range(3): g.step(1.0)
                    for x in (Pc[la][0], Pc[lb][1], G[g0]): n[idx[x]].push(1.0)
                    g.step(1.0)
                    P = P_toward(g, [n[idx[G[la]]], n[idx[G[lb]]]])
                    a, b_ = P[n[idx[G[la]]]], P[n[idx[G[lb]]]]
                    shares_P.append(a / (a + b_) if a + b_ > 0 else float("nan"))
                    A1, A2 = n[idx[G[la]]].activation, n[idx[G[lb]]].activation
                    shares_a.append(A1 / (A1 + A2))
        cons = len(f.nethra) - 15
    m = lambda v: sum(x for x in v if x == x) / max(1, sum(1 for x in v if x == x))
    print(f"with F {with_F}: share of gaze expectation at the food location: P {m(shares_P):.3f} (min {min(shares_P):.3f}), "
          f"activation {m(shares_a):.3f} (min {min(shares_a):.3f}); constructed ~{cons}")
