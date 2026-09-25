"""Context after small factors: C1, C2 alone and X->Y, X->Z alone first, then C1 with X->Y and
C2 with X->Z.  Compares the current core with the per-part subtraction prototype.  Reads, on a
frozen copy after C+X: live activation and P toward Y and Z; lists what phase 2 built."""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nethra_presence as core, partwise_prototype as partwise
NAMES = ["C1", "C2", "X", "Y", "Z", "F1", "F2"]
C1, C2, X, Y, Z, F1, F2 = range(7)
def P_toward(g, reads):
    A = g.current_interval_integral; out = {t: 0.0 for t in reads}
    for (rel, mem), row in g._physical_incidences(g.current_event).items():
        if mem in out:
            q = row["g"] * (A.get(rel, 0.0) - A.get(mem, 0.0))
            if q > 0: out[mem] += q
    return out
def run(cls, small, seed=0):
    rng = random.Random(seed)
    f = cls(); L = [f.new() for _ in range(7)]
    def show(xs):
        for x in xs: L[x].push(1.0)
        f.step(1.0)
    if small:
        for _ in range(20):
            for c in rng.sample([C1, C2], 2):
                for _ in range(3): show([c])
                show([F1])
            for s in rng.sample([Y, Z], 2):
                show([X]); show([s]); show([F2])
    n_small = len(f.nethra)
    for _ in range(30):
        for c, s in rng.sample([(C1, Y), (C2, Z)], 2):
            show([c, X]); show([c, s]); show([c, F1]); show([F2])
    res = []
    for c, good, bad in ((C1, Y, Z), (C2, Z, Y)):
        g = cls.from_checkpoint_dict(f.checkpoint_dict()); g.topology_and_evidence_change = False
        M = g.nethra
        for _ in range(3): M[F2].push(1.0); g.step(1.0)
        M[c].push(1.0); M[X].push(1.0); g.step(1.0)
        P = P_toward(g, [M[good], M[bad]])
        res.append(f"{NAMES[c]}+X: act {NAMES[good]} {M[good].activation:.5f} {NAMES[bad]} {M[bad].activation:.5f} | P {NAMES[good]} {P[M[good]]:.5f} {NAMES[bad]} {P[M[bad]]:.5f}")
    idx = {n: i for i, n in enumerate(f.nethra)}
    nm = lambda n: NAMES[idx[n]] if idx[n] < 7 else f"N{idx[n]}"
    phase2 = [f"N{idx[n]}: " + " | ".join("{" + ",".join(sorted(map(nm, R))) + "}" for R in n.routes) for n in f.nethra[n_small:]]
    return res, phase2
for name, cls in (("current core", core.NethraField), ("per-part V1", partwise.ExactSourcePartwiseField)):
    for small in (True, False):
        res, phase2 = run(cls, small)
        print(f"== {name}, small factors first {small}"); print("  " + "\n  ".join(res)); print("  built in phase 2:", *phase2[:8], sep="\n    ")
