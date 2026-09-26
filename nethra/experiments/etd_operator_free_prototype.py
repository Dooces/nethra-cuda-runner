"""Prototype (not in core): ETD operators applied as Q (f * (Q^T x)) instead of 7 formed N x N matrices."""
import sys, json, random, itertools, time
import numpy as np
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..")); from nethra import NethraField
class Fast(NethraField):
    def _etd_prepare(self, compiled, h):
        key = ("etdq", h)
        if key in compiled: return compiled[key]
        N = compiled["N"]; Aop = np.zeros((N, N)); ei, ej, eg = compiled["ei"], compiled["ej"], compiled["eg"]
        np.add.at(Aop, (ei, ej), eg); np.add.at(Aop, (ej, ei), eg); np.add.at(Aop, (ei, ei), -eg); np.add.at(Aop, (ej, ej), -eg)
        Aop[np.diag_indices(N)] -= self.leakage; Aop /= self.capacitance
        mu, Q = np.linalg.eigh(Aop)
        q1, _q2, _q3 = self._phi(h * mu / 2); p1, p2, p3 = self._phi(h * mu)
        d = dict(A=Aop, Q=Q, Ainv=1.0 / mu, E=np.exp(h * mu), E2=np.exp(h * mu / 2), Q2=q1,
                 f1=p1 - 3 * p2 + 4 * p3, f2=2 * (p2 - 2 * p3), f3=4 * p3 - p2)
        compiled[key] = d; return d
    def _etd_interval(self, initial, dt, interval_nodes, compiled):
        pieces = max(1, self.etd_pieces); h = float(dt) / pieces; d = self._etd_prepare(compiled, h)
        Q = d["Q"]; ap = lambda f, x: Q @ (f * (Q.T @ x))
        ext = np.array([n.external for n in interval_nodes], dtype=float)
        a = np.array([initial[n] for n in interval_nodes], dtype=float); area = np.zeros(len(interval_nodes))
        A = d["A"]; D = self._derivative_compiled; Nl = lambda x: D(compiled, x, ext) - A @ x
        for _ in range(pieces):
            Na = Nl(a); a2 = ap(d["E2"], a) + (h / 2) * ap(d["Q2"], Na); Nb = Nl(a2)
            b2 = ap(d["E2"], a) + (h / 2) * ap(d["Q2"], Nb); Nc = Nl(b2)
            c = ap(d["E2"], a2) + (h / 2) * ap(d["Q2"], 2 * Nc - Na); Nd = Nl(c)
            nxt = ap(d["E"], a) + h * (ap(d["f1"], Na) + ap(d["f2"], Nb + Nc) + ap(d["f3"], Nd))
            intN = h * (Na + 2 * Nb + 2 * Nc + Nd) / 6.0; area += ap(d["Ainv"], nxt - a - intN); a = nxt
        return ({n: float(v) for n, v in zip(interval_nodes, a)}, {n: float(v) for n, v in zip(interval_nodes, area)})
size, nb = sys.argv[1], int(sys.argv[2])
d = json.load(open(f"big{size}.json"))
pairs = list(itertools.combinations(range(8), 2)); random.Random(7).shuffle(pairs)
out = {}
for cls in (NethraField, Fast):
    f = cls.from_checkpoint_dict(d["ck"]); N = {int(k): f.nethra[v] for k, v in d["N"].items()}; r2 = random.Random(3)
    t = time.perf_counter()
    for _ in range(nb):
        c = r2.randrange(28)
        for s in (1000, 1001, 2000 + c):
            for x in list(pairs[c]) + [s]: N[x].push(1.0)
            f.step(1.0)
    out[cls.__name__] = (time.perf_counter() - t, np.array([n.activation for n in f.nethra]), len(f.nethra),
                         sum(len(v) for v in f.incidence_evidence.values()))
a, b = out["NethraField"], out["Fast"]
m = min(len(a[1]), len(b[1]))
print(f"{size}: default {1000*a[0]/(3*nb):.1f} ms/int, prototype {1000*b[0]/(3*nb):.1f} ms/int; after {3*nb} intervals: "
      f"Nethra {a[2]}/{b[2]}, evidence keys {a[3]}/{b[3]}, max |activation diff| {np.abs(a[1][:m]-b[1][:m]).max():.2e}")
