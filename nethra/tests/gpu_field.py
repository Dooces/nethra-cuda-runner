"""Execution only: integrate each interval on the GPU (cupy), same equation as nethra.py.

GpuField runs the ETD integrator of nethra.py (_etd_prepare / _etd_interval / _derivative_compiled)
with its arrays on the device, for every interval size (the CPU `auto` switches to RK4 above 400
Nethra because a dense eigendecomposition per interval is too slow there).  Topology, evidence
change, closure and construction are untouched and stay on the CPU.  Results are not bit-identical
to the CPU (different summation order); compare with a tolerance.
NETHRA_GPU=0 (or no cupy) runs the same code with numpy, for checking the code path."""
import os
import numpy as np
import nethra_presence as core
try:
    import cupy as cp
except ImportError:
    cp = None
xp = cp if (cp is not None and os.environ.get("NETHRA_GPU", "1") != "0") else np
ON_GPU = xp is not np


def _host(x):
    return cp.asnumpy(x) if ON_GPU else x


class GpuMixin:
    @staticmethod
    def _phi_dev(z):
        small = xp.abs(z) < 1e-3
        zs = xp.where(small, 1.0, z)
        ez = xp.exp(zs)
        p1 = xp.where(small, 1 + z / 2 + z * z / 6, (ez - 1) / zs)
        p2 = xp.where(small, 0.5 + z / 6 + z * z / 24, (ez - 1 - zs) / zs ** 2)
        p3 = xp.where(small, 1 / 6 + z / 24 + z * z / 120, (ez - 1 - zs - zs * zs / 2) / zs ** 3)
        return p1, p2, p3

    def _dev(self, compiled):
        d = compiled.get("dev")
        if d is None:
            d = compiled["dev"] = {k: xp.asarray(compiled[k]) for k in
                                   ("ei", "ej", "eg", "inc_r", "inc_n", "inc_g", "pair_r", "pair_1", "pair_2", "pair_i")}
            d["N"] = compiled["N"]
        return d

    def _etd_prepare(self, compiled, h):
        key = ("etd", h)
        if key in compiled:
            return compiled[key]
        N = compiled["N"]
        Aop = np.zeros((N, N))
        ei, ej, eg = compiled["ei"], compiled["ej"], compiled["eg"]
        np.add.at(Aop, (ei, ej), eg); np.add.at(Aop, (ej, ei), eg)
        np.add.at(Aop, (ei, ei), -eg); np.add.at(Aop, (ej, ej), -eg)
        Aop[np.diag_indices(N)] -= self.leakage
        Aop /= self.capacitance
        A = xp.asarray(Aop)
        mu, Q = xp.linalg.eigh(A)
        op = lambda f: (Q * f) @ Q.T
        E = op(xp.exp(h * mu)); E2 = op(xp.exp(h * mu / 2))
        q1, _q2, _q3 = self._phi_dev(h * mu / 2)
        p1, p2, p3 = self._phi_dev(h * mu)
        data = dict(A=A, Ainv=op(1.0 / mu), E=E, E2=E2, Q2=op(q1),
                    f1=op(p1 - 3 * p2 + 4 * p3), f2=op(2 * (p2 - 2 * p3)), f3=op(4 * p3 - p2))
        compiled[key] = data
        return data

    def _derivative_dev(self, c, a, external):
        N = c["N"]
        current = external - self.leakage * a
        if c["eg"].size:
            flow = c["eg"] * (a[c["ei"]] - a[c["ej"]])
            current = current - xp.bincount(c["ei"], weights=flow, minlength=N) + xp.bincount(c["ej"], weights=flow, minlength=N)
        if self.convergence_gain > 0.0 and c["pair_r"].size:
            inc_r, inc_n = c["inc_r"], c["inc_n"]
            p = c["inc_g"] * (a[inc_n] - a[inc_r])
            positive = p > 0.0
            p = xp.where(positive, p, 0.0)
            count = xp.bincount(inc_r, weights=positive.astype(float), minlength=N)
            total = xp.bincount(inc_r, weights=p, minlength=N)
            pair_sum = xp.bincount(c["pair_r"], weights=p[c["pair_1"]] * p[c["pair_2"]] * c["pair_i"], minlength=N)
            valid = (count >= 2.0) & (total > 0.0)
            safe_total = xp.where(valid, total, 1.0)
            bonus = xp.where(valid, xp.minimum(total, self.convergence_gain * (2.0 * pair_sum / safe_total)), 0.0)
            bonus = xp.where(bonus > 0.0, bonus, 0.0)
            current = current + bonus
            give = bonus[inc_r] * p / safe_total[inc_r]
            current = current - xp.bincount(inc_n, weights=give, minlength=N)
        return current / self.capacitance

    def _etd_interval(self, initial, dt, interval_nodes, compiled):
        pieces = max(1, self.etd_pieces)
        h = float(dt) / pieces
        d = self._etd_prepare(compiled, h)
        c = self._dev(compiled)
        ext = xp.asarray(np.array([n.external for n in interval_nodes], dtype=float))
        a = xp.asarray(np.array([initial[n] for n in interval_nodes], dtype=float))
        area = xp.zeros(len(interval_nodes))
        A = d["A"]
        Nl = lambda x: self._derivative_dev(c, x, ext) - A @ x
        for _ in range(pieces):
            Na = Nl(a)
            a2 = d["E2"] @ a + (h / 2) * (d["Q2"] @ Na)
            Nb = Nl(a2)
            b2 = d["E2"] @ a + (h / 2) * (d["Q2"] @ Nb)
            Nc = Nl(b2)
            cc = d["E2"] @ a2 + (h / 2) * (d["Q2"] @ (2 * Nc - Na))
            Nd = Nl(cc)
            nxt = d["E"] @ a + h * (d["f1"] @ Na + d["f2"] @ (Nb + Nc) + d["f3"] @ Nd)
            intN = h * (Na + 2 * Nb + 2 * Nc + Nd) / 6.0
            area = area + d["Ainv"] @ (nxt - a - intN)
            a = nxt
        a, area = _host(a), _host(area)
        if not (np.isfinite(a).all() and np.isfinite(area).all()):
            raise FloatingPointError("non-finite Nethra field state after ETD interval")
        return ({n: float(v) for n, v in zip(interval_nodes, a)},
                {n: float(v) for n, v in zip(interval_nodes, area)})

    def _rk4_interval(self, initial, dt, edges, neighbors, interval_nodes, compiled=None):
        if compiled is None:
            compiled = self._compile_interval(interval_nodes, edges, neighbors)
        return self._etd_interval(initial, dt, interval_nodes, compiled)


class GpuField(GpuMixin, core.NethraField):
    pass
