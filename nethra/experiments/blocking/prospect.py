"""Kamin stream (human.py test 1).  At each O interval: complete prospective state at O vs the observation's own
response, same coordinate and span.  E = zero-source run, S = actual - zero.  Both at interval end and as interval
integral.  surv = O's start activation decayed over the interval (leak only); redist = E_end - surv.
Also: evidence X's incidences gained in this step (moved before _construct), and what _accounted returns."""
import sys
from math import exp
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")); from nethra import NethraField
CAP = {}
orig_rk4 = NethraField._rk4_interval
def rk4(self, initial, dt, *a, **k):
    out = orig_rk4(self, initial, dt, *a, **k); CAP.setdefault("runs", []).append(out); CAP["a0"] = dict(initial); return out
NethraField._rk4_interval = rk4
orig_acc = NethraField._accounted
def acc(self, before, after):
    out = orig_acc(self, before, after); CAP.setdefault("acc", []).append(out); return out
NethraField._accounted = acc
orig_mv = NethraField._move_evidence_and_construct
def mv(self, sc, man, *a, **k):
    ev0 = CAP["xev"](); orig_c = self._construct
    def c(*aa, **kk):
        CAP["dx_before_construct"] = CAP["xev"]() - ev0; return orig_c(*aa, **kk)
    self._construct = c
    try: r = orig_mv(self, sc, man, *a, **k)
    finally: del self._construct
    CAP["mmp"] = r; return r
NethraField._move_evidence_and_construct = mv
def Pflow(f, n):
    A = f.current_interval_integral; p = 0.0
    for (r, m), row in f._physical_incidences(f.current_event).items():
        if m is n:
            q = row["g"] * (A.get(r, 0.0) - A.get(m, 0.0))
            if q > 0: p += q
    return p
for group in ("blocked", "control"):
    f = NethraField(); L = [f.new() for _ in range(5)]; O, X = L[2], L[1]
    nm = {id(x): s for x, s in zip(L, "AXOZK")}
    CAP["xev"] = lambda: sum(max(c.values(), default=0) for (r, rt, m), c in f.incidence_evidence.items() if m is X)
    def show(xs):
        for x in xs: L[x].push(1.0)
        f.step(1.0)
    pre = 0 if group == "blocked" else 4
    print(f"\n{group}   trial | E_end  surv  redist  S_end | E_int  S_int | P(flow) | dEvid X before construct | accounted by")
    trials = [("pre", t) for t in range(1, 61)] + [("cmp", t) for t in range(1, 121)]
    for ph, t in trials:
        show([0, 1] if ph == "cmp" else [pre])
        p = Pflow(f, O)          # flow statistic that step() will use for O
        CAP.clear(); CAP["xev"] = CAP.get("xev") or (lambda: sum(max(c.values(), default=0) for (r, rt, m), c in f.incidence_evidence.items() if m is X))
        show([2])
        (z_end, z_int), (a_end, a_int) = CAP["runs"][-2], CAP["runs"][-1]
        a0 = CAP["a0"].get(O, 0.0); surv = a0 * exp(-f.leakage * 1.0 / f.capacitance)
        E, S = z_end[O], a_end[O] - z_end[O]; Ei, Si = z_int.get(O, 0), a_int.get(O, 0) - z_int.get(O, 0)
        accs = CAP.get("acc", [])
        who = sorted({f"#{f._order[r]}" for a in accs for (r, _l, _r) in a})
        if (ph, t) in {("pre", 1), ("pre", 10), ("pre", 60), ("cmp", 1), ("cmp", 10), ("cmp", 30), ("cmp", 60), ("cmp", 90), ("cmp", 120)}:
            print(f"  {ph} {t:3d} | {E:.3f} {surv:.3f} {E-surv:+.3f} {S:.3f} | {Ei:.3f} {Si:.3f} | {p:.3f} | {CAP.get('dx_before_construct', 0):9.2f} | {who if accs else 'no call (by-parts path?)'}")
        show([3])
