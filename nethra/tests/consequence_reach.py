"""Consequence reach.  Objects approach a contact zone near the eyes (contact trials) or pass it
(miss trials).  Some trials push F in the interval after the last position (the consequence).
Two silent intervals separate trials.  Optional whole-field colour Nethra (one per colour, pushed
1.0 while an object of that colour is seen).
Read after each interval with the object at k intervals before its last position: P toward F
(sum over F's incidences of max(0, g(A_rel - A_F))) and live activation of F.
usage: reach.py R th n_vision n_trials steps colour(0/1) seed"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
import sys, random, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
A = sys.argv[1:]; sys.argv = [sys.argv[0], A[0]]
import binocular3d as b
import nethra_presence as core
TH = float(A[1]); NV, NT, N = int(A[2]), int(A[3]), int(A[4]); COL = int(A[5]); SEED = int(A[6]) if len(A) > 6 else 1
rng = random.Random(SEED)
f = core.NethraField(frontier_tolerance=float(os.environ.get("TOL", "0.01")), source_similarity_threshold=TH)
for _ in range(2 * b.R * b.R): f.new()
F = f.new(); COLS = [f.new(), f.new()]          # F, colour 0 (food), colour 1
K = len(f.nethra)
T = (250, 460, 30)
def trial(kind, colour, consequence, read):
    start = (rng.uniform(60, 440), rng.uniform(60, 440), rng.uniform(380, 480))
    if kind == "contact": end = T
    else: end = (250 + rng.choice((-1, 1)) * rng.uniform(150, 200), rng.uniform(40, 200), 30)
    out = []
    for i in range(N):
        s = (i + 1) / N
        p = tuple(round(start[a] + s * (end[a] - start[a]) + rng.uniform(-5, 5)) for a in range(3))
        p = tuple(min(499, max(0, x)) for x in p)
        b.push(f, (0, 1), [p])
        if COL: COLS[colour].push(1.0)
        f.step(1.0)
        if read:
            Aint = f.current_interval_integral; P = 0.0
            within = {F} | {rel for rel, _ in f.member_to_routeuses.get(F, ())}
            for (rel, mem), row in f._physical_incidences(f.current_event, within=within).items():
                if mem is F:
                    q = row["g"] * (Aint.get(rel, 0.0) - Aint.get(F, 0.0))
                    if q > 0: P += q
            out.append((N - i, P, F.activation))
    if consequence: F.push(1.0); f.step(1.0)
    for _ in range(2): f.step(1.0)
    return out
kinds = [("contact", 0), ("miss", 0), ("contact", 1), ("miss", 1)] if COL else [("contact", 0), ("miss", 0)]
for _ in range(NV):                              # vision only: objects pass by, nothing consequential
    trial("miss", rng.choice((0, 1)), False, False)
t0 = time.perf_counter(); res = {k: [] for k in kinds}
for t in range(NT):
    kind, colour = kinds[t % len(kinds)]
    cons = kind == "contact" and colour == 0
    r = trial(kind, colour, cons, t >= NT // 2)
    if r: res[(kind, colour)].append(r)
print(f"R={b.R} th {TH} vision {NV} trials {NT} steps {N} colour {COL}: constructed {len(f.nethra) - K}, "
      f"F routes {sum(1 for _ in f.member_to_routeuses.get(F, ()))}, {1000 * (time.perf_counter() - t0) / (NT * (N + 3)):.1f} ms/int")
med = lambda v: sorted(v)[len(v) // 2]
print("k (intervals before last position): " + " ".join(f"{k:>7d}" for k in range(N, 0, -1)))
for key, rs in res.items():
    name = f"{key[0]}{'/food' if COL and key[1] == 0 else '/other' if COL else ''}"
    for j, lab in ((1, "P toward F"), (2, "act F")):
        print(f"{name:14s} {lab:10s}: " + " ".join(f"{med([r[i][j] for r in rs]):7.1e}" for i in range(N)))
