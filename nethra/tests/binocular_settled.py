"""One object on a closed 3D loop across the box, each lap jittered by +-J units per axis.
Phases: left eye 2 laps, right eye 2 laps, both eyes LAPS laps.  Per lap (both eyes):
built, stored patterns, ms/interval, frontier; internal state = refound constructed Nethra with a 3D
location (b.location): distance from the object to their centroid; expectation = P (prior flow toward
the retinal Nethra in the 5x5 windows) turned into a 3D point (P-weighted image position per eye,
triangulated), distance to the object's NEXT position vs distance of the current position to it.
usage: settled.py R th PER LAPS J [tol]"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
import sys, math, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
A = sys.argv[1:]; sys.argv = [sys.argv[0], A[0]]
import binocular3d as b
import nethra_presence as core
TH = float(A[1]); PER = int(A[2]); LAPS = int(A[3]); J = int(A[4]); TOL = float(A[5]) if len(A) > 5 else 0.01
rng = random.Random(1)
base = [(250 + 200 * math.sin(2 * math.pi * k / PER), 250 + 150 * math.sin(4 * math.pi * k / PER + .5),
         250 + 200 * math.cos(2 * math.pi * k / PER)) for k in range(PER)]
phases = [("L", (0,))] * 2 + [("Rt", (1,))] * 2 + [("B", (0, 1))] * LAPS
pos = []
for _ in phases:
    for k in range(PER):
        pos.append(tuple(min(499, max(0, round(base[k][i] + rng.randint(-J, J)))) for i in range(3)))
pos.append(pos[-PER])
f = core.NethraField(frontier_tolerance=TOL, source_similarity_threshold=TH)
for _ in range(2 * b.R * b.R): f.new()
K = len(f.nethra); cache = {}
rec = {}
_orig = f._move_evidence_and_construct
def _wrap(source_current, manifestation, *a, **kw):
    out = _orig(source_current, manifestation, *a, **kw)
    rec['M'] = sum(manifestation.get(n, 0.0) for n in source_current)
    rec['U'] = sum(max(0.0, out.get(n, 0.0)) for n in source_current)
    rec['P'] = sum(manifestation.get(n, 0.0) - out.get(n, 0.0) for n in source_current)
    return out
f._move_evidence_and_construct = _wrap
def Ppoint(P):
    s = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    for n, v in P.items():
        e, rc = divmod(f._order[n], b.R * b.R); r, c = divmod(rc, b.R)
        s[e][0] += v * b.CENTRE[c]; s[e][1] += v * b.CENTRE[r]; s[e][2] += v
    if s[0][2] <= 0 or s[1][2] <= 0: return None
    uL, vL, uR, vR = s[0][0] / s[0][2], s[0][1] / s[0][2], s[1][0] / s[1][2], s[1][1] / s[1][2]
    d = uL - uR
    if d <= 0: return None
    zD = b.FOCAL * (b.EYES[1] - b.EYES[0]) / d
    return (uL * zD / b.FOCAL + b.EYES[0], (vL + vR) / 2 * zD / b.FOCAL + 250, zD - b.DIST)
med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")
steps = [b.dist(base[k], base[(k + 1) % PER]) for k in range(PER)]
print(f"R={b.R} th {TH} PER {PER} jitter {J} tol {TOL}; base step {min(steps):.0f}-{max(steps):.0f} units")
t = 0
for li, (ph, eyes) in enumerate(phases):
    n0, p0 = len(f.nethra), len(f.source_patterns); wall = 0.0; fr = []; um = []; sd = []; pl = []; pp = []; pe = []; ce = []; ref = []
    for k in range(PER):
        b.push(f, eyes, [pos[t]])
        t0 = time.perf_counter(); f.step(1.0); wall += time.perf_counter() - t0
        fr.append(f.frontier_sizes[-1]); um.append((rec.get('U', 0), rec.get('M', 0), rec.get('P', 0)))
        if ph == "B":
            closed = [n for n in f.closure(f.previous_explicit, f.current_source_event) if f._order[n] >= K]
            ref.append(len(closed))
            pts = [q for q in (b.location(n, cache) for n in closed) if q is not None]
            if pts: sd.append(b.dist(tuple(sum(q[i] for q in pts) / len(pts) for i in range(3)), pos[t]))
            cells = [f.nethra[b.R * b.R * e + b.R * r + c] for e in eyes for r, c in b.window(e, pos[t])]
            q = Ppoint(b.prior_flow(f, cells))
            if q is not None: pe.append(b.dist(q, pos[t + 1])); ce.append(b.dist(pos[t], pos[t + 1])); pp.append(b.dist(q, pos[t - 1]))
            for e in eyes: pl.append(b.ahead(f, e, pos[t], pos[t + 1], pos[t - 1])[0])
        t += 1
    line = f"{ph:2s} lap {li + 1:2d}: built {len(f.nethra) - n0:3d} patterns+{len(f.source_patterns) - p0:3d} frontier {sum(fr) / PER:5.0f} {1000 * wall / PER:6.1f} ms"
    if ph == "B":
        line += (f" | refound {sum(ref) / PER:4.1f} state dist med {med(sd):4.0f} | P point: to next med {med(pe):4.0f} "
                 f"(now to next {med(ce):4.0f}; to prev {med(pp):4.0f}) P lead {sum(x for x in pl if x == x) / max(1, sum(1 for x in pl if x == x)):+.2f} sp")
    line += f' | unresolved/M {sum(u for u, m, p in um) / max(1e-300, sum(m for u, m, p in um)):.3f} P/M at source {sum(p for u, m, p in um) / max(1e-300, sum(m for u, m, p in um)):.3f}'
    print(line, flush=True)
print(f"constructed {len(f.nethra) - K}, patterns {len(f.source_patterns)}")
