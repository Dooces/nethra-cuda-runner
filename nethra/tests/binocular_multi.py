"""Objects on jittered closed loops.  Each object alone first (left eye, right eye, both eyes), then all
together.  usage: multi.py R th J alone_laps joint_intervals chunk [tol]"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
import sys, math, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
A = sys.argv[1:]; sys.argv = [sys.argv[0], A[0]]
import binocular3d as b
import nethra_presence as core
TH = float(A[1]); J = int(A[2]); AL = int(A[3]); NJ = int(A[4]); CH = int(A[5]); TOL = float(A[6]) if len(A) > 6 else 0.01
rng = random.Random(1)
def loop(c, rx, ry, rz, per, fy):
    return [(c[0] + rx * math.sin(2 * math.pi * k / per), c[1] + ry * math.sin(fy * 2 * math.pi * k / per + .5),
             c[2] + rz * math.cos(2 * math.pi * k / per)) for k in range(per)]
LOOPS = [loop((250, 130, 250), 200, 80, 200, 40, 2), loop((250, 370, 250), 180, 80, 200, 30, 1)]
counters = [0] * len(LOOPS)
def where(i):
    base = LOOPS[i][counters[i] % len(LOOPS[i])]; counters[i] += 1
    return tuple(min(499, max(0, round(base[a] + rng.randint(-J, J)))) for a in range(3))
import partwise_prototype as partwise
FIELD = (partwise.ExactSourcePartwiseField if os.environ.get("EXACTSRC") else partwise.PartwiseField) if os.environ.get("PARTWISE") else core.NethraField
f = FIELD(frontier_tolerance=TOL, source_similarity_threshold=TH)
if os.environ.get("PARTWISE"): f.subtraction_log = []
for _ in range(2 * b.R * b.R): f.new()
K = len(f.nethra); cache = {}; origin = {}
med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")
def run(phase, objs, eyes, n, chunk):
    for s in range(0, n, chunk):
        n0, p0 = len(f.nethra), len(f.source_patterns); wall = 0.0; fr = []; sd = {i: [] for i in objs}; ref = []
        for _ in range(min(chunk, n - s)):
            ps = {i: where(i) for i in objs}
            b.push(f, eyes, list(ps.values()))
            t0 = time.perf_counter(); f.step(1.0); wall += time.perf_counter() - t0
            m0 = len(f.nethra)
            fr.append(f.frontier_sizes[-1])
            for nn in f.nethra[n0:]: origin.setdefault(nn, phase)
            if len(eyes) == 2:
                closed = [nn for nn in f.closure(f.previous_explicit, f.current_source_event) if f._order[nn] >= K]
                ref.append(len(closed))
                # internal state per object: located single-object Nethra nearest to each object
                pts = [q for q in (b.location(nn, cache) for nn in closed if origin.get(nn, "").startswith("alone")) if q is not None]
                for i, p in ps.items():
                    mine = [q for q in pts if min(ps, key=lambda j: b.dist(q, ps[j])) == i]
                    if mine: sd[i].append(b.dist(tuple(sum(q[a] for q in mine) / len(mine) for a in range(3)), p))
        k = min(chunk, n - s)
        print(f"{phase:10s} {s:4d}-{s + k - 1:4d}: built {len(f.nethra) - n0:3d} patterns+{len(f.source_patterns) - p0:3d} "
              f"frontier {sum(fr) / k:5.0f} {1000 * wall / k:6.1f} ms" + (f" | refound {sum(ref) / k:5.1f} state dist per object "
              + " ".join(f"{i}:{med(v):.0f}({len(v)}/{k})" for i, v in sd.items()) if ref else ""), flush=True)
for i, L in enumerate(LOOPS):
    if not os.environ.get("NOMONO"):
        run(f"alone{i}-L", [i], (0,), AL * len(L), AL * len(L))
        run(f"alone{i}-R", [i], (1,), AL * len(L), AL * len(L))
    run(f"alone{i}-B", [i], (0, 1), 2 * AL * len(L), AL * len(L))
run("together", list(range(len(LOOPS))), (0, 1), NJ, CH)
print(f"constructed {len(f.nethra) - K}, patterns {len(f.source_patterns)}")
if getattr(f, "subtraction_log", None) is not None:
    from collections import Counter; print("together subtraction (handles, unaccounted before, after):", Counter(f.subtraction_log[-NJ:]).most_common(8))
