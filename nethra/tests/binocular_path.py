"""One object going round the same closed path, seen by both eyes (binocular3d.py's world and
retina).  Does the field come to carry its expectation ahead of the object?

Path: a circle of radius RADIUS in the x-z plane (depth changes along it), centre (250, 250, 250),
PER intervals per lap, integer positions (so the graded source patterns recur every lap).
Reads after step t, before t+1 is pushed, per eye:
  P next/prev    prior flow the field carries toward the retinal Nethra for the next step (the P
                 step t+1 subtracts from the manifestation), sampled through the receptive tents at
                 the object's next image point, over the same at its previous image point.
  act next/prev  live activation sampled the same way; "ref" = a field fed the same stream with
                 topology_and_evidence_change=False (leftover only).
  state          distance from the object to the centre of the 3D locations of the refound
                 constructed Nethra (see binocular3d.py).
  built, ms/int  construction per lap and wall time per step (reads excluded), one numeric thread.

usage: python3 binocular_path.py [R] [laps] [frontier tolerance] [radius] [intervals per lap]"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys, math, time
ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], ARGS[0] if ARGS else "16"]
import binocular3d as b
import nethra_presence as core

LAPS = int(ARGS[1]) if len(ARGS) > 1 else 20
TOL = float(ARGS[2]) if len(ARGS) > 2 else 0.01
RADIUS = int(ARGS[3]) if len(ARGS) > 3 else 150
PER = int(ARGS[4]) if len(ARGS) > 4 else 60
PATH = [(round(250 + RADIUS * math.cos(2 * math.pi * k / PER)), 250,
         round(250 + RADIUS * math.sin(2 * math.pi * k / PER))) for k in range(PER)]


def gm(v):
    w = [x for x in v if x == x and x > 0]
    return math.exp(sum(math.log(x) for x in w) / len(w)) if w else float("nan")


def med(v):
    w = sorted(x for x in v if x == x)
    return w[len(w) // 2] if w else float("nan")


if __name__ == "__main__":
    f = core.NethraField(frontier_tolerance=TOL)
    g = core.NethraField(frontier_tolerance=TOL, topology_and_evidence_change=False)
    for fld in (f, g):
        for _ in range(2 * b.R * b.R): fld.new()
    built_in = set(); loc_cache = {}
    step = (sum(((PATH[(k + 1) % PER][i] - PATH[k][i]) ** 2 for i in range(3))) ** .5 for k in range(PER))
    print(f"R={b.R}, frontier tolerance {TOL}, radius {RADIUS}, {PER} intervals per lap "
          f"(mean step {sum(step) / PER:.1f} units; grid spacing ~{2 * 500 / (b.R - 1) / 2:.0f}-{2 * 1000 / (b.R - 1) / 2:.0f} units)")
    for lap in range(LAPS):
        n0 = len(f.nethra); wall = 0.0; pr = []; ar = []; rr = []; state = []
        for k in range(PER):
            p, q, o = PATH[k], PATH[(k + 1) % PER], PATH[(k - 1) % PER]
            m0 = len(f.nethra)
            for fld in (f, g): b.push(fld, (0, 1), [p])
            t0 = time.perf_counter(); f.step(1.0); wall += time.perf_counter() - t0
            g.step(1.0)
            built_in.update(f.nethra[m0:])
            if lap == 0 and k == 0: continue
            for e in (0, 1):
                _pl, p_ratio, a_ratio = b.ahead(f, e, p, q, o)
                pr.append(p_ratio); ar.append(a_ratio); rr.append(b.ahead(g, e, p, q, o, with_p=False)[2])
            closed = f.closure(f.previous_explicit, f.current_source_event)
            pts = [x for x in (b.location(n, loc_cache) for n in closed if n in built_in) if x is not None]
            if pts:
                state.append(b.dist(tuple(sum(x[i] for x in pts) / len(pts) for i in range(3)), p))
        if lap < 3 or (lap + 1) % 5 == 0:
            print(f"lap {lap + 1:3d}: built {len(f.nethra) - n0:3d}  P next/prev {gm(pr):5.2f}  act next/prev {gm(ar):.3f} "
                  f"(ref {gm(rr):.3f})  state {med(state):5.1f}  {1000 * wall / PER:5.1f} ms/int", flush=True)
