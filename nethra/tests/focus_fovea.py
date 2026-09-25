"""Focus test: coarse periphery (background) + fine gaze-centred fovea + gaze Nethra (focus).
World 500^3, eyes at z=-500, x=220/280 (as binocular3d).  Per eye:
  periphery: RP x RP tent cells over the whole image (world-fixed);
  fovea:     RF x RF tent cells over +-FW image units around the image of the fixation point.
Gaze: 3D grid G^3 of tent cells over the box, pushed at the fixation point (trilinear, 8 cells).
Pursuit device (outside the core): fixation(t) = position of the focused object at t-1.
Stream: object 0 alone tracked (L0 laps), object 1 alone tracked (L1 laps), then both, gaze on 0.
usage: focus_fovea.py FIELD(core|top) TOL alone_laps NJ chunk
env: NOFOV=1 / NOGAZE=1 drop that input; FOVMODE=sustained|transient|both (change-signalling fovea
Nethra pushed with the increase of each fovea cell's input since the previous interval);
GAZEGRID=n gaze cells per axis (default 6); LEAK=leakage (default 1)."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
import sys, math, random, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nethra_presence as core, top_conduction_prototype as top
KIND, TOL, AL, NJ, CH = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
RP, RF, FW, G, J, TH = 8, 8, 0.2, 6, 10, 0.8
WORLD, BASE, DIST = 500, 60.0, 500.0
FOCAL = DIST / (WORLD / 2 + BASE / 2); EYES = (WORLD / 2 - BASE / 2, WORLD / 2 + BASE / 2)
def project(p, ex):
    x, y, z = p; return FOCAL * (x - ex) / (z + DIST), FOCAL * (y - WORLD / 2) / (z + DIST)
def tents(u, lo, hi, n):
    h = (hi - lo) / (n - 1); s = (u - lo) / h; i = math.floor(s); fr = s - i
    return [(k, w) for k, w in ((i, 1 - fr), (i + 1, fr)) if 0 <= k < n and w > 0]
f = (top.TopField if KIND == "top" else core.NethraField)(frontier_tolerance=TOL, source_similarity_threshold=TH,
                                                           leakage=float(os.environ.get("LEAK", "1")))
PER = [[f.new() for _ in range(RP * RP)] for _ in EYES]
FOV = [[f.new() for _ in range(RF * RF)] for _ in EYES]
FOVMODE = os.environ.get("FOVMODE", "sustained")   # sustained | transient | both
G = int(os.environ.get("GAZEGRID", G))
TFOV = [[f.new() for _ in range(RF * RF)] for _ in EYES] if FOVMODE != "sustained" else None
GAZE = [f.new() for _ in range(G ** 3)]
last_fovea = [{}, {}]
K = len(f.nethra)
def periphery_shares(p, e):
    u, v = project(p, EYES[e]); return [(r * RP + c, wu * wv) for c, wu in tents(u, -1, 1, RP) for r, wv in tents(v, -1, 1, RP)]
def fovea_shares(p, fix, e):
    u, v = project(p, EYES[e]); u0, v0 = project(fix, EYES[e])
    return [(r * RF + c, wu * wv) for c, wu in tents(u - u0, -FW, FW, RF) for r, wv in tents(v - v0, -FW, FW, RF)]
def gaze_shares(fix):
    out = []
    for i, wx in tents(fix[0], 0, WORLD - 1, G):
        for j, wy in tents(fix[1], 0, WORLD - 1, G):
            for k, wz in tents(fix[2], 0, WORLD - 1, G): out.append((i * G * G + j * G + k, wx * wy * wz))
    return out
def push(objs, fix):
    for e in (0, 1):
        x = {}
        for p in objs:
            for i, w in periphery_shares(p, e): PER[e][i].push(w)
            if not os.environ.get("NOFOV"):
                for i, w in fovea_shares(p, fix, e):
                    x[i] = x.get(i, 0.0) + w
                    if FOVMODE != "transient": FOV[e][i].push(w)
        if TFOV is not None:          # change-signalling fovea: the increase of each cell's input
            for i, v in x.items():
                d = v - last_fovea[e].get(i, 0.0)
                if d > 0: TFOV[e][i].push(d)
        last_fovea[e] = x
    if not os.environ.get("NOGAZE"):
        for i, w in gaze_shares(fix): GAZE[i].push(w)
def P_toward(cells):
    cells = set(cells); A = f.current_interval_integral; P = {}
    within = set(cells)
    for m in cells:
        for rel, _r in f.member_to_routeuses.get(m, ()): within.add(rel)
    for (rel, mem), row in f._physical_incidences(f.current_event, within=within).items():
        if mem in cells:
            q = row["g"] * (A.get(rel, 0.0) - A.get(mem, 0.0))
            if q > 0: P[mem] = P.get(mem, 0.0) + q
    return P
def centroid(P, coords):
    tot = sum(P.values())
    if tot <= 0: return None
    return tuple(sum(P[n] * coords[n][a] for n in P) / tot for a in range(2))
FOVC = {n: (-FW + (i % RF) * 2 * FW / (RF - 1), -FW + (i // RF) * 2 * FW / (RF - 1)) for e in (0, 1) for i, n in enumerate(FOV[e])}
PERC = {n: (-1 + (i % RP) * 2 / (RP - 1), -1 + (i // RP) * 2 / (RP - 1)) for e in (0, 1) for i, n in enumerate(PER[e])}
def loop(c, rx, ry, rz, per, fy):
    return [(c[0] + rx * math.sin(2 * math.pi * k / per), c[1] + ry * math.sin(fy * 2 * math.pi * k / per + .5),
             c[2] + rz * math.cos(2 * math.pi * k / per)) for k in range(per)]
LOOPS = [loop((250, 130, 250), 200, 80, 200, 40, 2), loop((250, 370, 250), 180, 80, 200, 37, 1)]
rng = random.Random(1); counters = [0, 0]
def where(i):
    base = LOOPS[i][counters[i] % len(LOOPS[i])]; counters[i] += 1
    return tuple(min(499, max(0, round(base[a] + rng.randint(-J, J)))) for a in range(3))
def dist2(a, b): return math.hypot(a[0] - b[0], a[1] - b[1])
def structural_next(cells_of_interest):
    """Cells in the after route of every constructed Nethra whose before route (its first route) is
    complete in this interval's closure (read under current topology); each such route counts once."""
    closed = f.closure(f.previous_explicit, f.current_source_event)
    count = {}
    for n in closed:
        if f._order[n] < K or len(n.routes) < 2: continue
        routes = list(n.routes)
        if not routes[0] <= closed: continue
        for R in routes[1:]:
            for m in R:
                if m in cells_of_interest: count[m] = count.get(m, 0) + 1
    return count
def cellset(shares, cells):
    return {cells[i] for i, w in shares}
def split(P, prevc, curc, nextc):
    tot = sum(P.values())
    if tot <= 0: return None
    g = lambda S: sum(v for n, v in P.items() if n in S) / tot
    return g(prevc - curc - nextc), g(curc), g(nextc - curc - prevc)
def run(phase, objs, n, chunk):
    fix = None; prev = None; nxt = [where(i) for i in objs]; before = None; fix_before = None
    for s in range(0, n, chunk):
        n0 = len(f.nethra); wall = 0.0; fr = []; ef = []; bf = []; eb = []; bb = []; sf = []; sb = []; ep = []; bp = []; sb_ = []; sp_ = []; sbb = []; spb = []
        for _ in range(min(chunk, n - s)):
            ps = nxt; nxt = [where(i) for i in objs]
            fix = prev if prev is not None else ps[0]
            push(ps, fix)
            t0 = time.perf_counter(); f.step(1.0); wall += time.perf_counter() - t0
            fr.append(f.frontier_sizes[-1])
            if before is not None:
                for e in (0, 1):
                    Pf = P_toward(FOV[e])
                    x = split(Pf, cellset(fovea_shares(before[0], fix_before, e), FOV[e]), cellset(fovea_shares(ps[0], fix, e), FOV[e]),
                              cellset(fovea_shares(nxt[0], ps[0], e), FOV[e]))
                    if x: sf.append(x)
                    if len(objs) > 1:
                        u0b = project(ps[1], EYES[e])
                        win = [n for n in PER[e] if max(abs(PERC[n][0] - u0b[0]), abs(PERC[n][1] - u0b[1])) <= 2 * 2 / (RP - 1)]
                        x = split(P_toward(win), cellset(periphery_shares(before[1], e), PER[e]), cellset(periphery_shares(ps[1], e), PER[e]),
                                  cellset(periphery_shares(nxt[1], e), PER[e]))
                        if x: sb.append(x)
            before = ps; fix_before = fix
            prev = ps[0]
            # focused: expected fovea offset at t+1 (fixation then = ps[0]) vs actual and vs current offset
            for e in (0, 1):
                c = centroid(P_toward(FOV[e]), FOVC)
                u1 = project(nxt[0], EYES[e]); u0 = project(ps[0], EYES[e]); uf = project(fix, EYES[e])
                actual = (u1[0] - u0[0], u1[1] - u0[1]); current = (u0[0] - uf[0], u0[1] - uf[1])
                if c is not None and FOVMODE != "transient": ef.append(dist2(c, actual)); bf.append(dist2(current, actual))
                for j in range(len(objs)):     # each object in the periphery, window of 2 spacings
                    u1b = project(nxt[j], EYES[e]); u0b = project(ps[j], EYES[e])
                    win = [n for n in PER[e] if max(abs(PERC[n][0] - u0b[0]), abs(PERC[n][1] - u0b[1])) <= 2 * 2 / (RP - 1)]
                    cb = centroid(P_toward(win), PERC)
                    if cb is not None: (eb if j else ep).append(dist2(cb, u1b)); (bb if j else bp).append(dist2(u0b, u1b))
                    cs = centroid(structural_next(set(win)), PERC)
                    if cs is not None: (sb_ if j else sp_).append(dist2(cs, u1b)); (sbb if j else spb).append(dist2(u0b, u1b))
        k = min(chunk, n - s); gm = lambda v: math.exp(sum(math.log(max(x, 1e-9)) for x in v) / len(v)) if v else float("nan")
        print(f"{phase:8s} {s:4d}-{s + k - 1:4d}: built {len(f.nethra) - n0:3d} frontier {sum(fr) / k:5.0f} {1000 * wall / k:6.1f} ms | "
              f"focused: next fovea offset off by {gm(ef):.3f} (staying put {gm(bf):.3f}); next image point off by {gm(ep):.3f} (staying put {gm(bp):.3f})"
              + (f" | background: next image point off by {gm(eb):.3f} (staying put {gm(bb):.3f})" if len(objs) > 1 else ""), flush=True)
        print(f"{'':8s} structural next (after routes of Nethra refound by their before route): focused {gm(sp_):.3f} (staying put {gm(spb):.3f}, {len(sp_)}/{2 * k} read)"
              + (f", background {gm(sb_):.3f} (staying put {gm(sbb):.3f}, {len(sb_)}/{2 * k} read)" if len(objs) > 1 else ""), flush=True)
        mean3 = lambda v: "/".join(f"{sum(x[i] for x in v) / len(v):.2f}" for i in range(3)) if v else "-"
        print(f"{'':8s} share of P toward previous-only / current / next-only cells: focused fovea {mean3(sf)}" + (f", background periphery {mean3(sb)}" if len(objs) > 1 else ""), flush=True)
run("alone0", [0], AL * 40, AL * 40)
run("alone1", [1], AL * 37, AL * 37)
run("both", [0, 1], NJ, CH)
print(f"constructed {len(f.nethra) - K}; image units: fovea spacing {2 * FW / (RF - 1):.3f}, periphery spacing {2 / (RP - 1):.3f}")
