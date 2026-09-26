"""Tiny streams for flipped channels (flip_prototype.py), 300 trials each (bounce: 300 intervals),
probed every BLOCK trials on a frozen copy (3 silent intervals, push the probe intervals, read the
live activation of the named Nethra after the last one).

usage: EXPECT=off|structure|residual CLOSURE=field|absent FLIPTO=all|leaves python3 flip_streams.py STREAM [seed]
Every condition is direction="split", integrator RK4; EXPECT=off is bit-identical to the core with
those settings.  Outcomes are never silent (construction builds nothing across a silent interval).

Predictions (EXPECT=structure, CLOSURE=field; written before running):
negfeature  A -> X, A B -> Y, C -> X.  The A B -> Y handle H gets flipped member X (the A -> X
            Nethra, refound by its before route in the A B interval, expects X).  H pulls X when
            A and B are present.  X after AB / A falls from about 1.1 (off) toward 0.5 over 300
            trials; X after CB / C only slightly below 1 (B reaches H by conduction alone); Y after
            AB rises (pulled charge enters H, which conducts to Y).
negpattern  A -> X, B -> X, A B -> Y.  Same handle, same flip: X after AB / A falls from about
            1.8 toward 1 or below.
operator    [c_i, O+] -> c(i+1), [c_i, O-] -> c(i-1), O+ held out on c3.  First sight of an O-
            transition is accounted by the reverse sequence Nethra (direction-blind per-part
            accounting), so the forward sequence Nethra c(i-1) -> c_i gets flipped c(i+1).
            It also pulls in O+ trials, where c(i+1) comes, so the flip weakens there.  Expect a
            small, unclear change on trained operands; none on held-out c3.
prob        X -> B (75%) or C (25%).  X -> B Nethra gets flipped C and X -> C gets flipped B;
            the flipped C on X -> B strengthens in 75% of trials.  Share of B after X rises from
            about 0.59 toward 0.65-0.75; a share near 1.0 would be a winner forming.
bounce      6 positions, 0..5..0 (period 10), 300 intervals.  At the turns the continuation is
            expected and absent, so turn Nethra get flipped members.  Probe moving right (p1, p2)
            and moving left (p3, p2); share of the next cell in the direction of motion.  Expect
            equal to or above off.
extinction  A -> X 100, A -> Y 100, A -> X 100.  A -> Y handle gets flipped X; X after A falls
            further than off during A -> Y.  Evidence of A -> X is spared (the net residual at
            X counts the pull), so relearning is faster than first learning (savings).
EXPECT=residual: many more flips (every over-carried Nethra); broader pulls; A -> X itself may be
            pulled in negfeature.  CLOSURE=absent: equal to field except where a flipped member is
            present together with its Nethra.
"""
import os, sys, random, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flip_prototype as fp

core = fp.load_core(); FF = fp.make(core)
EXPECT = os.environ.get("EXPECT", "off"); CLOSURE = os.environ.get("CLOSURE", "field")
FLIPTO = os.environ.get("FLIPTO", "all")
STREAM = sys.argv[1]; SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
BLOCK = int(os.environ.get("BLOCK", 50))

f = FF(expect=None if EXPECT == "off" else EXPECT, closure_mode=CLOSURE, flip_to=FLIPTO)
t0 = time.perf_counter()
rng = random.Random(SEED)
lowest = [0.0]; NI = [0]


def show(xs):
    for x in xs: x.push(1.0)
    f.step(1.0)
    NI[0] += 1
    lo = min(n.activation for n in f.nethra)
    if lo < lowest[0]: lowest[0] = lo


def probe(intervals, reads):
    g = f.frozen_copy(); idx = {n: j for j, n in enumerate(f.nethra)}
    for _ in range(3): g.step(1.0)
    for xs in intervals:
        for x in xs: g.nethra[idx[x]].push(1.0)
        g.step(1.0)
    return [g.nethra[idx[r]].activation for r in reads]


def run(trials, pf):
    rows = [pf()]
    for k, t in enumerate(trials, 1):
        for xs in t: show(xs)
        if k % BLOCK == 0: rows.append(pf())
    return rows


fill = lambda F: [rng.choice(F)]
share = lambda u, d: u / (u + d) if u + d > 0 else float("nan")

if STREAM in ("negfeature", "negpattern"):
    A, B, C, X, Y = (f.new() for _ in range(5)); F = [f.new() for _ in range(4)]
    if STREAM == "negfeature":
        kinds = [([A], [X]), ([A, B], [Y]), ([C], [X])]
        probes = [("A", [A]), ("AB", [A, B]), ("B", [B]), ("C", [C]), ("CB", [C, B])]
    else:
        kinds = [([A], [X]), ([B], [X]), ([A, B], [Y])]
        probes = [("A", [A]), ("B", [B]), ("AB", [A, B])]
    tr = [[fill(F), cue, out, fill(F)] for _ in range(100) for cue, out in kinds]
    rng.shuffle(tr)
    def pf():
        row = []
        for _, cue in probes:
            x, y = probe([cue], [X, Y]); row += [x, y]
        return row
    rows = run(tr, pf)
    hdr = "X,Y after " + " ".join(p for p, _ in probes)
elif STREAM == "operator":
    K, H, T = 6, 3, 1
    c = [f.new() for _ in range(K)]; Op, Om = f.new(), f.new(); F = [f.new() for _ in range(4)]
    for _ in range(8):
        for k in range(K): show([c[k]])
    tr = [(i, +1) for i in range(K) if i != H] + [(i, -1) for i in range(K)]
    tr = tr * 25; rng.shuffle(tr)
    tr = [[fill(F), [c[i], Op if d > 0 else Om], [c[(i + d) % K]], fill(F)] for i, d in tr]
    def pf():
        row = []
        for x in (T, H):
            for tag in ([Op], [Om], []):
                u, d = probe([[c[x]] + tag], [c[(x + 1) % K], c[(x - 1) % K]]); row.append(share(u, d))
        return row
    rows = run(tr, pf)
    hdr = "share c(x+1): trained O+ O- none | held-out O+ O- none"
elif STREAM == "prob":
    X, Bo, Co = f.new(), f.new(), f.new(); F = [f.new() for _ in range(4)]
    tr = [[fill(F), [X], [Bo if rng.random() < 0.75 else Co], fill(F)] for _ in range(300)]
    def pf():
        b, cc = probe([[X]], [Bo, Co]); return [share(b, cc), b, cc]
    rows = run(tr, pf)
    hdr = "share B, act B, act C"
elif STREAM == "bounce":
    p = [f.new() for _ in range(6)]
    path = list(range(6)) + list(range(4, 0, -1))          # 0..5..1, period 10
    tr = [[[p[k]] for k in path] for _ in range(30)]
    BLOCK = 5
    def pf():
        r3, r1 = probe([[p[1]], [p[2]]], [p[3], p[1]])     # moving right
        l1, l3 = probe([[p[3]], [p[2]]], [p[1], p[3]])     # moving left
        return [share(r3, r1), share(l1, l3)]
    rows = run(tr, pf)
    hdr = "share of next cell along motion: moving right, moving left  (block = 5 laps)"
elif STREAM == "extinction":
    A, X, Y = f.new(), f.new(), f.new(); F = [f.new() for _ in range(4)]
    BLOCK = 20
    tr = [[fill(F), [A], [X], fill(F)] for _ in range(100)] + \
         [[fill(F), [A], [Y], fill(F)] for _ in range(100)] + \
         [[fill(F), [A], [X], fill(F)] for _ in range(100)]
    rows = run(tr, lambda: probe([[A]], [X, Y]))
    hdr = "X, Y after A  (block = 20; 0-100 A->X, 100-200 A->Y, 200-300 A->X)"
else:
    raise SystemExit("unknown stream")

live = [e for e in f.flips.values() if e > 0.0]
hmax = max((f.conductance(e) for e in live), default=0.0)
print(f"{STREAM} seed={SEED} expect={EXPECT} closure={CLOSURE} to={FLIPTO}  Nethra {len(f.nethra)}  flips {len(live)}"
      f" (of {len(f.flips)})  max h {hmax:.3f}  lowest activation {lowest[0]:.1e}"
      f"  ms/interval incl. probes {1000 * (time.perf_counter() - t0) / max(1, NI[0]):.1f}")
print("  " + hdr)
for k, r in enumerate(rows):
    print(f"  {k * BLOCK:4d} " + " ".join(f"{v:.4f}" for v in r))
