"""Tiny streams for the drain prototype (drain_prototype.py), a few hundred trials each, probed
every BLOCK trials on a frozen copy (3 silent intervals, push the probe, one interval, read the
live activation of the outcome leaves).  Conditions: DRAIN = off | leaves | built | both.
Integration RK4 in every condition (drain off is bit-identical to the core with integrator="rk4").

usage: DRAIN=leaves [DRATE=1] python3 drain_streams.py STREAM [seed]

Streams and predictions (written before running, from the code):

negfeature  A -> X, A B -> (silent), C -> X; each a third of 300 trials.  Probes A, AB, B, C, CB.
    off:    X after AB / A about 0.9-1.1 (B has no incidence toward X; only A's own excitatory
            evidence dips after AB trials), CB / C about 1.
    leaves: B -| X grows in AB trials only; A -| X grows in AB trials and shrinks in A trials
            (M at X is large there), so about 0.  AB / A < 1; CB / C < 1 (B transfers:
            summation).  B alone: X about as off.
    built:  the Nethra built for (filler -> A B) are refound only with A and B together, so they
            drain X in AB and not in CB: AB / A < 1, CB / C about 1 (configural, no transfer).
negpattern  A -> X, B -> X, A B -> (silent).  Probes A, B, AB.
    off:    AB > A (contributions add).
    leaves: A -| X and B -| X each grow in AB trials and shrink in their own X trials: no stable
            net drain, AB stays >= A (like any elemental account).
    built:  AB-only Nethra drain X: AB / A falls below off, possibly below 1.
operator    the op_abstract stream: 6 symbols cycled forward 8 laps, then [c_i, O+] -> c(i+1),
            [c_i, O-] -> c(i-1), O+ held out on c3; 25 per (operand, op).  Probes: share of
            activation c(x+1) / (c(x+1) + c(x-1)) for trained c1 and held-out c3, tags O+, O-, none.
    off:    about 0.49 / 0.49 trained (as measured before).
    leaves: the tag is operand in half its trials and result-neighbour in the other half; each
            leaf's drain grows in one kind of trial and shrinks in another: no selection.
    built:  conjunction Nethra (c_i, O-) drain c(i+1): trained share under O- < under O+.
            Held-out c3 under O+: no conjunction Nethra with O+ and c3: about as off.
prob        X -> B (75%) or C (25%).  Probe X: share B / (B + C).
    off:    about 0.6-0.65 (notes: 0.63).  leaves / built: within 0.05 of off (drains toward
            the rarer outcome grow in 75% of trials and shrink harder in 25%: no winner).
ring        6 positions pushed in a cycle, 50 laps (the trial is one lap).  Probe: position 2
            pushed after 3 silent intervals; share of activation p3 / (p3 + p1).
    all:    small effect: P toward the previous cell is 0 (its own leftover), so no drain is
            learned there; drains grow only where P > 0 and the outcome did not come.
extinction  A -> X 100 trials, A -> (silent) 100, A -> X 100.  Probe A: activation of X.
    off:    rises, falls during extinction, rises again.
    leaves: falls faster in extinction (A -| X grows), rises again about as fast as off
            (the drain shrinks at rate ~ A_X * M, large once X comes).

Also recorded per run: drain incidences at the end, the largest h, the lowest activation seen
(must stay >= 0), ms per interval.
"""
import os, sys, random, time, statistics as st
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drain_prototype as dp

core = dp.load_core(); DF = dp.make(core)
DRAIN = os.environ.get("DRAIN", "off")
STREAM = sys.argv[1]; SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
BLOCK = int(os.environ.get("BLOCK", 50))

DRATE = float(os.environ.get("DRATE", 1))   # multiple of the stated rate; >1 only to measure the ceiling
f = DF(drain_source=None if DRAIN == "off" else DRAIN)
f.drain_rate *= DRATE
HMAX = float(os.environ.get("HMAX", 1))   # drain ceiling as a multiple of g_max; >1 only to measure
f.drain_h_max *= HMAX
t0 = time.perf_counter()
rng = random.Random(SEED)
lowest = [0.0]
N_INTERVALS = [0]


def show(xs):
    for x in xs: x.push(1.0)
    f.step(1.0)
    N_INTERVALS[0] += 1
    lo = min(n.activation for n in f.nethra)
    if lo < lowest[0]: lowest[0] = lo


def probe(cues, reads):
    g = f.frozen_copy(); idx = {n: j for j, n in enumerate(f.nethra)}
    for _ in range(3): g.step(1.0)
    for x in cues: g.nethra[idx[x]].push(1.0)
    g.step(1.0)
    return [g.nethra[idx[r]].activation for r in reads]


def trials_then_probe(trials, probe_fn):
    rows = [probe_fn()]
    for k, t in enumerate(trials, 1):
        for xs in t: show(xs)
        if k % BLOCK == 0: rows.append(probe_fn())
    return rows


fill = lambda F: [rng.choice(F)]
share = lambda u, d: u / (u + d) if u + d > 0 else float("nan")
out_hdr, rows = "", []

if STREAM in ("negfeature", "negpattern"):
    A, B, C, X = f.new(), f.new(), f.new(), f.new(); F = [f.new() for _ in range(4)]
    if STREAM == "negfeature":
        kinds = [([A], [X]), ([A, B], []), ([C], [X])]
        probes = [("A", [A]), ("AB", [A, B]), ("B", [B]), ("C", [C]), ("CB", [C, B])]
    else:
        kinds = [([A], [X]), ([B], [X]), ([A, B], [])]
        probes = [("A", [A]), ("B", [B]), ("AB", [A, B])]
    tr = []
    for _ in range(100):
        for cue, out in kinds: tr.append([fill(F), cue, out, fill(F)])
    rng.shuffle(tr)
    rows = trials_then_probe(tr, lambda: [probe(c, [X])[0] for _, c in probes])
    out_hdr = "X after " + " ".join(p for p, _ in probes)
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
                u, d = probe([c[x]] + tag, [c[(x + 1) % K], c[(x - 1) % K]])
                row.append(share(u, d))
        return row
    rows = trials_then_probe(tr, pf)
    out_hdr = "share c(x+1): trained O+ O- none | held-out O+ O- none"
elif STREAM == "prob":
    X, Bo, Co = f.new(), f.new(), f.new(); F = [f.new() for _ in range(4)]
    tr = [[fill(F), [X], [Bo if rng.random() < 0.75 else Co], fill(F)] for _ in range(300)]
    def pf():
        b, cc = probe([X], [Bo, Co]); return [share(b, cc), b, cc]
    rows = trials_then_probe(tr, pf)
    out_hdr = "share B, act B, act C"
elif STREAM == "ring":
    p = [f.new() for _ in range(6)]
    tr = [[[p[k]] for k in range(6)] for _ in range(50)]
    BLOCK = 5
    def pf():
        u, d = probe([p[2]], [p[3], p[1]]); return [share(u, d), u, d]
    rows = trials_then_probe(tr, pf)
    out_hdr = "share p3, act p3, act p1  (block = 5 laps)"
elif STREAM == "extinction":
    A, X = f.new(), f.new(); F = [f.new() for _ in range(4)]
    BLOCK = 20
    tr = [[fill(F), [A], [X], fill(F)] for _ in range(100)] + \
         [[fill(F), [A], [], fill(F)] for _ in range(100)] + \
         [[fill(F), [A], [X], fill(F)] for _ in range(100)]
    rows = trials_then_probe(tr, lambda: [probe([A], [X])[0]])
    out_hdr = "X after A  (block = 20 trials; 0-100 acquisition, 100-200 extinction, 200-300 again)"
else:
    raise SystemExit("unknown stream")

hmax = max((f.drain_h(d) for d in f.drain_evidence.values()), default=0.0)
print(f"{STREAM} seed={SEED} drain={DRAIN} rate x{DRATE:g} ceiling x{HMAX:g}  Nethra {len(f.nethra)}  drain incidences {len(f.drain_evidence)}"
      f"  max h {hmax:.3f}  lowest activation {lowest[0]:.2e}"
      f"  ms/interval incl. probes {1000 * (time.perf_counter() - t0) / max(1, N_INTERVALS[0]):.1f}")
print("  " + out_hdr)
for k, r in enumerate(rows):
    print(f"  {k * BLOCK:4d} " + " ".join(f"{v:.4f}" for v in r))
