"""Nethra-only extension tests.  One stream, one pass, no labels.  Read: P toward candidates at the named interval
(prior-interval flow); counted when the right Nethra is the unique highest P; reported per quarter of the stream.
  cycles  N symbols in 2 cycles that share one symbol (common factor): c1 = 0..L-1, c2 = L-1 shared + L..2L-2.
          Cycle chosen per pass; at the shared symbol, the next symbol depends on which cycle is running.
  switch  12 cue->outcome pairs.  Phase A: map A; phase B: same cues, permuted outcomes; phase A again.
          Share per 25% of each phase: how fast it switches and returns.
  common  motifs  ctx(3 symbols) + X Y -> outcome; X Y shared by all K motifs; outcome set by the context.
usage: python3 ext.py TEST ARGS"""
import sys, random, time, os
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..")); from nethra import NethraField
f = NethraField(direction=os.environ.get("DIR", "shared")); N = {}; T0 = time.perf_counter()
def node(j):
    if j not in N: N[j] = f.new()
    return N[j]
def see(xs):
    for x in xs: node(x).push(1.0)
    f.step(1.0)
def pick(cands):
    A = f.current_interval_integral; P = {}
    for (r, m), row in f._physical_incidences(f.current_event).items():
        q = row["g"] * (A.get(r, 0.0) - A.get(m, 0.0))
        if q > 0: P[m] = P.get(m, 0.0) + q
    v = [P.get(N.get(c), 0.0) for c in cands]; mx = max(v)
    return cands[v.index(mx)] if mx > 0 and v.count(mx) == 1 else None
def quarters(ok): q = max(1, len(ok) // 4); return " ".join(f"{sum(ok[i:i+q])/len(ok[i:i+q]):.2f}" for i in range(0, len(ok), q))
def cap():
    if time.perf_counter() - T0 > 270: print("stopped at 270 s"); return True
rng = random.Random(1); test = sys.argv[1]
import os; HOLD = os.environ.get('HOLD') == '1'
if test == "cycles":
    L, passes = int(sys.argv[2]), int(sys.argv[3]); S = 50; h = L // 2
    c1 = [0 + i for i in range(L - 1)]; c1.insert(h, S)          # cycle 1, shared symbol S in the middle
    c2 = [20 + i for i in range(L - 1)]; c2.insert(h, S)         # cycle 2, same S at the same place
    allsym = sorted(set(c1 + c2)); ok, oks = [], []
    for p in range(passes):
        cyc = rng.choice((c1, c2)); seq = cyc * 2                  # each pass runs its cycle twice
        for i, s_ in enumerate(seq):
            see([s_])
            if i + 1 < len(seq):
                g = pick(allsym) == seq[i + 1]; ok.append(g)
                if s_ == S: oks.append(g)
        see([999])
        if cap(): break
    print(f"cycles L={L}, 2 cycles sharing S, {p+1} passes: next step right-highest share per quarter {quarters(ok)}; "
          f"after S (needs the cycle) {quarters(oks)}  (chance {1/len(allsym):.2f})  Nethra {len(f.nethra)}  {time.perf_counter()-T0:.0f}s")
if test == "switch":
    K, reps = int(sys.argv[2]), int(sys.argv[3])
    cues = list(range(K)); outs = list(range(100, 100 + K)); mA = dict(zip(cues, outs))
    perm = outs[1:] + outs[:1]; mB = dict(zip(cues, perm))
    for name, mp in (("A", mA), ("B", mB), ("A again", mA)):
        ok = []
        for _ in range(reps * K):
            c = rng.choice(cues); see([c]); ok.append(pick(outs) == mp[c]); see([mp[c]]); see([999])
            if cap(): break
        print(f"  phase {name:8s}: right-highest share per quarter {quarters(ok)}   [{time.perf_counter()-T0:.0f}s]", flush=True)
    print(f"switch K={K}, {reps} per cue per phase, chance {1/K:.2f}, Nethra {len(f.nethra)}")
if test == "common":
    K, reps = int(sys.argv[2]), int(sys.argv[3])
    ctx = [[10 * k + i for i in range(3)] for k in range(K)]; X, Y = 900, 901; outs = [500 + k for k in range(K)]
    ok = []
    for _ in range(reps * K):
        k = rng.randrange(K)
        for s in ctx[k]: see([s])
        hold = ctx[k] if HOLD else []
        see([X] + hold); see([Y] + hold); ok.append(pick(outs) == outs[k]); see([outs[k]]); see([999])
        if cap(): break
    print(f"common factor hold={HOLD}: {K} motifs ctx3+X Y->out, X Y shared: right-highest share after Y per quarter {quarters(ok)} "
          f"(chance {1/K:.2f})  Nethra {len(f.nethra)}  {time.perf_counter()-T0:.0f}s")
