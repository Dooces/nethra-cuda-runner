"""Bigger alphabet, context selection with a common factor.  P features; every pair is a context (k regimes),
each context -> its own outcome.  A and B (common factor) present in every block.  Read: P toward each outcome
at the B interval, last quarter of blocks; counted when the context's own outcome Nethra has the unique highest P."""
import sys, os, random, itertools, time
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..")); from nethra import NethraField
P_, K_, REP = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
pairs = list(itertools.combinations(range(P_), 2)); rng = random.Random(7); rng.shuffle(pairs); ctx = pairs[:K_]
f = NethraField(); N = {}
def node(j):
    if j not in N: N[j] = f.new()
    return N[j]
def see(xs):
    for x in xs: node(x).push(1.0)
    f.step(1.0)
def Pread():
    A = f.current_interval_integral; P = {}
    for (r, m), row in f._physical_incidences(f.current_event).items():
        q = row["g"] * (A.get(r, 0.0) - A.get(m, 0.0))
        if q > 0: P[m] = P.get(m, 0.0) + q
    return P
A, B = 1000, 1001; ok = []; t = time.perf_counter(); nb = REP * K_
for blk in range(nb):
    c = rng.randrange(K_)
    for s in (A, B, 2000 + c):
        see(list(ctx[c]) + [s])
        if s == B and blk >= nb * 3 // 4:
            P = Pread(); e = [P.get(N.get(2000 + j), 0.0) for j in range(K_)]
            ok.append(e[c] == max(e) and e.count(max(e)) == 1)
    if time.perf_counter() - t > 280: print("stopped at 280 s, block", blk); break
print(f"features {P_}  regimes {K_}  blocks {blk+1}  right-highest share {sum(ok)/max(1,len(ok)):.2f} (chance {1/K_:.2f})  "
      f"Nethra {len(f.nethra)}  {1000*(time.perf_counter()-t)/(3*(blk+1)):.1f} ms/interval")
