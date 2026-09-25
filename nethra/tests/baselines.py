"""Is Nethra doing anything a trivial learner cannot?  Two textbook baselines on the same worlds,
same presentation (set of present symbols per interval), read the same way:
  HEB: counts W[i,j] += 1 whenever i is present at t and j at t+1; expectation for j = sum over
       present i of W[i,j]   (a first-order associative count table, no structure)
  RW : Rescorla-Wagner / delta rule: W[i,j] += a * (y_j - sum_present W[.,j]) for present i,
       y_j = 1 if j is present next, else 0   (error-driven linear associator)"""
import random, itertools
from collections import defaultdict
class HEB:
    def __init__(s): s.W = defaultdict(float); s.prev = ()
    def see(s, present, learn=True):
        if learn:
            for i in s.prev:
                for j in present: s.W[(i, j)] += 1.0
        s.prev = tuple(present)
    def expect(s, j): return sum(s.W[(i, j)] for i in s.prev)
class RW(HEB):
    def __init__(s, a=0.1, universe=64): super().__init__(); s.a = a; s.U = range(universe)
    def see(s, present, learn=True):
        if learn and s.prev:
            pres = set(present)
            for j in s.U:
                err = (1.0 if j in pres else 0.0) - sum(s.W[(i, j)] for i in s.prev)
                if err:
                    for i in s.prev: s.W[(i, j)] += s.a * err
        s.prev = tuple(present)
def regimes_overlap(M, P=6, k=15):
    pairs = list(itertools.combinations(range(P), 2)); rng = random.Random(7); rng.shuffle(pairs); ctx = pairs[:k]
    A, B = P, P + 1; m = M(); ok = []
    for blk in range(40 * k):
        c = rng.randrange(k)
        for s in (A, B, P + 2 + c):
            m.see(list(ctx[c]) + [s])
            if s == B and blk >= 30 * k:
                e = [m.expect(P + 2 + j) for j in range(k)]; ok.append(e[c] == max(e) and e.count(max(e)) == 1)
    return sum(ok) / len(ok)
def continual(M, k=4):
    rng = random.Random(1); A, B = 2 * k, 2 * k + 1; m = M()
    def blk(c, learn=True, read=False):
        r = None
        for s in (A, B, 2 * k + 2 + c):
            m.see([c, s], learn)
            if read and s == B:
                e = [m.expect(2 * k + 2 + j) for j in range(2 * k)]; r = e[c] == max(e) and e.count(max(e)) == 1
        return r
    for _ in range(40 * k): blk(rng.randrange(k))
    for _ in range(100 * k): blk(k + rng.randrange(k))
    return sum(blk(c, False, True) for c in range(k)) / k
def blocking(M):
    res = {}
    for g in ("blocked", "control"):
        m = M(); pre = 0 if g == "blocked" else 4
        for _ in range(60): m.see([pre]); m.see([2]); m.see([3])
        for _ in range(30): m.see([0, 1]); m.see([2]); m.see([3])
        m.see([1], learn=False); res[g] = m.expect(2)
    return res["blocked"] / res["control"] if res["control"] else float('nan')
def xor(M):
    m = M(); rng = random.Random(9); ok = []
    for blk in range(900):
        o, i = rng.choice([0, 1]), rng.choice([2, 3]); tail = 6 if (o == 0) == (i == 2) else 7
        for s in (4, 5, tail):
            m.see([o, i, s])
            if s == 5 and blk >= 750: ok.append(m.expect(tail) > m.expect(13 - tail))
    return sum(ok) / len(ok)
if __name__ == "__main__":
  for name, M in (("HEB counts", HEB), ("Rescorla-Wagner", RW)):
    print(f"{name:16s}: 15 overlapping regimes {regimes_overlap(M):.2f} | continual (old set after 400 new-only blocks) {continual(M):.2f} | "
          f"blocking ratio {blocking(M):.2f} | XOR last 150 blocks {xor(M):.2f}")
  print("Nethra (from tests): 15 overlapping regimes 0.83 | continual 1.00 | blocking ratio 0.40 (seed 14) | XOR 0.43-0.65")

class CFG(HEB):
    """configural lookup: key = the whole set of present symbols"""
    def see(s, present, learn=True):
        if learn and s.prev:
            for j in present: s.W[(frozenset(s.prev), j)] += 1.0
        s.prev = tuple(present)
    def expect(s, j): return s.W[(frozenset(s.prev), j)]
def completion(M):
    """objects shown back to back; cue with part of an object; is the missing member expected?"""
    objs = [[0, 1, 2], [3, 4, 5, 9], [6, 7, 8, 9]]; m = M(); rng = random.Random(2)
    for _ in range(150): m.see(rng.choice(objs))
    m.see([3, 4], learn=False)
    e = {j: m.expect(j) for j in range(10) if j not in (3, 4)}
    return max(e, key=e.get) == 5 and e[5] > 0
print(f"configural table: 15 overlapping regimes {regimes_overlap(CFG):.2f} | continual {continual(CFG):.2f} | blocking ratio {blocking(CFG):.2f} | XOR {xor(CFG):.2f} | partial-cue completion {completion(CFG)}")
if __name__ == "__main__":
  for name, M in (("HEB counts", HEB), ("Rescorla-Wagner", RW)):
      print(f"{name}: partial-cue completion {completion(M)}")
