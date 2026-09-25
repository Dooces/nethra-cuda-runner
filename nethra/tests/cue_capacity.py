"""1 A->B then A->C, with and without a differentiating cue present (cue1 during A->B, cue2 during A->C).
2 capacity/overlap: k regimes whose contexts share features.  Context i = 2 active features chosen from
  a pool of P; the more the pool is shared, the more contexts overlap.  Correct continuation top?"""
import random, itertools, nethra_presence as core
def F(): return core.NethraField(g_min=0.0, admission_seed=14.0)
def show(f, L, xs):
    for x in xs: L[x].push(1.0)
    f.step(1.0)
# 1. 0=A 1=B 2=C 3=cue1 4=cue2 5=Z filler
for cued in (False, True):
    f = F(); L = [f.new() for _ in range(6)]
    c1 = [3] if cued else []; c2 = [4] if cued else []
    for _ in range(40): show(f, L, [0] + c1); show(f, L, [1] + c1); show(f, L, [5])
    for _ in range(40): show(f, L, [0] + c2); show(f, L, [2] + c2); show(f, L, [5])
    f.native_learning = False; res = []
    for cue in ([], [3], [4]) if cued else ([],):
        for _ in range(4): show(f, L, [5])
        show(f, L, [0] + cue); res.append(f"cue {cue or 'none'}: B {L[1].activation:.4f} C {L[2].activation:.4f}")
    print(f"{'with' if cued else 'without'} differentiating cue: " + " | ".join(res))
# 2. capacity with overlapping contexts
for P, k in ((16, 8), (8, 8), (6, 8), (6, 15)):
    pairs = list(itertools.combinations(range(P), 2)); rng = random.Random(7); rng.shuffle(pairs); ctx = pairs[:k]
    K = P + 2 + k; A, B = P, P + 1
    f = F(); L = [f.new() for _ in range(K)]; ok = []
    for blk in range(40 * k):
        c = rng.randrange(k)
        for s in (A, B, P + 2 + c):
            show(f, L, list(ctx[c]) + [s])
            if s == B and blk >= 30 * k:
                acts = [L[P + 2 + j].activation for j in range(k)]; ok.append(acts[c] == max(acts))
    shared = sum(1 for a, b in itertools.combinations(ctx, 2) if set(a) & set(b)) / (k * (k - 1) / 2)
    print(f"{k} regimes from pairs of {P} features (fraction of context pairs sharing a feature {shared:.2f}): "
          f"correct continuation top {sum(ok)/len(ok):.2f}, Nethra {len(f.nethra)}")
