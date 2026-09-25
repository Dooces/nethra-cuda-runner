"""Robustness and confidence.  World: k regimes; context = 2 features from a pool of P; block =
[ctx+A], [ctx+B], [ctx+T_c].  After training, probe (learning off) with:
  clean   : both context features
  partial : only ONE of the two context features
  noisy   : both features + 1 random distractor feature
  noisy2  : ONE feature + 2 distractors
Report: correct continuation is the unique top (accuracy) and, for Nethra, confidence readouts:
  share = top candidate activation / sum over candidates; total = sum over candidates (familiarity).
Baselines as in baselines.py (count table HEB, Rescorla-Wagner RW, configural lookup CFG)."""
import random, itertools, statistics as st, nethra_presence as core
from baselines import HEB, RW, CFG
P, k = 10, 12
pairs = list(itertools.combinations(range(P), 2)); rng0 = random.Random(7); rng0.shuffle(pairs); ctx = pairs[:k]
A, B = P, P + 1; T = [P + 2 + j for j in range(k)]; K = P + 2 + k
def probe_sets(c, cond, rng):
    a, b = ctx[c]
    others = [x for x in range(P) if x not in ctx[c]]
    if cond == "clean": return [a, b]
    if cond == "partial": return [rng.choice([a, b])]
    if cond == "noisy": return [a, b, rng.choice(others)]
    if cond == "noisy2": return [rng.choice([a, b])] + rng.sample(others, 2)
CONDS = ("clean", "partial", "noisy", "noisy2")
def run_nethra():
    f = core.NethraField(); L = [f.new() for _ in range(K)]; rng = random.Random(11)
    for _ in range(30 * k):
        c = rng.randrange(k)
        for s in (A, B, T[c]):
            for x in ctx[c] + (s,): L[x].push(1.0)
            f.step(1.0)
    f.topology_and_evidence_change = False; out = {}
    for cond in CONDS:
        rows = []
        for rep in range(4):
            for c in range(k):
                feats = probe_sets(c, cond, rng)
                for _ in range(3): f.step(1.0)                       # let leftovers fade
                for s in (A, B):
                    for x in feats + [s]: L[x].push(1.0)
                    f.step(1.0)
                acts = [L[t].activation for t in T]; tot = sum(acts); top = max(acts)
                rows.append((acts[c] == top and acts.count(top) == 1, top / tot if tot else 0, tot))
        out[cond] = rows
    return out
def run_base(M):
    m = M(); rng = random.Random(11)
    for _ in range(30 * k):
        c = rng.randrange(k)
        for s in (A, B, T[c]): m.see(list(ctx[c]) + [s])
    out = {}
    for cond in CONDS:
        ok = []
        for rep in range(4):
            for c in range(k):
                feats = probe_sets(c, cond, rng)
                m.see([], learn=False); m.see(feats + [A], learn=False); m.see(feats + [B], learn=False)
                e = [m.expect(t) for t in T]; top = max(e)
                ok.append(e[c] == top and e.count(top) == 1 and top > 0)
        out[cond] = sum(ok) / len(ok)
    return out
if __name__ == "__main__":
    res = run_nethra()
    print("accuracy (correct continuation is the unique top):")
    print("  Nethra          " + "  ".join(f"{c} {sum(r[0] for r in res[c])/len(res[c]):.2f}" for c in CONDS))
    for name, M in (("count table", HEB), ("Rescorla-Wagner", lambda: RW(universe=K)), ("configural", CFG)):
        o = run_base(M); print(f"  {name:15s} " + "  ".join(f"{c} {o[c]:.2f}" for c in CONDS))
    print("Nethra confidence readouts (mean):")
    for c in CONDS:
        r = res[c]
        right = [x[1] for x in r if x[0]]; wrong = [x[1] for x in r if not x[0]]
        print(f"  {c:8s} share {st.mean(x[1] for x in r):.3f} (when right {st.mean(right) if right else float('nan'):.3f}, "
              f"when wrong {st.mean(wrong) if wrong else float('nan'):.3f})   familiarity(total) {st.mean(x[2] for x in r):.4f}")
