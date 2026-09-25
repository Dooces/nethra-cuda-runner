"""Tests with known human/animal signatures.  Contract core unchanged: g_min 0, one push per present
Nethra per interval, exposure only, no labels; readout = live field.  Filler Nethra Z separates trials.
1 blocking (Kamin): pretrain A->O, then AX->O.  Does X alone prime O less than after AX->O alone?
2 interference: A->B, then A->C.  After A: B vs C (recency; is the old one still there?)
3 spacing: same number of P->Q (massed) and R->S (spaced) trials, both ending at the same time,
  then unrelated material.  Which association primes more?
4 configural learning: XOR of two contexts vs a simple discrimination, learning curve."""
import sys, random, nethra_presence as core
SEED = float(sys.argv[1])
m = lambda xs: sum(xs) / len(xs) if xs else float('nan')
def F(): return core.NethraField(g_min=0.0, admission_seed=SEED)
def show(f, L, items):
    for x in items: L[x].push(1.0)
    f.step(1.0)
def probe(f, L, cue, read, quiet=3):
    f.topology_and_evidence_change = False
    for _ in range(quiet): f.step(1.0)
    show(f, L, cue)
    return [L[r].activation for r in read]

# 1 blocking.  0=A 1=X 2=O 3=Z(filler) 4=K (control pretraining cue)
res = {}
for group in ("blocked", "control"):
    f = F(); L = [f.new() for _ in range(5)]
    pre = 0 if group == "blocked" else 4
    for _ in range(60): show(f, L, [pre]); show(f, L, [2]); show(f, L, [3])
    for _ in range(30): show(f, L, [0, 1]); show(f, L, [2]); show(f, L, [3])
    res[group] = probe(f, L, [1], [2])[0]
print(f"seed {SEED} blocking: O primed by X alone  blocked {res['blocked']:.4f}  control {res['control']:.4f}  "
      f"(blocked/control = {res['blocked']/res['control']:.2f})")

# 2 interference.  0=A 1=B 2=C 3=Z 4..7 unrelated
f = F(); L = [f.new() for _ in range(8)]
for _ in range(40): show(f, L, [0]); show(f, L, [1]); show(f, L, [3])
for _ in range(40): show(f, L, [0]); show(f, L, [2]); show(f, L, [3])
b, c = probe(f, L, [0], [1, 2])
rng = random.Random(3); f.topology_and_evidence_change = True
for _ in range(150): show(f, L, [rng.randrange(4, 8)])
b2, c2 = probe(f, L, [0], [1, 2])
print(f"seed {SEED} interference A->B then A->C: after A  B {b:.4f}  C {c:.4f}  | after 150 unrelated intervals  B {b2:.4f}  C {c2:.4f}")

# 3 spacing.  0=P 1=Q 2=R 3=S 4=Z 5..9 unrelated
f = F(); L = [f.new() for _ in range(10)]; rng = random.Random(4)
timeline = []
for t in range(30):                               # spaced: R->S every 10th slot over 300 slots
    timeline += [("R",)] + [("u",)] * 9
timeline = timeline[:-30] + [("P",)] * 30         # massed P->Q in the last 30 slots
for slot in timeline:
    if slot[0] == "R": show(f, L, [2]); show(f, L, [3]); show(f, L, [4])
    elif slot[0] == "P": show(f, L, [0]); show(f, L, [1]); show(f, L, [4])
    else: show(f, L, [rng.randrange(5, 10)])
q_now, = probe(f, L, [0], [1]); s_now, = probe(f, L, [2], [3])
f.topology_and_evidence_change = True
for _ in range(200): show(f, L, [rng.randrange(5, 10)])
q_late, = probe(f, L, [0], [1]); s_late, = probe(f, L, [2], [3])
print(f"seed {SEED} spacing (27 spaced R->S vs 30 massed P->Q, ending together): immediately  massed {q_now:.4f} spaced {s_now:.4f}"
      f" | after 200 unrelated intervals  massed {q_late:.4f} spaced {s_late:.4f}")

# 4 learning curves: simple (context P|Q decides C|D) vs XOR (P|Q with X|Y decides)
def curve(xor):
    f = F(); L = [f.new() for _ in range(8)]; rng = random.Random(9); ok = []; out = []
    for blk in range(900):
        o, i = rng.choice([0, 1]), rng.choice([2, 3])
        tail = (6 if (o == 0) == (i == 2) else 7) if xor else (6 if o == 0 else 7)
        for s in (4, 5, tail):
            L[o].push(1.0); L[i].push(1.0); L[s].push(1.0); f.step(1.0)
            if s == 5:
                good, bad = (6, 7) if tail == 6 else (7, 6)
                ok.append(L[good].activation > L[bad].activation)
        if (blk + 1) % 150 == 0: out.append(round(m(ok), 2)); ok = []
    return out
print(f"seed {SEED} learning curve, correct continuation primed more, per 150 blocks:  simple {curve(False)}  XOR {curve(True)}")
