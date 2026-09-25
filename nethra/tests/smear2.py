"""Graded smear counting 0..4 (triangular receptive Nethra, width 1.5, so at value v the neighbours
v-1 and v+1 are ALSO sourced, equally, at 1/3).  The previous value v-1 additionally has leftover.
Fair question: does learning raise v+1 relative to v-1?  ratio = a(v+1) / a(v-1), trained vs
never-learned (same inputs), for interior values v = 1, 2, 3."""
import sys, nethra_presence as core
SEED = float(sys.argv[1])
C = [0, 1, 2, 3, 4]
def smear(v): return [max(0.0, 1 - abs(v - c) / 1.5) for c in C]
out = {}
for learn in (True, False):
    f = core.NethraField(g_min=0.0, admission_seed=SEED, topology_and_evidence_change=learn); R = [f.new() for _ in range(5)]
    seq = [0, 1, 2, 3, 4] * 80; ratios = {v: [] for v in (1, 2, 3)}
    for i, v in enumerate(seq):
        for k, j in enumerate(smear(v)):
            if j: R[k].push(j)
        f.step(1.0)
        if i > len(seq) // 2 and v in ratios:
            ratios[v].append(R[v + 1].activation / R[v - 1].activation)
    out[learn] = {v: sum(x) / len(x) for v, x in ratios.items()}
print(f"seed {SEED} a(v+1)/a(v-1) after value v=1,2,3: trained {[round(out[True][v], 3) for v in (1, 2, 3)]}  never-learned {[round(out[False][v], 3) for v in (1, 2, 3)]}")
