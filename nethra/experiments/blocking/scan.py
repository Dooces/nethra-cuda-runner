import sys, os, time
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")); from nethra import NethraField
from bl import P_toward
KW = eval(os.environ.get("KW", "{}"))
def run(comp_list):
    res = {}
    for group in ("blocked", "control"):
        f = NethraField(**KW); L = [f.new() for _ in range(5)]
        def show(xs):
            for x in xs: L[x].push(1.0)
            f.step(1.0)
        pre = 0 if group == "blocked" else 4
        for _ in range(60): show([pre]); show([2]); show([3])
        done = 0
        for c in comp_list:
            for _ in range(c - done): show([0, 1]); show([2]); show([3])
            done = c
            g = NethraField.from_checkpoint_dict(f.checkpoint_dict()); g.topology_and_evidence_change = False
            Lg = [g.nethra[f._order[n]] for n in L]
            for _ in range(3): g.step(1.0)
            Lg[1].push(1.0); g.step(1.0)
            xl, xp = Lg[2].activation, P_toward(g, Lg[2])
            g = NethraField.from_checkpoint_dict(f.checkpoint_dict()); g.topology_and_evidence_change = False
            Lg = [g.nethra[f._order[n]] for n in L]
            for _ in range(3): g.step(1.0)
            Lg[0 if group == "blocked" else 4].push(1.0); g.step(1.0)
            res[(group, c)] = (xl, xp, Lg[2].activation)
    return res
t = time.perf_counter(); C = [30, 120, 400]; r = run(C)
print(f"KW {KW}  [{time.perf_counter()-t:.0f}s]")
for c in C:
    b, k = r[("blocked", c)], r[("control", c)]
    print(f"  compound {c:3d}: X->O live {b[0]:.4f}/{k[0]:.4f} = {b[0]/k[0]:.2f}  P {b[1]:.4f}/{k[1]:.4f} = {b[1]/k[1] if k[1] else float('nan'):.2f}   pretrained cue->O live {b[2]:.4f}/{k[2]:.4f}")
