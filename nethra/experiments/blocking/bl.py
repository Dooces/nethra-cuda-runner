"""Blocking factor scan. Base = human.py test 1 (Kamin, separate fields). One factor changed at a time.
  0=A 1=X 2=O 3=Z filler 4=K control cue.  Read: live (human.py: 3 quiet steps, push cue, activation of O)
  and P (flow toward O after the cue interval)."""
import sys, os, time
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")); from nethra import NethraField
def P_toward(f, n):
    A = f.current_interval_integral; p = 0.0
    for (r, m), row in f._physical_incidences(f.current_event).items():
        if m is n:
            q = row["g"] * (A.get(r, 0.0) - A.get(m, 0.0))
            if q > 0: p += q
    return p
def run(variant, pre_n=60, comp_n=30):
    out = {}
    for group in ("blocked", "control"):
        f = NethraField(); L = [f.new() for _ in range(5)]
        def show(xs):
            for x in xs: L[x].push(1.0)
            f.step(1.0)
        pre = 0 if group == "blocked" else 4
        for _ in range(pre_n):
            show([pre]); show([2]); show([3])
        for _ in range(comp_n):
            if variant == "cue_with_outcome": show([0, 1]); show([0, 1, 2]); show([0, 1, 3])
            else: show([0, 1]); show([2]); show([3])
        g = NethraField.from_checkpoint_dict(f.checkpoint_dict()); g.topology_and_evidence_change = False
        Lg = [g.nethra[f._order[n]] for n in L]
        for _ in range(3): g.step(1.0)
        Lg[1].push(1.0); g.step(1.0)
        out[group] = (Lg[2].activation, P_toward(g, Lg[2]))
    b, c = out["blocked"], out["control"]
    return f"live {b[0]:.4f}/{c[0]:.4f} = {b[0]/c[0]:.2f}   P {b[1]:.4f}/{c[1]:.4f} = {b[1]/c[1] if c[1] else float('nan'):.2f}"
if __name__ == "__main__":
  t = time.perf_counter()
  for v in ("base", "cue_with_outcome"):
      for comp in (30, 120):
          print(f"{v:18s} compound {comp:3d}: {run(v, 60, comp)}   [{time.perf_counter()-t:.0f}s]", flush=True)
