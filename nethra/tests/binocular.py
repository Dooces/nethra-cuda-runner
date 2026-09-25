"""Binocular 1D retina with moving objects.  Each eye has W receptive Nethra (positions).  An object
of width 2 at position x is seen by the left eye at x and by the right eye at x + d (d = disparity =
depth cue).  Objects move one position per interval (left or right) and bounce; depth d in {0,1,2}
changes occasionally.  Numbers only: each receptive Nethra seeing the object is pushed 1.0.
Measured: Nethra growth, cost per interval, how many Nethra are active per interval (frontier), and
anticipation: after an interval, is the object's NEXT position (left eye) the most primed
non-currently-seen left-eye Nethra?"""
import sys, time, random, nethra_presence as core
W = int(sys.argv[1]); T = int(sys.argv[2]); BUDGET = float(sys.argv[3])
rng = random.Random(3)
f = core.NethraField(g_min=0.0, admission_seed=14.0)
left = [f.new() for _ in range(W)]; right = [f.new() for _ in range(W + 2)]
x, v, d = W // 2, 1, 1
t0 = time.process_time(); rows = []; hits = []; last = t0; lastN = len(f.nethra)
for t in range(1, T + 1):
    seen_l = [x, x + 1]; seen_r = [x + d, x + 1 + d]
    for p in seen_l: left[p].push(1.0)
    for p in seen_r: right[p].push(1.0)
    f.step(1.0)
    nx = x + v
    if nx < 0 or nx + 1 >= W: v = -v; nx = x + v
    if t > T // 2:
        cand = {p: left[p].activation for p in range(W) if p not in seen_l}
        want = [p for p in (nx, nx + 1) if p not in seen_l]
        hits.append(max(cand, key=cand.get) in want)
    x = nx
    if rng.random() < 0.05: d = rng.randrange(3)
    if t % 150 == 0:
        now = time.process_time()
        active = sum(1 for n in f.nethra if abs(n.activation) > 1e-3)
        rows.append(f"t{t}: Nethra {len(f.nethra)} (+{len(f.nethra)-lastN}), active>1e-3 {active}, {1000*(now-last)/150:.0f} ms/int")
        last = now; lastN = len(f.nethra)
    if time.process_time() - t0 > BUDGET: break
print(f"W={W} (inputs {2*W+2}): " + " | ".join(rows))
if hits: print(f"   next left-eye position is the most primed unseen left-eye Nethra in {sum(hits)/len(hits):.2f} of intervals (second half)")
