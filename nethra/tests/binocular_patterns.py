"""How many distinct structural source patterns (cosine matching at the given thresholds) a dense
grid of single-object positions over the 500^3 box produces.  No field dynamics.
usage: binocular_patterns.py R grid_step th [th ...]"""
import os, sys, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
A = sys.argv[1:]; sys.argv = [sys.argv[0], A[0]]
import binocular3d as b
import nethra_presence as core
step = int(A[1])
for th in map(float, A[2:]):
    f = core.NethraField(source_similarity_threshold=th)
    for _ in range(2 * b.R * b.R): f.new()
    pts = [(x, y, z) for x in range(0, 500, step) for y in range(0, 500, step) for z in range(0, 500, step)]
    random.Random(1).shuffle(pts)
    counts = []
    for i, p in enumerate(pts):
        src = {f.nethra[b.R * b.R * e + b.R * r + c]: w for e, r, c, w in b.shares(p)}
        f._canonical_source_event(src)
        if (i + 1) in (len(pts) // 8, len(pts) // 4, len(pts) // 2, len(pts)): counts.append(len(f.source_patterns))
    print(f"R={b.R} grid step {step} ({len(pts)} points) th {th}: patterns after 1/8,1/4,1/2,all of points: {counts}", flush=True)
