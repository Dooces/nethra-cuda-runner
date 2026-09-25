"""Cycles at different periods, one long stream fed several times.  Watch structure build.

Each cycle k has its own Nethra, one per phase (period 3: 3 Nethra, period 5: 5, period 7: 7).
Every interval pushes 1.0 onto the current phase Nethra of every cycle (co-present), then steps.
Nothing else is fed.  The short cycles recur often; the joint pattern recurs only every
lcm(periods) intervals, so its structure can only be constructed after the Nethra that closure
refinds for the short cycles exist.

The stream is STREAM intervals long and is fed PASSES times back to back (the phases restart at
the start of each pass).  Every REPORT intervals, measured over the intervals since the last report:
  Nethra      total Nethra and how many construction added
  refound     mean number of constructed Nethra refound by closure after an interval
  depth       deepest relation-of-relation among the Nethra refound at the report interval
  next share  per cycle: live activation after the interval, over that cycle's phase Nethra that
              were not pushed this interval; the share on the phase pushed in the next interval
              (equal spread would be 1/(period-1))
  ms/int      process time per interval

usage: python3 cycles.py [passes] [stream] [periods, comma-separated] [report]"""
import sys, time, nethra_presence as core
sys.setrecursionlimit(100000)
PASSES = int(sys.argv[1]) if len(sys.argv) > 1 else 3
STREAM = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
PERIODS = [int(p) for p in (sys.argv[3] if len(sys.argv) > 3 else "3,5,7").split(",")]
REPORT = int(sys.argv[4]) if len(sys.argv) > 4 else 500

def depth(n, memo):
    """Relation-of-relation depth through stored routes (0 for a Nethra with no routes)."""
    if n in memo: return memo[n]
    memo[n] = 0
    v = 0 if not n.routes else 1 + max((depth(x, memo) for r in n.routes for x in r), default=0)
    memo[n] = v; return v

f = core.NethraField(g_min=0.0, admission_seed=14.0)
phase_nethra = [[f.new() for _ in range(p)] for p in PERIODS]
print(f"periods {PERIODS}, stream {STREAM} intervals x {PASSES} passes, admission seed 14", flush=True)
for p_i in range(PASSES):
    last_n = len(f.nethra); t0 = time.process_time(); refound = []; share = [[] for _ in PERIODS]
    for t in range(STREAM):
        for k, p in enumerate(PERIODS): phase_nethra[k][t % p].push(1.0)
        f.step(1.0)
        refound.append(sum(1 for n in f.previous_closure if n.routes))
        for k, p in enumerate(PERIODS):
            others = [q for q in range(p) if q != t % p]
            acts = {q: max(0.0, phase_nethra[k][q].activation) for q in others}
            total = sum(acts.values())
            share[k].append(acts[(t + 1) % p] / total if total > 0 else 0.0)
        if (t + 1) % REPORT == 0:
            now = time.process_time(); memo = {}
            d = max((depth(n, memo) for n in f.previous_closure if n.routes), default=0)
            shares = "  ".join(f"p{p} {sum(s) / len(s):.2f}" for p, s in zip(PERIODS, share))
            print(f"pass {p_i + 1} t {t + 1:5d}: Nethra {len(f.nethra):5d} (+{len(f.nethra) - last_n:4d})  "
                  f"refound {sum(refound) / len(refound):6.1f}  depth {d:3d}  next share {shares}  "
                  f"{1000 * (now - t0) / REPORT:6.1f} ms/int", flush=True)
            last_n = len(f.nethra); t0 = now; refound = []; share = [[] for _ in PERIODS]
