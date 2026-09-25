"""Sub-pattern transfer.  Is a new next interval after an already-experienced sequence
carried faster than the same amount of experience with its parts, or no experience at all?

Phase 1 (exposure only):
  chunk     Q R S pushed in consecutive intervals, inside context C1 or C2, between random fillers
  elements  Q2, R2, S2 pushed equally often, each alone between fillers, never as Q2 R2 S2
  novel     Q3 R3 S3 never pushed
Phase 2 (new context C3 co-present): each sequence is followed in the next interval by its own new Nethra T, T2, T3.
After every phase-2 exposure a copy of the field is probed with learning off: the sequence is
pushed again in C3 and the live activation of T, T2, T3 is read.  Also read: how many
constructed Nethra are refound at S (last interval of the sequence) (closure).

Flat pairwise associations only see S->T, S2->T2, S3->T3, so they have no way to prefer the chunk
over the elements.  Reference: count table, Rescorla-Wagner, configural lookup from baselines.py,
same stream, read the same way.

Variant "bare": phase 1 without C1/C2, so phase-1 routes do not contain a context.

usage: python3 transfer.py [admission_seed] [world_seeds] [ctx|bare]"""
import sys, random, statistics as st, nethra_presence as core
SEED = float(sys.argv[1]) if len(sys.argv) > 1 else 14.0
WORLDS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
PHASE1_CTX = (sys.argv[3] if len(sys.argv) > 3 else "ctx") == "ctx"
Q, R, S, Q2, R2, S2, Q3, R3, S3, T, T2, T3, C1, C2, C3 = range(15)
FILL = list(range(15, 21)); K = 21
SEQS = {"chunk": (Q, R, S, T), "elements": (Q2, R2, S2, T2), "novel": (Q3, R3, S3, T3)}
TRIALS, EXPOSURES = 120, 8

def phase1(rng):
    """List of present-sets, one per interval."""
    out = []
    for i in range(TRIALS):
        ctx = ([C1] if i % 2 == 0 else [C2]) if PHASE1_CTX else []
        out += [[rng.choice(FILL)] + ctx] + [[x] + ctx for x in (Q, R, S)] + [[rng.choice(FILL)] + ctx]
        parts = [Q2, R2, S2]; rng.shuffle(parts)
        for x in parts: out += [[rng.choice(FILL)], [x]]
        out += [[rng.choice(FILL)]]
    return out

def phase2_block(rng):
    order = list(SEQS); rng.shuffle(order); out = []
    for name in order:
        a, b, c, t = SEQS[name]
        out += [[rng.choice(FILL), C3], [a, C3], [b, C3], [c, C3], [t, C3]]
    return out

def show(f, L, xs):
    for x in xs: L[x].push(1.0)
    f.step(1.0)

def probe_nethra(f):
    g = core.NethraField.from_checkpoint_dict(f.checkpoint_dict()); g.native_learning = False
    L = g.nethra[:K]; row = {}
    for name, (a, b, c, t) in SEQS.items():
        for _ in range(3): g.step(1.0)
        for x in (a, b, c): show(g, L, [x, C3])
        acts = [L[o].activation for o in (T, T2, T3)]
        refound = sum(1 for n in g.previous_closure if n.routes)
        row[name] = (L[t].activation, L[t].activation / sum(acts) if sum(acts) > 0 else 0.0, refound)
    return row

def run_nethra(ws):
    rng = random.Random(ws); f = core.NethraField(g_min=0.0, admission_seed=SEED); L = [f.new() for _ in range(K)]
    for xs in phase1(rng): show(f, L, xs)
    n1 = len(f.nethra); rows = [probe_nethra(f)]
    for _ in range(EXPOSURES):
        for xs in phase2_block(rng): show(f, L, xs)
        rows.append(probe_nethra(f))
    return rows, n1, len(f.nethra)

def run_base(M, ws):
    rng = random.Random(ws); m = M()
    def probe():
        row = {}
        for name, (a, b, c, t) in SEQS.items():
            m.see([], learn=False)
            for x in (a, b, c): m.see([x, C3], learn=False)
            e = [m.expect(o) for o in (T, T2, T3)]
            row[name] = (m.expect(t), m.expect(t) / sum(e) if sum(e) > 0 else 0.0)
        return row
    for xs in phase1(rng): m.see(xs)
    rows = [probe()]
    for _ in range(EXPOSURES):
        for xs in phase2_block(rng): m.see(xs)
        rows.append(probe())
    return rows

def table(label, runs, k, fmt):
    for name in SEQS:
        vals = [fmt.format(st.mean(r[e][name][k] for r in runs)) for e in range(EXPOSURES + 1)]
        print(f"  {label:9s} {name:9s} " + " ".join(vals))

if __name__ == "__main__":
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        from baselines import HEB, RW, CFG                      # baselines.py prints on import
    res = [run_nethra(ws) for ws in range(WORLDS)]
    runs = [r[0] for r in res]
    print(f"phase 1 {'with' if PHASE1_CTX else 'without'} context, admission seed {SEED}, {WORLDS} world seeds; columns = after 0..{EXPOSURES} phase-2 exposures")
    print(f"Nethra count after phase 1 {st.mean(r[1] for r in res):.0f}, after phase 2 {st.mean(r[2] for r in res):.0f}")
    print("Nethra, activation of its own next Nethra after the sequence:"); table("act", runs, 0, "{:.4f}")
    print("Nethra, share of its own next Nethra among T, T2, T3:"); table("share", runs, 1, "{:.2f}")
    print("Nethra, constructed Nethra refound at S (last interval of the sequence):"); table("refound", runs, 2, "{:5.1f}")
    for bname, M in (("count", HEB), ("RW", lambda: RW(universe=K)), ("config", CFG)):
        b = [run_base(M, ws) for ws in range(WORLDS)]
        print(f"{bname}, share of its own next Nethra:"); table(bname, b, 1, "{:.2f}")
