"""Binocular vision of point objects moving in a 500 x 500 x 500 box, small factors first.

World: point objects, integer positions, constant integer velocity, reflected at the walls.
Eyes: two pinhole eyes at z = -500, x = 250 -/+ 30, y = 250, looking along +z; the box's near face
fills the image.  Each eye has an R x R retina of receptive Nethra with overlapping tent fields one
grid spacing wide; an object pushes each receptive Nethra it falls on with its share (shares of one
object sum to 1 per eye).  Nothing else is pushed.

Stream (one seed), small factors first (CLAUDE.md section 4):
  A  left eye only, one object        B  right eye only, one object
  C  both eyes, one object            D  both eyes, two objects
Control: a separate field that gets C for len(A)+len(B)+len(C) intervals, then D.

Reads after step t, before t+1 is pushed (per eye, object 0):
  P         prior flow the field carries toward each retinal Nethra for the next step (the P that
            step t+1 subtracts from the manifestation M).  "P lead": P centroid over the read window
            minus object 0's image position, along the motion, in grid spacings.  "next/prev": P
            sampled at object 0's next image point over its previous one (through the tents); same
            for live activation, and for activation of the reference field.
  lead      activation-weighted image position over the receptive Nethra within 2 grid spacings of
            object 0's image, minus object 0's image position at t, along its image motion, in grid
            spacings.  Negative = behind (leftover), positive = ahead.  "ref" is the same read on a
            field fed the same stream with topology_and_evidence_change=False (leftover only).
  refound   constructed Nethra refound under current topology after step t, split by the phase that
            built them.
  internal state  the refound constructed Nethra built from one object (phases A-C): each has a 3D
            location from the graded source pattern the field indexed it by (share-weighted image
            position per eye through the receptive centres, then triangulation; Nethra from one eye
            only have none).  Read: how
            often something located is refound, what share of it lies within 50 world units of an
            object, and (one object) the distance from the object to the centre of it.
  occlusion (end of C and of D) on frozen copies (evidence change off): object 0 is no longer
            pushed for 5 intervals while it keeps moving; each interval the same local read is taken
            around where object 0 now is: its offset from the hidden object along the motion, and
            the local activation mass left.
Cost: wall time per interval, one numeric thread, frontier tolerance TOL (declared approximation;
0 = exact).

usage: python3 binocular3d.py [R] [A] [B] [C] [D] [TOL]"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys, time, random, json
import nethra_presence as core

R = int(sys.argv[1]) if len(sys.argv) > 1 else 16
LEN = {k: int(sys.argv[i]) if len(sys.argv) > i else d for k, i, d in (("A", 2, 600), ("B", 3, 600), ("C", 4, 1200), ("D", 5, 600))}
TOL = float(sys.argv[6]) if len(sys.argv) > 6 else 0.01
SEED = 1
WORLD, BASE, DIST = 500, 60.0, 500.0
FOCAL = DIST / (WORLD / 2 + BASE / 2)
EYES = (WORLD / 2 - BASE / 2, WORLD / 2 + BASE / 2)
H = 2.0 / (R - 1)
CENTRE = [-1.0 + i * H for i in range(R)]


def project(p, ex):
    x, y, z = p
    return FOCAL * (x - ex) / (z + DIST), FOCAL * (y - WORLD / 2) / (z + DIST)


def tents(u):
    s = (u + 1.0) / H; i = int(s // 1); frac = s - i
    return [(k, w) for k, w in ((i, 1.0 - frac), (i + 1, frac)) if 0 <= k < R and w > 0.0]


def shares(p):
    out = []
    for e, ex in enumerate(EYES):
        u, v = project(p, ex)
        for c, wu in tents(u):
            for r, wv in tents(v):
                out.append((e, r, c, wu * wv))
    return out


class Mover:
    def __init__(self, pos, vel):
        self.p = list(pos); self.v = list(vel)

    def peek(self):
        q = []
        for k in range(3):
            n = self.p[k] + self.v[k]
            if n < 0 or n > WORLD - 1: n = self.p[k] - self.v[k]
            q.append(n)
        return tuple(q)

    def step(self):
        for k in range(3):
            n = self.p[k] + self.v[k]
            if n < 0 or n > WORLD - 1:
                self.v[k] = -self.v[k]; n = self.p[k] + self.v[k]
            self.p[k] = n
        return tuple(self.p)


def local_read(f, e, p):
    """Activation-weighted image position within 2 spacings of p's image in eye e, and the mass."""
    u0, v0 = project(p, EYES[e])
    c0 = round((u0 + 1.0) / H); r0 = round((v0 + 1.0) / H)
    su = sv = tot = 0.0
    for r in range(max(0, r0 - 2), min(R, r0 + 3)):
        for c in range(max(0, c0 - 2), min(R, c0 + 3)):
            a = f.nethra[e * R * R + r * R + c].activation
            if a > 0.0:
                su += a * CENTRE[c]; sv += a * CENTRE[r]; tot += a
    return (su / tot, sv / tot, tot) if tot > 0 else (float("nan"), float("nan"), 0.0)


def near(e, p, others):
    """Is another object's image within 3 grid spacings of p's image in eye e?"""
    u0, v0 = project(p, EYES[e])
    for o in others:
        u, v = project(o, EYES[e])
        if max(abs(u - u0), abs(v - v0)) < 3 * H: return True
    return False


def along(f, e, p, q, others=()):
    """Offset of the local read around p from p's image, along the image motion p -> q, in spacings.
    Not read (nan) while another object's image is within 3 spacings in that eye."""
    if near(e, p, others): return float("nan"), float("nan")
    u0, v0 = project(p, EYES[e]); u1, v1 = project(q, EYES[e])
    du, dv = u1 - u0, v1 - v0; norm = (du * du + dv * dv) ** .5
    uu, vv, mass = local_read(f, e, p)
    if norm == 0 or mass == 0: return float("nan"), mass
    return ((uu - u0) * du + (vv - v0) * dv) / norm / H, mass


def window(e, p):
    u0, v0 = project(p, EYES[e]); c0 = round((u0 + 1.0) / H); r0 = round((v0 + 1.0) / H)
    return [(r, c) for r in range(max(0, r0 - 2), min(R, r0 + 3)) for c in range(max(0, c0 - 2), min(R, c0 + 3))]


def prior_flow(f, cells):
    """P toward the given Nethra for the next step: sum over their incidences of
    max(0, g (A_relation - A_cell)), A = this interval's activation integrals (as nethra.py does)."""
    cells = set(cells); within = set(cells)
    for m in cells:
        for rel, _route in f.member_to_routeuses.get(m, ()):
            within.add(rel)
    A = f.current_interval_integral; P = {}
    for (rel, mem), row in f._physical_incidences(f.current_event, within=within).items():
        if mem in cells:
            q = row["g"] * (A.get(rel, 0.0) - A.get(mem, 0.0))
            if q > 0.0: P[mem] = P.get(mem, 0.0) + q
    return P


def point_value(value, e, p):
    """Field quantity at p's image point in eye e, through the receptive tents."""
    u, v = project(p, EYES[e])
    return sum(wu * wv * value(r, c) for c, wu in tents(u) for r, wv in tents(v))


def ahead(f, e, p, q, o, others=(), with_p=True):
    """(P lead, P next/prev, activation next/prev) for object 0 in eye e: P centroid over the read
    window minus p's image along the motion (spacings); field sampled at the next image point q over
    the previous one o.  nan while another object is within 3 spacings."""
    nan = float("nan")
    if near(e, p, others) or near(e, q, others) or near(e, o, others): return nan, nan, nan
    cell = lambda r, c: f.nethra[e * R * R + r * R + c]
    act = lambda r, c: max(0.0, cell(r, c).activation)
    a_prev = point_value(act, e, o)
    a_ratio = point_value(act, e, q) / a_prev if a_prev > 0 else nan
    if not with_p: return nan, nan, a_ratio
    win = window(e, p); P = prior_flow(f, [cell(r, c) for r, c in win])
    pv = lambda r, c: P.get(cell(r, c), 0.0)
    tot = sum(pv(r, c) for r, c in win)
    u0, v0 = project(p, EYES[e]); u1, v1 = project(q, EYES[e]); du, dv = u1 - u0, v1 - v0; norm = (du * du + dv * dv) ** .5
    if tot > 0 and norm > 0:
        uu = sum(pv(r, c) * CENTRE[c] for r, c in win) / tot; vv = sum(pv(r, c) * CENTRE[r] for r, c in win) / tot
        p_lead = ((uu - u0) * du + (vv - v0) * dv) / norm / H
    else:
        p_lead = nan
    p_prev = point_value(pv, e, o)
    p_ratio = point_value(pv, e, q) / p_prev if p_prev > 0 else nan
    return p_lead, p_ratio, a_ratio


def pattern_point(pattern, idx):
    """3D point of one stored graded source pattern: share-weighted image position per eye (exact
    for tent fields), then triangulation.  None unless both eyes are in the pattern."""
    sums = [[0.0, 0.0, 0.0] for _ in range(2)]
    for m, w in pattern:
        i = idx[m]
        if i < 2 * R * R:
            e, rc = divmod(i, R * R); r, c = divmod(rc, R)
            sums[e][0] += w * CENTRE[c]; sums[e][1] += w * CENTRE[r]; sums[e][2] += w
    if not (sums[0][2] > 0 and sums[1][2] > 0): return None
    uL, vL = sums[0][0] / sums[0][2], sums[0][1] / sums[0][2]
    uR, vR = sums[1][0] / sums[1][2], sums[1][1] / sums[1][2]
    d = uL - uR
    if d <= 0: return None
    zD = FOCAL * (EYES[1] - EYES[0]) / d
    return (uL * zD / FOCAL + EYES[0], (vL + vR) / 2 * zD / FOCAL + WORLD / 2, zD - DIST)


def location(n, cache):
    """3D location of a constructed Nethra: mean point of the 'after' patterns of the source
    transitions the field indexed it by.  None if those patterns are not in both eyes."""
    f = n._field
    pairs = f.relation_source_events.get(n, ())
    hit = cache.get(n)
    if hit is not None and hit[0] == len(pairs): return hit[1]
    pts = [q for q in (pattern_point(after, f._order) for _before, after in pairs) if q is not None]
    out = tuple(sum(q[k] for q in pts) / len(pts) for k in range(3)) if pts else None
    cache[n] = (len(pairs), out)
    return out


def dist(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** .5


def build_stream(rng):
    def rnd_mover():
        pos = tuple(rng.randrange(40, WORLD - 40) for _ in range(3))
        vel = tuple(rng.choice((-1, 1)) * rng.randint(2, 6) for _ in range(3))
        return Mover(pos, vel)
    m0, m1 = rnd_mover(), rnd_mover()
    stream = []                    # (phase, eyes, [positions], object 0's next position, movers' state)
    for phase, eyes, nobj in (("A", (0,), 1), ("B", (1,), 1), ("C", (0, 1), 1), ("D", (0, 1), 2)):
        for _ in range(LEN[phase]):
            ps = [m0.step()] + ([m1.step()] if nobj == 2 else [])
            state = [(tuple(m.p), tuple(m.v)) for m in ((m0, m1) if nobj == 2 else (m0,))]
            stream.append((phase, eyes, ps, m0.peek(), state))
    return stream


def new_field(build):
    f = core.NethraField(topology_and_evidence_change=build, frontier_tolerance=TOL)
    for _ in range(2 * R * R): f.new()
    return f


def push(f, eyes, ps):
    for p in ps:
        for e, r, c, w in shares(p):
            if e in eyes: f.nethra[e * R * R + r * R + c].push(w)


def occlusion(f, g, stream, t_end, k=5):
    """Frozen copies of field f and reference g; object 0 stops being pushed for k intervals."""
    out = []
    for fld in (f, g):
        h = core.NethraField.from_checkpoint_dict(fld.checkpoint_dict()); h.topology_and_evidence_change = False
        movers = [Mover(p, v) for p, v in stream[t_end][4]]       # all objects keep moving
        rows = []
        for i in range(k):
            ps = [m.step() for m in movers]
            push(h, (0, 1), ps[1:])                                 # object 0 is not pushed
            h.step(1.0)
            rows.append([along(h, e, ps[0], movers[0].peek(), ps[1:]) for e in (0, 1)])
        out.append(rows)
    return out


def run(label, stream, dev):
    f, g = new_field(True), new_field(False)
    K = len(f.nethra); built_in = {}; stats = {}; loc_cache = {}
    t_start = time.perf_counter()
    for t, (phase, eyes, ps, nxt, _state) in enumerate(stream):
        if not dev and phase in ("A", "B"):
            eyes = (0, 1)                                          # control: binocular throughout
        n0 = len(f.nethra); t0 = time.perf_counter()
        for fld in (f, g): push(fld, eyes, ps)
        f.step(1.0)
        wall = time.perf_counter() - t0
        g.step(1.0)
        for n in f.nethra[n0:]: built_in[n] = phase
        s = stats.setdefault(phase, dict(n=0, wall=0.0, built=0, frontier=0, lead=[[], []], ref=[[], []], refound={},
                                         state=[], near=[], mono=[], plead=[[], []], pratio=[[], []], aratio=[[], []],
                                         raratio=[[], []]))
        s["n"] += 1; s["wall"] += wall; s["built"] += len(f.nethra) - n0; s["frontier"] += f.frontier_sizes[-1]
        if phase in ("C", "D"):
            prev0 = stream[t - 1][2][0]
            for e in (0, 1):
                s["lead"][e].append(along(f, e, ps[0], nxt, ps[1:])[0]); s["ref"][e].append(along(g, e, ps[0], nxt, ps[1:])[0])
                pl, pr, ar = ahead(f, e, ps[0], nxt, prev0, ps[1:])
                s["plead"][e].append(pl); s["pratio"][e].append(pr); s["aratio"][e].append(ar)
                s["raratio"][e].append(ahead(g, e, ps[0], nxt, prev0, ps[1:], with_p=False)[2])
            closed = f.closure(f.previous_explicit, f.current_source_event)
            pts, mono = [], 0
            for n in closed:
                if n in built_in:
                    s["refound"][built_in[n]] = s["refound"].get(built_in[n], 0) + 1
                    if built_in[n] == "D": continue                  # two-object Nethra: no single location
                    q = location(n, loc_cache)
                    if q is None: mono += 1
                    else: pts.append(q)
            s["mono"].append(mono)
            if pts:
                # internal state: 3D locations of refound single-object Nethra vs the objects
                d_near = [min(dist(q, p) for p in ps) for q in pts]
                s["near"].append(sum(1 for d in d_near if d < 50) / len(d_near))
                if len(ps) == 1:
                    centre = tuple(sum(q[k] for q in pts) / len(pts) for k in range(3))
                    s["state"].append(dist(centre, ps[0]))
        if t + 1 in (LEN["A"] + LEN["B"] + LEN["C"], len(stream)):
            s.setdefault("occl", occlusion(f, g, stream, t))
    print(f"\n[{label}] R={R}, frontier tolerance {TOL}, total wall {time.perf_counter() - t_start:.0f} s, "
          f"constructed {len(f.nethra) - K}, stored patterns {len(f.source_patterns)}")
    mean = lambda v: (lambda w: sum(w) / len(w) if w else float("nan"))([x for x in v if x == x])
    for phase in ("A", "B", "C", "D"):
        s = stats.get(phase)
        if not s: continue
        line = (f"  {phase}: {s['n']:5d} intervals  built {s['built']:5d}  frontier {s['frontier'] / s['n']:6.0f}  "
                f"{1000 * s['wall'] / s['n']:6.1f} ms/interval")
        if phase in ("C", "D"):
            line += (f"  lead L {mean(s['lead'][0]):+.3f} (ref {mean(s['ref'][0]):+.3f})  "
                     f"R {mean(s['lead'][1]):+.3f} (ref {mean(s['ref'][1]):+.3f})  refound per interval by phase built: "
                     + ", ".join(f"{k} {v / s['n']:.1f}" for k, v in sorted(s["refound"].items())))
            gm = lambda v: (lambda w: (__import__("math").exp(sum(__import__("math").log(x) for x in w) / len(w)) if w else float("nan")))([x for x in v if x == x and x > 0])
            line += (f"\n     expectation: P lead L {mean(s['plead'][0]):+.3f} R {mean(s['plead'][1]):+.3f};  "
                     f"next/prev (geometric mean): P L {gm(s['pratio'][0]):.2f} R {gm(s['pratio'][1]):.2f}, "
                     f"activation L {gm(s['aratio'][0]):.3f} R {gm(s['aratio'][1]):.3f} "
                     f"(ref L {gm(s['raratio'][0]):.3f} R {gm(s['raratio'][1]):.3f})")
            covered = sum(1 for _ in s["near"]) / s["n"]
            line += (f"\n     internal state: intervals with a refound located Nethra {covered:.2f}; of those, share of "
                     f"located Nethra within 50 of an object {mean(s['near']):.2f}; mean monocular Nethra refound "
                     f"{mean(s['mono']):.1f}" + (f"; one object: distance from the object to the centre of the "
                     f"located Nethra, median {sorted(s['state'])[len(s['state']) // 2]:.1f}" if s["state"] else ""))
        print(line)
        if "occl" in s:
            (fr, gr) = s["occl"]
            print("     occlusion, offset of the read from the hidden object (L, R; mass):  " + "  ".join(
                f"[{i + 1}] {fr[i][0][0]:+.2f},{fr[i][1][0]:+.2f};{fr[i][0][1]:.3f} (ref {gr[i][0][0]:+.2f},{gr[i][1][0]:+.2f};{gr[i][0][1]:.3f})"
                for i in range(len(fr))))


if __name__ == "__main__":
    stream = build_stream(random.Random(SEED))
    print(f"seed {SEED}; phases " + ", ".join(f"{k} {v}" for k, v in LEN.items()))
    run("small factors first", stream, True)
    run("control: binocular from the start", stream, False)
