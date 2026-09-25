"""Cost of one interval with two objects in view, CPU vs GPU integration, current core vs top-only
conduction.  Builds the stream of binocular_multi.py (each object alone: left eye, right eye, both;
then both objects together; loops of 40 and PER2 positions, jitter J, threshold TH, R=16) up to
TOGETHER intervals, then continues MEASURE intervals from the same checkpoint in each variant.
Reports ms/interval (total and integration), Nethra integrated, max |a - reference| at the end.
usage: gpu_bench.py TOGETHER MEASURE [PER2]"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(v, "1")
import sys, math, time, random, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ARGS = sys.argv[1:]; sys.argv = [sys.argv[0], "16"]
import binocular3d as b
import nethra_presence as core
import gpu_field, top_conduction_prototype as top
TOGETHER, MEASURE = int(ARGS[0]), int(ARGS[1]); PER2 = int(ARGS[2]) if len(ARGS) > 2 else 37
TH, J, AL = 0.8, 10, 3
print("device:", "GPU (cupy)" if gpu_field.ON_GPU else "numpy fallback", flush=True)
if gpu_field.ON_GPU:
    import cupy as cp
    print("gpu:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode(), flush=True)


def loop(c, rx, ry, rz, per, fy):
    return [(c[0] + rx * math.sin(2 * math.pi * k / per), c[1] + ry * math.sin(fy * 2 * math.pi * k / per + .5),
             c[2] + rz * math.cos(2 * math.pi * k / per)) for k in range(per)]
LOOPS = [loop((250, 130, 250), 200, 80, 200, 40, 2), loop((250, 370, 250), 180, 80, 200, PER2, 1)]


def make_stream():
    rng = random.Random(1); counters = [0, 0]
    def where(i):
        base = LOOPS[i][counters[i] % len(LOOPS[i])]; counters[i] += 1
        return tuple(min(499, max(0, round(base[a] + rng.randint(-J, J)))) for a in range(3))
    out = []
    for i, L in enumerate(LOOPS):
        for eyes, n in (((0,), AL * len(L)), ((1,), AL * len(L)), ((0, 1), 2 * AL * len(L))):
            out += [(eyes, [where(i)]) for _ in range(n)]
    out += [((0, 1), [where(0), where(1)]) for _ in range(TOGETHER + MEASURE)]
    return out


class Timed:
    integ = 0.0
    def _rk4_interval(self, *a, **k):
        t0 = time.perf_counter(); out = super()._rk4_interval(*a, **k); self.integ += time.perf_counter() - t0
        return out


def classes():
    return {"core": core.NethraField, "top": top.TopField}


STREAM = make_stream(); BUILD = len(STREAM) - MEASURE
for struct, cls in classes().items():
    t0 = time.perf_counter()
    f = cls(frontier_tolerance=0.01, source_similarity_threshold=TH)
    for _ in range(2 * b.R * b.R): f.new()
    for eyes, ps in STREAM[:BUILD]:
        b.push(f, eyes, ps); f.step(1.0)
    ck = json.loads(json.dumps(f.checkpoint_dict()))
    print(f"\n== {struct}: built {len(f.nethra)} Nethra in {BUILD} intervals ({time.perf_counter() - t0:.0f} s)", flush=True)
    variants = [("cpu exact auto", False, 0.0, "auto"), ("cpu exact etd", False, 0.0, "etd"), ("gpu exact etd", True, 0.0, "etd"),
                ("cpu tol .01 auto", False, 0.01, "auto"), ("gpu tol .01 etd", True, 0.01, "etd")]
    ref = {}
    for name, gpu, tol, integ in variants:
        base = (top.GpuTopField if struct == "top" else gpu_field.GpuField) if gpu else cls
        V = type("V", (Timed, base), {})
        p = json.loads(json.dumps(ck)); p["parameters"].update(frontier_tolerance=tol, integrator=integ)
        g = V.from_checkpoint_dict(p)
        if struct == "top": g.recompute_covered()
        if gpu:   # warm-up on a copy (kernel compilation)
            w = V.from_checkpoint_dict(json.loads(json.dumps(p)))
            if struct == "top": w.recompute_covered()
            for eyes, ps in STREAM[BUILD:BUILD + 2]: b.push(w, eyes, ps); w.step(1.0)
        g.integ = 0.0; wall = 0.0; fr = 0
        for eyes, ps in STREAM[BUILD:]:
            b.push(g, eyes, ps)
            t1 = time.perf_counter(); g.step(1.0); wall += time.perf_counter() - t1
            fr += g.frontier_sizes[-1]
        acts = [n.activation for n in g.nethra]
        ref[name] = acts
        cmp = {"gpu exact etd": "cpu exact etd", "cpu exact etd": "cpu exact auto", "cpu tol .01 auto": "cpu exact auto", "gpu tol .01 etd": "cpu exact auto"}.get(name)
        diff = ""
        if cmp in ref:
            r = ref[cmp]; k = min(len(r), len(acts))
            diff = f" | max|a - {cmp}| {max(abs(x - y) for x, y in zip(acts[:k], r[:k])):.1e}, Nethra {len(acts)} vs {len(r)}"
        print(f"{name:18s} {1000 * wall / MEASURE:7.1f} ms/interval (integration {1000 * g.integ / MEASURE:6.1f}) integrated {fr / MEASURE:5.0f}{diff}", flush=True)
