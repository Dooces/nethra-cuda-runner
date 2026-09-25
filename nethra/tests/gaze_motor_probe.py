"""Train closed-loop with the retina reflex (as gaze_retina_device.py).  Then, on frozen copies at each of the
last 2*L training intervals, read motor Nethra activation mL, mR after the interval, trained vs lesioned
(constructed evidence 0).  The difference is what structure adds to each motor Nethra."""
import os,sys
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"gaze_retina_device.py")).read().split("t0=time.perf_counter()")[0])
def lesioned(g):
    for store in (g.incidence_evidence, g.incidence_evidence_in):
        for key,b in store.items():
            for s in b: b[s]=0.0
    for n in g.nethra:
        for r,b in n.routes.items():
            for s in b: b[s]=0.0
    g._incidence_plan={}
    return g
f,ret,mot,pos=make_field(); w=World(); rows=[]
for t in range(TRAIN):
    pre=(w.x,w.e,w.m)
    if t>=TRAIN-2*L:
        cps=[core.NethraField.from_checkpoint_dict(f.checkpoint_dict()) for _ in range(2)]
        lesioned(cps[1])
        vals=[]
        for g in cps:
            g.topology_and_evidence_change=False
            r2,m2,p2=handles(g); o=w.x-w.e
            if abs(o)<=R: r2[o].push(1.0)
            if EFF and w.m: m2[w.m].push(1.0)
            if PROP: p2[w.e].push(1.0)
            g.step(1.0); vals.append((m2[-1].activation,m2[1].activation))
    interval(f,w,ret,mot,pos)
    if t>=TRAIN-2*L: rows.append((t,pre,w.m,vals))
print(f"DIR={DIR}: t (x,e,m now) -> next m by reflex | trained mL mR | lesioned mL mR | structure adds mL mR")
for t,(x,e,m),mn,((tl,tr),(ll,lr)) in rows:
    print(f"  {t} x={x} e={e} o={x-e:+d} m={m:+d} -> {mn:+d} | {tl:.4f} {tr:.4f} | {ll:.4f} {lr:.4f} | {tl-ll:+.4f} {tr-lr:+.4f}")
