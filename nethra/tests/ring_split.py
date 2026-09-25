"""Per interval (last lap): share of positive P toward cells behind / at / ahead of the object
(signed circular offset of the cell centre from the current position, in cell spacings); and the
input-free continuation: copy the field (no evidence change), step once with nothing pushed, and
read each cell's activation minus leakage-only decay of its current activation."""
import os,sys,math
src=open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"ring_graded.py")).read()
exec(src.split("f=core.NethraField")[0])
def run():
    f=core.NethraField(top_only_conduction=TOP, leakage=LEAK)
    cells=[f.new() for _ in range(NC)]
    positions=[(STEP*t)%L for t in range(L)]
    tot={"behind":0,"at":0,"ahead":0}; cont_err=[]; stay=[]; nextcell_P=[]
    for lap in range(LAPS):
        for t,x in enumerate(positions):
            for i,w in weights(x).items(): cells[i].push(w)
            f.step(1.0)
            if lap<LAPS-1: continue
            xn=positions[(t+1)%L]
            P=P_toward(f)
            for i,c in enumerate(cells):
                v=P.get(c,0.0)
                off=((i*sp-x)+L/2)%L-L/2
                k="at" if abs(off)<sp*0.999 else ("ahead" if off>0 else "behind")
                tot[k]+=v
            g=core.NethraField.from_checkpoint_dict(f.checkpoint_dict()); g.topology_and_evidence_change=False
            now=[c.activation for c in g.nethra[:NC]]
            g.step(1.0)
            ex={i:g.nethra[i].activation-now[i]*math.exp(-LEAK) for i in range(NC)}
            pos={i:v for i,v in ex.items() if v>0}
            pc=circ(pos) if pos else None
            if pc is not None: cont_err.append(cd(pc,xn))
            stay.append(cd(x,xn))
    s=sum(tot.values()) or 1
    print(f"L={L} NC={NC} STEP={STEP} top={TOP} leak={LEAK}: positive P toward cells behind {tot['behind']/s:.2f} at {tot['at']/s:.2f} ahead {tot['ahead']/s:.2f};"
          f" input-free continuation gain as a point: |to next| {sum(cont_err)/max(1,len(cont_err)):.2f} ({len(cont_err)}/{len(stay)}) staying put {sum(stay)/len(stay):.2f}")
run()
