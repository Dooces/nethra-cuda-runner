#!/usr/bin/env python3
from collections import defaultdict
import copy
from nethra_route_flux_manifestation_audit import BirthSideField, raw_source_interval

TAUS=(.025,.10,.30)

def main():
    for k in (2,4,6):
        f=BirthSideField(g_min=.20,g_max=1.50,tau=100.,capacitance=1.,leakage=.6,convergence_gain=0.)
        nodes=[f.new() for _ in range(2+2*k)]
        A=nodes[0]; B=nodes[1]
        X=[nodes[2+i] for i in range(k)]
        Y=[nodes[2+k+i] for i in range(k)]
        def seg(i): return (X[i],A,B,Y[i])
        for _ in range(80):
            for i in range(k):
                for n in seg(i):
                    n.push(1.0); f.step(.12)

        eab=frozenset(((A,-1),(B,1)))
        direct=[]
        for i in range(k):
            eby=frozenset(((B,-1),(Y[i],1)))
            direct.append(f.birth_sides.get((eab,eby)))

        for tau in TAUS:
            rp=[]; rs=[]
            for target in range(k):
                g=copy.deepcopy(f)
                gs=g.nethra[:2+2*k]
                gA,gB=gs[0],gs[1]
                gX=[gs[2+i] for i in range(k)]
                gY=[gs[2+k+i] for i in range(k)]
                # Rebuild direct side refs from cloned metadata by source-event identities.
                geab=frozenset(((gA,-1),(gB,1)))
                gd=[]
                for i in range(k):
                    geby=frozenset(((gB,-1),(gY[i],1)))
                    gd.append(g.birth_sides.get((geab,geby)))

                tp={}; ts={}
                seq=[]
                for _ in range(2):
                    for i in range(k):
                        seq.extend((gX[i],gA,gB,gY[i]))
                seq.extend((gX[target],gA,gB))
                for n in seq:
                    raw_source_interval(g,(n,),.12,tp,ts,tau)

                sp=[]; ss=[]
                for sides in gd:
                    if sides is None:
                        sp.append(float('-inf')); ss.append(float('-inf')); continue
                    r,left,right=sides
                    sp.append(tp.get((r,left),0.0))
                    ss.append(ts.get((r,left),0.0))
                op=sorted(range(k),key=lambda i:sp[i],reverse=True)
                os=sorted(range(k),key=lambda i:ss[i],reverse=True)
                rp.append(op.index(target)+1); rs.append(os.index(target)+1)
            print("CONTEXT_QUICK",{"k":k,"tau":tau,
                  "positive_rank1":sum(r==1 for r in rp),"positive_ranks":rp,
                  "signed_rank1":sum(r==1 for r in rs),"signed_ranks":rs})
if __name__=="__main__":main()
