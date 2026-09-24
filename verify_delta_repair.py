#!/usr/bin/env python3
from nethra import NethraField

def depth(field):
    memo={}
    visiting=set()
    def d(n):
        if n in memo: return memo[n]
        if n in visiting: return 0
        visiting.add(n)
        best=0
        for route in n.routes:
            for m in route:
                best=max(best,1+d(m))
        visiting.remove(n)
        memo[n]=best
        return best
    return max((d(n) for n in field.nethra),default=0)

f=NethraField()
t=f.new(); p=f.new()
events=[]
for i,(tv,pv) in enumerate([(1.0,10.0),(2.0,11.5),(3.0,9.25),(4.0,13.75),(5.0,12.0),(6.0,15.5),(7.0,14.25),(8.0,18.0)]):
    t.push(tv)
    p.push(pv)
    delta=f.step(.1)
    events.append(tuple(sorted((id(n),round(v,12)) for n,v in f.current_source_event)))
    print("STEP",i+1,"DT",repr(delta[t]),repr(delta[p]),"EVENT",[(round(v,12)) for _,v in f.current_source_event],"N",len(f.nethra),"ROUTES",sum(len(n.routes) for n in f.nethra),"DEPTH",depth(f))

assert len(set(events)) == len(events), events
assert any(abs(v) not in (0.0,1.0) for _,v in f.current_source_event)
assert len(f.nethra) > 3
assert depth(f) > 1
print("DELTA_REPAIR_PASS",len(f.nethra),sum(len(n.routes) for n in f.nethra),depth(f))
