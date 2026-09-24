#!/usr/bin/env python3
import csv,io,urllib.request
from datetime import datetime,timezone
from nethra import NethraField
u="https://raw.githubusercontent.com/K2hmed/AAPL-Stock-Data/refs/heads/main/CSV%20Files/aapl_raw_data.csv"
req=urllib.request.Request(u,headers={"User-Agent":"Mozilla/5.0"})
rows=[]
for r in csv.DictReader(io.StringIO(urllib.request.urlopen(req,timeout=30).read().decode())):
    p=r.get("adjusted_close") or r.get("close")
    if p:
        d=datetime.strptime(r["date"],"%Y-%m-%d").replace(tzinfo=timezone.utc)
        rows.append((d.timestamp()/2_000_000_000.0,float(p)/300.0,r["date"]))
rows=rows[-40:]
f=NethraField()
t=f.new();p=f.new()
def dep():
    memo={}; visiting=set()
    def d(n):
        if n in memo:return memo[n]
        if n in visiting:return 0
        visiting.add(n);v=0
        for route in n.routes:
            for m in route:v=max(v,1+d(m))
        visiting.remove(n);memo[n]=v;return v
    return max((d(n) for n in f.nethra),default=0)
for i,(ts,px,date) in enumerate(rows,1):
    t.push(ts);p.push(px);f.step(.1)
    if i in (1,2,3,5,10,20,40):
        print(i,date,len(f.nethra),sum(len(n.routes) for n in f.nethra),dep(),[(round(v,9)) for _,v in f.current_source_event])
print("FINAL",len(f.nethra),sum(len(n.routes) for n in f.nethra),dep())
assert len(f.nethra)>3
assert dep()>1
