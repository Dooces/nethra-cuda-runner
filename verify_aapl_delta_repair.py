#!/usr/bin/env python3
import csv, io, urllib.request
from datetime import datetime, timezone
from nethra import NethraField

URL="https://raw.githubusercontent.com/K2hmed/AAPL-Stock-Data/refs/heads/main/CSV%20Files/aapl_raw_data.csv"
N=400

def load():
    req=urllib.request.Request(URL,headers={"User-Agent":"Mozilla/5.0"})
    txt=urllib.request.urlopen(req,timeout=30).read().decode()
    rows=[]
    for r in csv.DictReader(io.StringIO(txt)):
        p=r.get("adjusted_close") or r.get("close")
        if not p: continue
        d=datetime.strptime(r["date"],"%Y-%m-%d").replace(tzinfo=timezone.utc)
        rows.append((d.timestamp()/2_000_000_000.0,float(p)/300.0,r["date"]))
    return rows[-N:]

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

rows=load()
f=NethraField()
t=f.new(); p=f.new()
for i,(ts,price,date) in enumerate(rows,1):
    t.push(ts); p.push(price); f.step(.1)
    if i in (1,2,3,10,25,50,100,200,400):
        print("AAPL",i,date,"N",len(f.nethra),"ROUTES",sum(len(n.routes) for n in f.nethra),"DEPTH",depth(f),"EVENT",len(f.current_source_event))
print("AAPL_FINAL",len(f.nethra),sum(len(n.routes) for n in f.nethra),depth(f))
