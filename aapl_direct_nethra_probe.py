#!/usr/bin/env python3
import copy, csv, io, math, urllib.request
from datetime import datetime, timezone
from nethra import NethraField

URL="https://raw.githubusercontent.com/K2hmed/AAPL-Stock-Data/refs/heads/main/CSV%20Files/aapl_raw_data.csv"
DT=0.1
TIME_SCALE=2_000_000_000.0
PRICE_SCALE=300.0
TEST_N=100

def load():
    req=urllib.request.Request(URL,headers={"User-Agent":"Mozilla/5.0"})
    txt=urllib.request.urlopen(req,timeout=30).read().decode()
    rows=[]
    for r in csv.DictReader(io.StringIO(txt)):
        p=r.get("adjusted_close") or r.get("close")
        if not p: continue
        d=datetime.strptime(r["date"],"%Y-%m-%d").replace(tzinfo=timezone.utc)
        rows.append((d.timestamp(),float(p),r["date"]))
    return rows

def own_prediction(field, price):
    physical=field._physical_incidences(field.current_event)
    A=field.current_interval_integral
    total=0.0
    for (relation,member),row in physical.items():
        if member is not price: continue
        g=float(row["g"])
        q=g*(float(A.get(relation,0.0))-float(A.get(member,0.0)))
        if q>0: total+=q
    return total

def manifestation_per_dollar(field, time_n, price_n, ts):
    base=copy.deepcopy(field)
    bt=base.nethra[field.nethra.index(time_n)]
    bp=base.nethra[field.nethra.index(price_n)]
    base.native_learning=False
    bt.push(ts/TIME_SCALE)
    base.step(DT)
    b=bp.activation

    one=copy.deepcopy(field)
    ot=one.nethra[field.nethra.index(time_n)]
    op=one.nethra[field.nethra.index(price_n)]
    one.native_learning=False
    ot.push(ts/TIME_SCALE)
    op.push(1.0/PRICE_SCALE)
    one.step(DT)
    return one.capacitance*max(0.0,op.activation-b)

def main():
    rows=load()
    train=rows[:-TEST_N]
    test=rows[-TEST_N:]
    f=NethraField()
    time_n=f.new(); price_n=f.new()

    for ts,p,_ in train:
        time_n.push(ts/TIME_SCALE)
        price_n.push(p/PRICE_SCALE)
        f.step(DT)

    print("TRAIN_ROWS",len(train))
    print("TEST_ROWS",len(test))
    print("TRAIN_END",train[-1][2],train[-1][1])
    print("NETHRA_AFTER_TRAIN",len(f.nethra))
    print("ROUTES_AFTER_TRAIN",sum(len(n.routes) for n in f.nethra))

    preds=[]
    prev=train[-1][1]
    for ts,actual,date in test:
        P=own_prediction(f,price_n)
        s=manifestation_per_dollar(f,time_n,price_n,ts)
        pred=P/s if s>1e-15 else float("nan")
        preds.append((date,pred,actual,prev,P,s))

        time_n.push(ts/TIME_SCALE)
        price_n.push(actual/PRICE_SCALE)
        f.step(DT)
        prev=actual

    good=[r for r in preds if math.isfinite(r[1])]
    mae=sum(abs(p-a) for _,p,a,_,_,_ in good)/len(good)
    med=sorted(abs(p-a) for _,p,a,_,_,_ in good)[len(good)//2]
    dir_ok=sum(((p-pr)>0)==((a-pr)>0) for _,p,a,pr,_,_ in good)/len(good)
    naive=sum(abs(pr-a) for _,_,a,pr,_,_ in good)/len(good)
    print("PREDICTIONS",len(good))
    print("MAE",mae)
    print("MEDIAN_AE",med)
    print("DIRECTION_ACCURACY",dir_ok)
    print("LAST_PRICE_NAIVE_MAE",naive)
    print("NETHRA_FINAL",len(f.nethra))
    print("ROUTES_FINAL",sum(len(n.routes) for n in f.nethra))
    print("SAMPLES_BEGIN")
    for r in good[:10]:
        print("PRED",r[0],f"{r[1]:.6f}",f"{r[2]:.6f}",f"err={r[1]-r[2]:+.6f}")
    print("SAMPLES_END")
    for r in good[-10:]:
        print("PRED",r[0],f"{r[1]:.6f}",f"{r[2]:.6f}",f"err={r[1]-r[2]:+.6f}")

if __name__=="__main__":
    main()
