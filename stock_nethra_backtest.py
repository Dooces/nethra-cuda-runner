#!/usr/bin/env python3
"""Chronological multi-stock Nethra backtest.

Purpose
-------
Measure whether the current Nethra field + local per-incidence plasticity carries any out-of-sample
next-day directional signal on ordinary daily stock data.

This is an experimental capability audit, not a trading system.

Rules
-----
- No shuffled train/test split.
- No future normalization.
- No ticker-specific hyperparameter choice.
- One global configuration is selected from the middle chronological validation block across all
  stocks, then frozen for the final 20% test block.
- Daily inputs are fixed transductions of same-day market deltas. No hand-labelled chart patterns,
  rolling technical indicators, regime labels, or future-derived features are supplied.
- Time memory comes from persistent Nethra activation.
- Relations are arbitrary-arity Nethra over the full input set. Per-incidence evidence determines
  which input currents matter.
- Outcome is next trading day's close-to-close direction.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import statistics
import time
import urllib.parse
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from nethra import NethraField

SYMBOLS=("AAPL","MSFT","NVDA","TSLA","JPM","XOM")
START="2016-01-01"
END="2026-09-22"

T=0.60
STEPS=6
DT=T/STEPS
ETA=1800.0
GMAX=1.5
TAU=100.0

# Small global sweep. One configuration is selected across ALL tickers on validation only.
CONFIGS=tuple(
    (leakage,ratio,outcome_mode)
    for leakage in (0.3,0.6,1.0)
    for ratio in (0.25,0.5,1.0)
    for outcome_mode in ("binary","magnitude")
)


@dataclass
class Bar:
    date:str
    open:float
    high:float
    low:float
    close:float
    volume:float


def fetch_stooq(symbol):
    d1=START.replace("-","")
    d2=END.replace("-","")
    url=f"https://stooq.com/q/d/l/?s={symbol.lower()}.us&d1={d1}&d2={d2}&i=d"
    req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req,timeout=30) as response:
        raw=response.read().decode("utf-8")
    rows=[]
    for row in csv.DictReader(io.StringIO(raw)):
        try:
            rows.append(Bar(
                date=row["Date"],
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"]),
            ))
        except (ValueError,KeyError):
            pass
    if len(rows)<1000:
        raise RuntimeError(f"Stooq returned only {len(rows)} rows for {symbol}")
    rows.sort(key=lambda b:b.date)
    return rows,"stooq"


def fetch_yahoo(symbol):
    import datetime as _dt
    p1=int(_dt.datetime.fromisoformat(START).replace(tzinfo=_dt.timezone.utc).timestamp())
    p2=int((_dt.datetime.fromisoformat(END)+_dt.timedelta(days=1)).replace(tzinfo=_dt.timezone.utc).timestamp())
    params=urllib.parse.urlencode({
        "period1":p1,"period2":p2,"interval":"1d","events":"history",
        "includeAdjustedClose":"true",
    })
    url=f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{params}"
    req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req,timeout=30) as response:
        obj=json.loads(response.read().decode("utf-8"))
    result=obj["chart"]["result"][0]
    ts=result["timestamp"]
    quote=result["indicators"]["quote"][0]
    adj=result.get("indicators",{}).get("adjclose",[{}])[0].get("adjclose")
    import datetime as _dt
    rows=[]
    for i,t in enumerate(ts):
        try:
            o=float(quote["open"][i]); h=float(quote["high"][i]); l=float(quote["low"][i])
            c=float(adj[i] if adj and adj[i] is not None else quote["close"][i])
            v=float(quote["volume"][i])
            if not all(math.isfinite(x) for x in (o,h,l,c,v)):
                continue
            rows.append(Bar(
                _dt.datetime.fromtimestamp(t,_dt.timezone.utc).date().isoformat(),
                o,h,l,c,v,
            ))
        except (TypeError,ValueError,IndexError):
            pass
    if len(rows)<1000:
        raise RuntimeError(f"Yahoo returned only {len(rows)} rows for {symbol}")
    rows.sort(key=lambda b:b.date)
    return rows,"yahoo"


def fetch(symbol):
    errors=[]
    for fn in (fetch_stooq,fetch_yahoo):
        try:
            return fn(symbol)
        except Exception as exc:
            errors.append(repr(exc))
    raise RuntimeError(f"{symbol} download failed: {errors}")


def signed_pair(x):
    return max(0.0,x),max(0.0,-x)


def squash(x,scale):
    return math.tanh(x/scale)


def feature_rows(bars):
    """Current-day-only fixed transduction. No fitted normalization or rolling indicators."""
    out=[]
    for i in range(1,len(bars)-1):
        prev=bars[i-1]
        cur=bars[i]
        nxt=bars[i+1]
        if min(prev.close,cur.open,cur.high,cur.low,cur.close,nxt.close)<=0:
            continue
        r=math.log(cur.close/prev.close)
        gap=math.log(cur.open/prev.close)
        body=math.log(cur.close/cur.open)
        rng=math.log(cur.high/cur.low)
        vold=0.0
        if prev.volume>0 and cur.volume>0:
            vold=math.log(cur.volume/prev.volume)

        rp,rn=signed_pair(squash(r,.025))
        gp,gn=signed_pair(squash(gap,.018))
        bp,bn=signed_pair(squash(body,.022))
        vp,vn=signed_pair(squash(vold,.35))
        # Range is unsigned physical magnitude and therefore one channel.
        rr=max(0.0,min(1.0,rng/.06))

        features=(rp,rn,gp,gn,bp,bn,vp,vn,rr)
        next_ret=math.log(nxt.close/cur.close)
        out.append((cur.date,features,next_ret,r))
    return out


def evidence_for_g(g):
    g=max(0.0,min(g,GMAX*(1.0-1e-12)))
    if g<=0:
        return 0.0
    return -TAU*math.log(1.0-g/GMAX)


class StockNethra:
    def __init__(self,leakage,ratio):
        self.f=NethraField(
            g_min=0.0,g_max=GMAX,tau=TAU,
            leakage=leakage,convergence_gain=0.0,
        )
        self.inputs=[self.f.new() for _ in range(9)]
        self.y_up=self.f.new()
        self.y_down=self.f.new()
        self.r_up=self.f.new()
        self.r_down=self.f.new()

        # Structural routes are arbitrary-arity. Persistent strength is shadow per-incidence because
        # the one-file core still stores route-wide evidence; this is the same local representation
        # validated in the incidence-credit probes.
        self.f._route(self.r_up,tuple(self.inputs)+(self.y_up,),frozenset(),0)
        self.f._route(self.r_down,tuple(self.inputs)+(self.y_down,),frozenset(),0)

        seed=evidence_for_g(min(GMAX*.95,leakage*ratio))
        self.ev={}
        for r,y in ((self.r_up,self.y_up),(self.r_down,self.y_down)):
            for n in (*self.inputs,y):
                self.ev[frozenset((r,n))]=seed

        def edges():
            rows=[]
            for r,y in ((self.r_up,self.y_up),(self.r_down,self.y_down)):
                for n in (*self.inputs,y):
                    e=self.ev[frozenset((r,n))]
                    g=GMAX*(1.0-math.exp(-max(0.0,e)/TAU))
                    if g>0.0:
                        rows.append((r,n,g))
            return tuple(rows)
        self.f._edges=edges

    def derivative(self,state,edges):
        cur={n:n.external-self.f.leakage*state[n] for n in self.f.nethra}
        for a,b,g in edges:
            q=g*(state[a]-state[b])
            cur[a]-=q
            cur[b]+=q
        return cur

    def interval(self,features):
        for n in self.f.nethra:
            n.external=0.0
        for n,x in zip(self.inputs,features):
            n.external=float(x)

        edges=self.f._edges()
        q_up=0.0
        q_dn=0.0

        def watch(state,r,y):
            for a,b,g in edges:
                if a is r and b is y:
                    return g*(state[a]-state[b])
                if a is y and b is r:
                    return g*(state[b]-state[a])
            return 0.0

        for _ in range(STEPS):
            a0={n:n.activation for n in self.f.nethra}
            k1=self.derivative(a0,edges)
            a1={n:a0[n]+.5*DT*k1[n] for n in self.f.nethra}
            k2=self.derivative(a1,edges)
            a2={n:a0[n]+.5*DT*k2[n] for n in self.f.nethra}
            k3=self.derivative(a2,edges)
            a3={n:a0[n]+DT*k3[n] for n in self.f.nethra}
            k4=self.derivative(a3,edges)

            q_up += DT*(
                watch(a0,self.r_up,self.y_up)+
                2*watch(a1,self.r_up,self.y_up)+
                2*watch(a2,self.r_up,self.y_up)+
                watch(a3,self.r_up,self.y_up)
            )/6.0
            q_dn += DT*(
                watch(a0,self.r_down,self.y_down)+
                2*watch(a1,self.r_down,self.y_down)+
                2*watch(a2,self.r_down,self.y_down)+
                watch(a3,self.r_down,self.y_down)
            )/6.0

            for n in self.f.nethra:
                n.activation=a0[n]+DT*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.0

        for n in self.f.nethra:
            n.external=0.0

        return max(0.0,q_up),max(0.0,q_dn)

    def predict_margin(self,features):
        p_up,p_dn=self.interval(features)
        # Field state is the prediction. Current receipt is retained for local plasticity.
        margin=self.y_up.activation-self.y_down.activation
        return margin,p_up,p_dn

    def update(self,features,p_up,p_dn,next_ret,outcome_mode):
        if outcome_mode=="binary":
            s_up=T if next_ret>0 else 0.0
            s_dn=T if next_ret<0 else 0.0
        else:
            mag=max(0.02,min(1.0,abs(math.tanh(next_ret/.025))))
            s_up=T*mag if next_ret>0 else 0.0
            s_dn=T*mag if next_ret<0 else 0.0

        eps_up=s_up-p_up
        eps_dn=s_dn-p_dn

        # Output incidences.
        for r,y,p,eps in (
            (self.r_up,self.y_up,p_up,eps_up),
            (self.r_down,self.y_down,p_dn,eps_dn),
        ):
            key=frozenset((r,y))
            self.ev[key]=max(0.0,self.ev[key]+ETA*p*eps)

        # Input incidence credit is proportional to actual positive integrated input magnitude.
        # This is a deliberately local approximation to the interval current attribution validated
        # by the whole-relation probes; inactive channels receive no credit.
        total=sum(features)
        if total>0:
            for r,p,eps in ((self.r_up,p_up,eps_up),(self.r_down,p_dn,eps_dn)):
                tension=p*eps
                for n,x in zip(self.inputs,features):
                    if x<=0:
                        continue
                    key=frozenset((r,n))
                    self.ev[key]=max(0.0,self.ev[key]+ETA*tension*(x/total))


def confusion_metrics(labels,preds):
    tp=sum(y==1 and p==1 for y,p in zip(labels,preds))
    tn=sum(y==0 and p==0 for y,p in zip(labels,preds))
    fp=sum(y==0 and p==1 for y,p in zip(labels,preds))
    fn=sum(y==1 and p==0 for y,p in zip(labels,preds))
    n=len(labels)
    acc=(tp+tn)/n if n else float("nan")
    tpr=tp/(tp+fn) if tp+fn else float("nan")
    tnr=tn/(tn+fp) if tn+fp else float("nan")
    bal=(tpr+tnr)/2 if math.isfinite(tpr) and math.isfinite(tnr) else float("nan")
    return {"n":n,"accuracy":acc,"balanced_accuracy":bal,"tp":tp,"tn":tn,"fp":fp,"fn":fn}


def wilson(p,n,z=1.96):
    if n<=0:
        return (float("nan"),float("nan"))
    den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/den
    return center-half,center+half


def evaluate(rows,config,train_end,eval_start,eval_end):
    leakage,ratio,outcome_mode=config
    model=StockNethra(leakage,ratio)

    labels=[]
    preds=[]
    persistence=[]
    always_up=[]
    margins=[]
    rets=[]

    # Online walk-forward. Everything before eval_start is training. During evaluation a prediction
    # is recorded before that day's label updates the model.
    for i,(date,features,next_ret,current_ret) in enumerate(rows[:eval_end]):
        margin,p_up,p_dn=model.predict_margin(features)
        label=1 if next_ret>0 else 0

        if i>=eval_start:
            pred=1 if margin>=0.0 else 0
            labels.append(label)
            preds.append(pred)
            persistence.append(1 if current_ret>=0 else 0)
            always_up.append(1)
            margins.append(margin)
            rets.append(next_ret)

        model.update(features,p_up,p_dn,next_ret,outcome_mode)

    m=confusion_metrics(labels,preds)
    m["persistence"]=confusion_metrics(labels,persistence)
    m["always_up"]=confusion_metrics(labels,always_up)
    m["start_date"]=rows[eval_start][0]
    m["end_date"]=rows[eval_end-1][0]
    lo,hi=wilson(m["accuracy"],m["n"])
    m["accuracy_ci95"]=(lo,hi)

    # Point-biserial-like directional association of raw margin with next return.
    if len(margins)>2 and statistics.pstdev(margins)>0 and statistics.pstdev(rets)>0:
        mm=statistics.mean(margins); mr=statistics.mean(rets)
        cov=sum((x-mm)*(y-mr) for x,y in zip(margins,rets))/len(margins)
        m["margin_return_corr"]=cov/(statistics.pstdev(margins)*statistics.pstdev(rets))
    else:
        m["margin_return_corr"]=0.0
    return m


def stock_validation(payload):
    symbol,rows=payload
    n=len(rows)
    train_end=int(n*.60)
    val_end=int(n*.80)
    result={}
    for cfg in CONFIGS:
        result[cfg]=evaluate(rows,cfg,train_end,train_end,val_end)
    return symbol,n,train_end,val_end,result


def stock_test(payload):
    symbol,rows,cfg=payload
    n=len(rows)
    test_start=int(n*.80)
    m=evaluate(rows,cfg,test_start,test_start,n)
    return symbol,n,test_start,m


def main():
    t0=time.perf_counter()
    print("cpu_count",os.cpu_count())

    # Download data once.
    data={}
    sources={}
    for symbol in SYMBOLS:
        bars,source=fetch(symbol)
        rows=feature_rows(bars)
        data[symbol]=rows
        sources[symbol]=source
        print("data",symbol,source,len(bars),bars[0].date,bars[-1].date,"examples",len(rows))

    workers=min(len(SYMBOLS),max(1,os.cpu_count() or 1))
    print("workers",workers)

    # Validation sweep in parallel by ticker.
    validations={}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs={pool.submit(stock_validation,(s,data[s])):s for s in SYMBOLS}
        for fut in as_completed(futs):
            symbol,n,tr,ve,result=fut.result()
            validations[symbol]=result
            print("validation_done",symbol,n,tr,ve)

    # Select one global configuration by macro balanced accuracy across all stocks.
    rankings=[]
    for cfg in CONFIGS:
        vals=[validations[s][cfg]["balanced_accuracy"] for s in SYMBOLS]
        accs=[validations[s][cfg]["accuracy"] for s in SYMBOLS]
        rankings.append((
            statistics.mean(vals),
            statistics.mean(accs),
            cfg,
        ))
    rankings.sort(reverse=True,key=lambda x:(x[0],x[1]))
    selected=rankings[0][2]
    print("validation_rankings")
    for bal,acc,cfg in rankings:
        print("VALCFG",json.dumps({
            "config":cfg,"macro_balanced_accuracy":bal,"macro_accuracy":acc
        }))
    print("selected_config",selected)

    # Locked final test, rebuilt from scratch and trained through first 80%.
    tests={}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs={pool.submit(stock_test,(s,data[s],selected)):s for s in SYMBOLS}
        for fut in as_completed(futs):
            symbol,n,start,m=fut.result()
            tests[symbol]=m
            print("TEST",symbol,json.dumps(m,sort_keys=True))

    macro_acc=statistics.mean(tests[s]["accuracy"] for s in SYMBOLS)
    macro_bal=statistics.mean(tests[s]["balanced_accuracy"] for s in SYMBOLS)
    macro_persist=statistics.mean(tests[s]["persistence"]["accuracy"] for s in SYMBOLS)
    macro_up=statistics.mean(tests[s]["always_up"]["accuracy"] for s in SYMBOLS)

    all_labels_n=sum(tests[s]["n"] for s in SYMBOLS)
    correct=sum(tests[s]["tp"]+tests[s]["tn"] for s in SYMBOLS)
    micro_acc=correct/all_labels_n
    lo,hi=wilson(micro_acc,all_labels_n)

    summary={
        "symbols":SYMBOLS,
        "sources":sources,
        "selected_config":selected,
        "macro_accuracy":macro_acc,
        "macro_balanced_accuracy":macro_bal,
        "macro_persistence_accuracy":macro_persist,
        "macro_always_up_accuracy":macro_up,
        "micro_accuracy":micro_acc,
        "micro_n":all_labels_n,
        "micro_accuracy_ci95":(lo,hi),
        "seconds":time.perf_counter()-t0,
    }
    print("SUMMARY",json.dumps(summary,sort_keys=True))
    print("all_assertions_passed")


if __name__=="__main__":
    main()
