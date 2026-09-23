#!/usr/bin/env python3
"""Replay-established, one-day-online Nethra stock experiment.

This experiment is built around the current Nethra machinery rather than conventional stock-model
evaluation.

Timeline
--------
Common data begins 2010-07-01.
Establishment sequence: 2010-07-01 through 2020-06-30 (~10 years).
Online sequence: 2020-07-01 onward.

The establishment sequence is replayed from its beginning repeatedly while persistent Nethra
evidence is retained. Each replay resets only transient activation, so the same historical sequence
is learned again without inventing a wrap-around transition from its end to its beginning.

During the online period:
  1. current day is presented once;
  2. the field's priming vector for tomorrow is recorded and scored only when tomorrow is revealed;
  3. that one newly revealed transition is rehearsed from the same pre-day field state until its
     local plasticity update settles or a small maximum repeat count is reached;
  4. the day is then committed once with the updated evidence and time advances by exactly one day.

There is never a multi-day forecast.

Architecture
------------
Each target stock has its own field. Inputs are fixed physical market deltas:
- its own adjusted close return, adjusted gap, adjusted intraday body, adjusted range, volume delta;
- same-day adjusted close returns of the other five stocks.

Positive/negative quantities are separate ordinary Nethra currents. No moving average, RSI,
candlestick label, rolling window, regime label, or future-normalized feature is supplied.

Prediction structure can be recursively deep. Level 1 relations connect the complete current input
field to UP/DOWN consequence Nethra. Each subsequent level is an ordinary Nethra whose direct support
includes the immediately preceding predictive Nethra handles plus the current inputs. Upper levels
never contain primitive history beyond current input; previous predictive handles carry the temporal
field state.

Per-incidence evidence follows the positive-flow local tension rule already tested elsewhere.
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
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from nethra import NethraField

SYMBOLS=("AAPL","MSFT","NVDA","TSLA","JPM","XOM")
START="2010-07-01"
END="2026-09-22"
ESTABLISH_END="2020-06-30"

T=.60
STEPS=5
DT=T/STEPS
TARGET_CURRENT=.02
TARGET_CHARGE=T*TARGET_CURRENT
ETA=2200.0
GMAX=1.5
TAU=100.0
LEAKAGE=.6
SEED_RATIO=.5

MAX_EPOCHS=64
MIN_EPOCHS=8
ONLINE_MAX_REHEARSAL=50
ONLINE_MIN_REHEARSAL=4

DEPTHS=(2,)


@dataclass(frozen=True)
class Bar:
    date:str
    open:float
    high:float
    low:float
    close:float
    volume:float


def fetch_yahoo(symbol):
    import datetime as dt
    p1=int(dt.datetime.fromisoformat(START).replace(tzinfo=dt.timezone.utc).timestamp())
    p2=int((dt.datetime.fromisoformat(END)+dt.timedelta(days=1)).replace(tzinfo=dt.timezone.utc).timestamp())
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
    q=result["indicators"]["quote"][0]
    adj=result["indicators"].get("adjclose",[{}])[0].get("adjclose")
    rows=[]
    for i,t in enumerate(ts):
        try:
            rawc=float(q["close"][i])
            adjc=float(adj[i]) if adj and adj[i] is not None else rawc
            if rawc<=0 or adjc<=0:
                continue
            factor=adjc/rawc
            # Adjust all OHLC consistently into the adjusted-close coordinate.
            o=float(q["open"][i])*factor
            h=float(q["high"][i])*factor
            l=float(q["low"][i])*factor
            v=float(q["volume"][i])
            if not all(math.isfinite(x) for x in (o,h,l,adjc,v)):
                continue
            rows.append(Bar(
                dt.datetime.fromtimestamp(t,dt.timezone.utc).date().isoformat(),
                o,h,l,adjc,v,
            ))
        except (TypeError,ValueError,IndexError):
            pass
    rows.sort(key=lambda x:x.date)
    return rows


def align_data():
    raw={s:fetch_yahoo(s) for s in SYMBOLS}
    maps={s:{b.date:b for b in rows} for s,rows in raw.items()}
    dates=sorted(set.intersection(*(set(m) for m in maps.values())))
    bars={s:[maps[s][d] for d in dates] for s in SYMBOLS}
    return dates,bars


def squash(x,scale):
    return math.tanh(x/scale)


def signed_pair(x):
    return max(0.0,x),max(0.0,-x)


def build_examples(dates,bars,target):
    idx=SYMBOLS.index(target)
    examples=[]
    for i in range(1,len(dates)-1):
        own_prev=bars[target][i-1]
        own=bars[target][i]
        own_next=bars[target][i+1]
        if min(own_prev.close,own.open,own.high,own.low,own.close,own_next.close)<=0:
            continue

        r=math.log(own.close/own_prev.close)
        gap=math.log(own.open/own_prev.close)
        body=math.log(own.close/own.open)
        rng=math.log(own.high/own.low)
        vold=math.log(own.volume/own_prev.volume) if own.volume>0 and own_prev.volume>0 else 0.0

        feats=[]
        for x,scale in ((r,.025),(gap,.018),(body,.022),(vold,.35)):
            feats.extend(signed_pair(squash(x,scale)))
        feats.append(max(0.0,min(1.0,rng/.06)))

        # Same-day cross-stock return currents. These are simultaneous observations, not future data.
        for other in SYMBOLS:
            if other==target:
                continue
            p=bars[other][i-1]
            c=bars[other][i]
            rr=math.log(c.close/p.close)
            feats.extend(signed_pair(squash(rr,.025)))

        next_ret=math.log(own_next.close/own.close)
        examples.append((dates[i],tuple(feats),next_ret,r))

    return examples


def evidence_for_g(g):
    g=max(0.0,min(g,GMAX*(1-1e-12)))
    return 0.0 if g<=0 else -TAU*math.log(1.0-g/GMAX)


class ReplayNethra:
    def __init__(self,n_inputs,depth):
        self.depth=depth
        self.f=NethraField(
            g_min=0.0,g_max=GMAX,tau=TAU,
            leakage=LEAKAGE,convergence_gain=0.0,
        )
        self.inputs=[self.f.new() for _ in range(n_inputs)]
        self.y_up=self.f.new()
        self.y_dn=self.f.new()

        self.up=[]
        self.dn=[]
        self.members={}
        self.ev={}

        seed=evidence_for_g(LEAKAGE*SEED_RATIO)

        for level in range(depth):
            ru=self.f.new(); rd=self.f.new()
            if level==0:
                common=tuple(self.inputs)
            else:
                common=tuple(self.inputs)+(self.up[level-1],self.dn[level-1])

            mu=common+(self.y_up,)
            md=common+(self.y_dn,)
            self.f._route(ru,mu,frozenset(),0)
            self.f._route(rd,md,frozenset(),0)
            self.up.append(ru); self.dn.append(rd)
            self.members[ru]=mu
            self.members[rd]=md

            for r,members in ((ru,mu),(rd,md)):
                for m in members:
                    self.ev[frozenset((r,m))]=seed

        def edges():
            rows=[]
            for r,members in self.members.items():
                for m in members:
                    e=self.ev[frozenset((r,m))]
                    g=GMAX*(1.0-math.exp(-max(0.0,e)/TAU))
                    if g>0:
                        rows.append((r,m,g))
            return tuple(rows)
        self.f._edges=edges

    def reset_activation(self):
        for n in self.f.nethra:
            n.activation=0.0
            n.external=0.0

    def snapshot(self):
        return tuple(n.activation for n in self.f.nethra)

    def restore(self,snap):
        for n,a in zip(self.f.nethra,snap):
            n.activation=float(a)
            n.external=0.0

    def derivative(self,state,edges):
        cur={n:n.external-self.f.leakage*state[n] for n in self.f.nethra}
        for a,b,g in edges:
            q=g*(state[a]-state[b])
            cur[a]-=q
            cur[b]+=q
        return {n:v/self.f.capacitance for n,v in cur.items()}

    def interval(self,features):
        for n in self.f.nethra:
            n.external=0.0
        for n,x in zip(self.inputs,features):
            n.external=float(x)

        edges=self.f._edges()
        charges=[0.0]*len(edges)

        def flows(state):
            return [g*(state[a]-state[b]) for a,b,g in edges]

        for _ in range(STEPS):
            a0={n:n.activation for n in self.f.nethra}
            k1=self.derivative(a0,edges)
            a1={n:a0[n]+.5*DT*k1[n] for n in self.f.nethra}; k2=self.derivative(a1,edges)
            a2={n:a0[n]+.5*DT*k2[n] for n in self.f.nethra}; k3=self.derivative(a2,edges)
            a3={n:a0[n]+DT*k3[n] for n in self.f.nethra}; k4=self.derivative(a3,edges)
            qs=(flows(a0),flows(a1),flows(a2),flows(a3))

            for i in range(len(edges)):
                charges[i]+=DT*(qs[0][i]+2*qs[1][i]+2*qs[2][i]+qs[3][i])/6.0
            for n in self.f.nethra:
                n.activation=a0[n]+DT*(k1[n]+2*k2[n]+2*k3[n]+k4[n])/6.0

        for n in self.f.nethra:
            n.external=0.0

        # Orient integrated charges.
        out=defaultdict(dict)
        incoming=defaultdict(float)
        for (a,b,g),q in zip(edges,charges):
            if q>0:
                out[a][b]=out[a].get(b,0.0)+q
                incoming[b]+=q
            elif q<0:
                z=-q
                out[b][a]=out[b].get(a,0.0)+z
                incoming[a]+=z

        p_up=incoming[self.y_up]
        p_dn=incoming[self.y_dn]
        margin=self.y_up.activation-self.y_dn.activation

        return {
            "margin":margin,
            "p_up":p_up,
            "p_dn":p_dn,
            "out":out,
            "incoming":incoming,
        }

    def surprise(self,receipt,next_ret):
        su=TARGET_CHARGE if next_ret>0 else 0.0
        sd=TARGET_CHARGE if next_ret<0 else 0.0
        # L1 residual is interpretable without pretending current is a probability.
        return abs(su-receipt["p_up"])+abs(sd-receipt["p_dn"])

    def update_from_receipt(self,receipt,next_ret):
        su=TARGET_CHARGE if next_ret>0 else 0.0
        sd=TARGET_CHARGE if next_ret<0 else 0.0
        eps={self.y_up:su-receipt["p_up"],self.y_dn:sd-receipt["p_dn"]}

        total_abs_update=0.0

        for level in range(self.depth):
            for r,y in ((self.up[level],self.y_up),(self.dn[level],self.y_dn)):
                # Positive flow actually driven by this relation into its outcome.
                py=receipt["out"].get(r,{}).get(y,0.0)
                tension=py*eps[y]

                # Outcome incidence.
                ky=frozenset((r,y))
                old=self.ev[ky]
                new=max(0.0,old+ETA*tension)
                self.ev[ky]=new
                total_abs_update+=abs(new-old)

                # Credit only direct Nethra that actually supplied this relation in this interval.
                suppliers=[]
                for m in self.members[r]:
                    if m is y:
                        continue
                    q=receipt["out"].get(m,{}).get(r,0.0)
                    if q>0:
                        suppliers.append((m,q))
                total=sum(q for _,q in suppliers)

                if total>0:
                    for m,q in suppliers:
                        k=frozenset((r,m))
                        old=self.ev[k]
                        new=max(0.0,old+ETA*tension*(q/total))
                        self.ev[k]=new
                        total_abs_update+=abs(new-old)

        return total_abs_update

    def rehearse_transition(self,pre_state,features,next_ret,max_repeats):
        previous_surprise=None
        final_surprise=None
        repeats=0

        for rep in range(max_repeats):
            self.restore(pre_state)
            receipt=self.interval(features)
            before=self.surprise(receipt,next_ret)
            delta=self.update_from_receipt(receipt,next_ret)
            final_surprise=before
            repeats=rep+1

            if rep+1>=ONLINE_MIN_REHEARSAL:
                if delta<1e-8:
                    break
                if previous_surprise is not None:
                    scale=max(previous_surprise,1e-12)
                    if abs(before-previous_surprise)/scale<5e-4:
                        break
            previous_surprise=before

        # Commit exactly one physical interval with the newly learned conductances.
        self.restore(pre_state)
        committed=self.interval(features)
        after=self.surprise(committed,next_ret)
        return repeats,final_surprise,after


def metrics(labels,preds):
    tp=sum(y==1 and p==1 for y,p in zip(labels,preds))
    tn=sum(y==0 and p==0 for y,p in zip(labels,preds))
    fp=sum(y==0 and p==1 for y,p in zip(labels,preds))
    fn=sum(y==1 and p==0 for y,p in zip(labels,preds))
    n=len(labels)
    acc=(tp+tn)/n if n else 0.0
    tpr=tp/(tp+fn) if tp+fn else 0.0
    tnr=tn/(tn+fp) if tn+fp else 0.0
    return {
        "n":n,"accuracy":acc,"balanced_accuracy":(tpr+tnr)/2,
        "tp":tp,"tn":tn,"fp":fp,"fn":fn,
    }


def run_target(target,examples,depth):
    est=[x for x in examples if x[0]<=ESTABLISH_END]
    online=[x for x in examples if x[0]>ESTABLISH_END]
    model=ReplayNethra(len(examples[0][1]),depth)

    epoch_rows=[]
    prev_epoch_probe=None

    # Repeatedly replay exact same establishment sequence.
    for epoch in range(1,MAX_EPOCHS+1):
        model.reset_activation()
        surprises=[]
        probe=[]

        for date,features,next_ret,current_ret in est:
            receipt=model.interval(features)
            surprises.append(model.surprise(receipt,next_ret))
            model.update_from_receipt(receipt,next_ret)

        # Probe the learned vector on a deterministic sparse sample of the establishment sequence
        # from zero transient state without altering evidence.
        saved=model.snapshot()
        model.reset_activation()
        step=max(1,len(est)//120)
        for date,features,next_ret,current_ret in est[::step]:
            receipt=model.interval(features)
            denom=receipt["p_up"]+receipt["p_dn"]+1e-15
            probe.append((receipt["p_up"]-receipt["p_dn"])/denom)
        model.restore(saved)

        probe_change=None
        if prev_epoch_probe is not None and probe:
            probe_change=sum(abs(a-b) for a,b in zip(probe,prev_epoch_probe))/len(probe)
        prev_epoch_probe=probe

        row={
            "epoch":epoch,
            "mean_surprise":statistics.mean(surprises),
            "probe_change":probe_change,
        }
        epoch_rows.append(row)

        if epoch>=MIN_EPOCHS and probe_change is not None and probe_change<2e-4:
            break

    # Re-run the established sequence once without learning so transient state ends exactly at the
    # end of known history under final evidence.
    model.reset_activation()
    for date,features,next_ret,current_ret in est:
        model.interval(features)

    labels=[]; preds=[]; flow_preds=[]; persist=[]
    strengths=[]; signed_expect=[]; surprises=[]
    correct_surprise=[]; wrong_surprise=[]
    rehearsal_counts=[]; dates=[]

    for date,features,next_ret,current_ret in online:
        pre=model.snapshot()

        # One and only one prediction before revealing this transition's outcome.
        receipt=model.interval(features)
        pred=1 if receipt["margin"]>=0 else 0
        flow_pred=1 if receipt["p_up"]>=receipt["p_dn"] else 0
        label=1 if next_ret>0 else 0
        denom=receipt["p_up"]+receipt["p_dn"]+1e-15
        signed=(receipt["p_up"]-receipt["p_dn"])/denom
        strength=abs(receipt["p_up"]-receipt["p_dn"])

        s=model.surprise(receipt,next_ret)

        labels.append(label); preds.append(pred); flow_preds.append(flow_pred)
        persist.append(1 if current_ret>=0 else 0)
        strengths.append(strength)
        signed_expect.append(signed)
        surprises.append(s)
        dates.append(date)
        (correct_surprise if pred==label else wrong_surprise).append(s)

        # Now the different/new datum is known. Rehearse that one transition from the same pre-day
        # state; then commit one interval and advance.
        reps,before,after=model.rehearse_transition(
            pre,features,next_ret,ONLINE_MAX_REHEARSAL
        )
        rehearsal_counts.append(reps)

    m=metrics(labels,preds)
    m["flow_prediction"]=metrics(labels,flow_preds)
    m["persistence"]=metrics(labels,persist)
    m["always_up"]=metrics(labels,[1]*len(labels))
    m["start_date"]=dates[0]
    m["end_date"]=dates[-1]
    m["establishment_examples"]=len(est)
    m["online_examples"]=len(online)
    m["epochs"]=epoch_rows
    m["epochs_used"]=len(epoch_rows)
    m["mean_rehearsals"]=statistics.mean(rehearsal_counts)
    m["mean_surprise"]=statistics.mean(surprises)
    m["surprise_correct"]=statistics.mean(correct_surprise) if correct_surprise else None
    m["surprise_wrong"]=statistics.mean(wrong_surprise) if wrong_surprise else None

    # Does stronger priming actually mean better prediction?
    order=sorted(range(len(strengths)),key=lambda i:strengths[i],reverse=True)
    for frac in (.1,.25,.5):
        k=max(1,int(len(order)*frac))
        ix=order[:k]
        m[f"top_{int(frac*100)}pct_strength_accuracy"]=sum(preds[i]==labels[i] for i in ix)/k
        m[f"top_{int(frac*100)}pct_strength_mean"]=statistics.mean(strengths[i] for i in ix)

    # Directional expectation vs actual signed outcome.
    ys=[1.0 if y else -1.0 for y in labels]
    if statistics.pstdev(signed_expect)>0:
        a=statistics.mean(signed_expect); b=statistics.mean(ys)
        cov=sum((x-a)*(y-b) for x,y in zip(signed_expect,ys))/len(ys)
        m["expectation_direction_corr"]=cov/(statistics.pstdev(signed_expect)*statistics.pstdev(ys))
    else:
        m["expectation_direction_corr"]=0.0

    return target,depth,m


def main():
    t0=time.perf_counter()
    dates,bars=align_data()
    print("common_data",len(dates),dates[0],dates[-1])

    examples={s:build_examples(dates,bars,s) for s in SYMBOLS}
    for s in SYMBOLS:
        est=sum(1 for x in examples[s] if x[0]<=ESTABLISH_END)
        online=len(examples[s])-est
        print("examples",s,len(examples[s]),"establish",est,"online",online)

    tasks=[(s,examples[s],d) for d in DEPTHS for s in SYMBOLS]
    workers=min(len(tasks),max(1,os.cpu_count() or 1))
    print("workers",workers)

    results={}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs={pool.submit(run_target,*t):(t[0],t[2]) for t in tasks}
        for fut in as_completed(futs):
            s,d,m=fut.result()
            results[(s,d)]=m
            print("RESULT",s,d,json.dumps(m,sort_keys=True),flush=True)

    for d in DEPTHS:
        rows=[results[(s,d)] for s in SYMBOLS]
        summary={
            "depth":d,
            "macro_accuracy":statistics.mean(r["accuracy"] for r in rows),
            "macro_balanced_accuracy":statistics.mean(r["balanced_accuracy"] for r in rows),
            "macro_flow_accuracy":statistics.mean(r["flow_prediction"]["accuracy"] for r in rows),
            "macro_flow_balanced_accuracy":statistics.mean(r["flow_prediction"]["balanced_accuracy"] for r in rows),
            "macro_always_up":statistics.mean(r["always_up"]["accuracy"] for r in rows),
            "macro_persistence":statistics.mean(r["persistence"]["accuracy"] for r in rows),
            "top10_strength_accuracy":statistics.mean(r["top_10pct_strength_accuracy"] for r in rows),
            "top25_strength_accuracy":statistics.mean(r["top_25pct_strength_accuracy"] for r in rows),
            "mean_epochs":statistics.mean(r["epochs_used"] for r in rows),
            "mean_rehearsals":statistics.mean(r["mean_rehearsals"] for r in rows),
            "mean_surprise":statistics.mean(r["mean_surprise"] for r in rows),
            "mean_expectation_corr":statistics.mean(r["expectation_direction_corr"] for r in rows),
        }
        print("SUMMARY",json.dumps(summary,sort_keys=True))

    print("total_seconds",time.perf_counter()-t0)
    print("all_assertions_passed")


if __name__=="__main__":
    main()
