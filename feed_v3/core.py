from __future__ import annotations

import bisect
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple

MASK=(1<<64)-1

def _u01(t:int,seed:int)->float:
    z=(int(t)+0x9E3779B97F4A7C15*(int(seed)+1))&MASK
    z=((z^(z>>30))*0xBF58476D1CE4E5B9)&MASK
    z=((z^(z>>27))*0x94D049BB133111EB)&MASK
    z^=z>>31
    return ((z>>11)&((1<<53)-1))/float(1<<53)

@dataclass(slots=True)
class Nethra:
    """The only graph node type."""
    nid:int
    members:Tuple[int,...]=()
    weight:float=0.0
    last_step:int=0
    confirmations:int=0

@dataclass(slots=True)
class Candidate:
    """Ephemeral directional evidence for one possible relation."""
    cid:int
    source:int
    consequence:int
    born_interval:int
    source_seen:int=1
    consequence_seen:int=1
    joint_seen:int=1
    evidence:float=0.0
    joint_updates:int=0

class ResourceCloud:
    """Bounded prospective competition.

    proposal_budget bounds new pair proposals per sampled interval.
    candidate_capacity bounds all ephemeral candidate state.
    Neither value is a truth threshold.
    """

    def __init__(
        self,
        *,
        sample_rate:float=0.0625,
        proposal_budget:int=64,
        candidate_capacity:int=65536,
        seed:int=0,
    ):
        if not (0.0<sample_rate<=1.0):
            raise ValueError("sample_rate must be in (0,1]")
        if proposal_budget<1 or candidate_capacity<1:
            raise ValueError("budgets must be positive")
        self.sample_rate=float(sample_rate)
        self.proposal_budget=int(proposal_budget)
        self.candidate_capacity=int(candidate_capacity)
        self.seed=int(seed)
        self.rng=random.Random(self.seed)
        self.n=0
        self.next_cid=0
        self.by_pair:Dict[Tuple[int,int],int]={}
        self.candidates:Dict[int,Candidate]={}
        self.by_source:Dict[int,set[int]]={}
        self.by_consequence:Dict[int,set[int]]={}
        self.proposals=0
        self.admitted=0
        self.dropped=0
        self.local_candidate_touches=0

    @staticmethod
    def _positive_items(a:Mapping[int,float])->tuple[list[int],list[float]]:
        ids=[]
        weights=[]
        for nid,value in a.items():
            w=float(value)
            if w>0.0:
                ids.append(int(nid))
                weights.append(w)
        return ids,weights

    @staticmethod
    def _cumulative(weights:list[float])->list[float]:
        out=[]
        total=0.0
        for w in weights:
            total+=w
            out.append(total)
        return out

    def _draw(self,ids:list[int],cumulative:list[float])->int:
        r=self.rng.random()*cumulative[-1]
        return ids[bisect.bisect_left(cumulative,r)]

    def _admit(self,source:int,consequence:int)->None:
        if source==consequence:
            return
        key=(int(source),int(consequence))
        if key in self.by_pair:
            return
        if len(self.candidates)>=self.candidate_capacity:
            self.dropped+=1
            return
        cid=self.next_cid
        self.next_cid+=1
        c=Candidate(cid,key[0],key[1],self.n)
        self.by_pair[key]=cid
        self.candidates[cid]=c
        self.by_source.setdefault(c.source,set()).add(cid)
        self.by_consequence.setdefault(c.consequence,set()).add(cid)
        self.admitted+=1

    def observe(
        self,
        sources:Mapping[int,float],
        consequences:Mapping[int,float],
        step:int,
    )->bool:
        if _u01(step,self.seed)>=self.sample_rate:
            return False
        if not sources or not consequences:
            return False

        self.n+=1
        source_ids=set(map(int,sources.keys()))
        consequence_ids=set(map(int,consequences.keys()))

        touched:set[int]=set()
        for x in source_ids:
            for cid in self.by_source.get(x,()):
                self.candidates[cid].source_seen+=1
                touched.add(cid)
        for y in consequence_ids:
            for cid in self.by_consequence.get(y,()):
                self.candidates[cid].consequence_seen+=1
                touched.add(cid)
        self.local_candidate_touches+=len(touched)

        gains_by_consequence:Dict[int,list[tuple[int,float]]]={}
        for cid in touched:
            c=self.candidates[cid]
            if c.source not in source_ids or c.consequence not in consequence_ids:
                continue
            c.joint_seen+=1
            c.joint_updates+=1
            age=max(1,self.n-c.born_interval+1)
            conditional=c.joint_seen/max(1,c.source_seen)
            baseline=c.consequence_seen/age
            gain=math.log(conditional/baseline) if conditional>0.0 and baseline>0.0 else 0.0
            if gain>0.0:
                gains_by_consequence.setdefault(c.consequence,[]).append((cid,gain))

        for group in gains_by_consequence.values():
            total=sum(g for _,g in group)
            if total<=0.0:
                continue
            inv=1.0/total
            for cid,gain in group:
                self.candidates[cid].evidence+=gain*inv

        xids,xw=self._positive_items(sources)
        yids,yw=self._positive_items(consequences)
        if not xids or not yids:
            return True
        xc=self._cumulative(xw)
        yc=self._cumulative(yw)

        proposed:set[Tuple[int,int]]=set()
        attempts=0
        max_attempts=self.proposal_budget*3
        while len(proposed)<self.proposal_budget and attempts<max_attempts:
            attempts+=1
            key=(self._draw(xids,xc),self._draw(yids,yc))
            if key[0]!=key[1]:
                proposed.add(key)
        self.proposals+=len(proposed)
        for source,consequence in proposed:
            self._admit(source,consequence)
        return True

    def __len__(self)->int:
        return len(self.candidates)

    def receipt(self)->dict:
        return {
            "sampled_intervals":self.n,
            "candidates":len(self.candidates),
            "capacity":self.candidate_capacity,
            "proposal_budget":self.proposal_budget,
            "proposals":self.proposals,
            "admitted":self.admitted,
            "dropped":self.dropped,
            "local_candidate_touches":self.local_candidate_touches,
        }

class NethraMemory:
    """One append-only Nethra graph for grounded and constructed Nethra."""

    def __init__(self,grounded_count:int,*,half_life:float=60000.0,tau:float=100.0):
        self.grounded_count=int(grounded_count)
        self.half_life=float(half_life)
        self.tau=float(tau)
        self.nodes:Dict[int,Nethra]={i:Nethra(i) for i in range(self.grounded_count)}
        self.next_nid=self.grounded_count
        self.by_members:Dict[Tuple[int,int],int]={}
        self.index:Dict[int,list[int]]={}

    @staticmethod
    def member_key(a:int,b:int)->Tuple[int,int]:
        a=int(a);b=int(b)
        return (a,b) if a<b else (b,a)

    def node(self,nid:int)->Nethra:
        return self.nodes[int(nid)]

    def effective_weight(self,n:Nethra,step:int)->float:
        if not n.members or self.half_life<=0.0:
            return n.weight
        dt=max(0,int(step)-int(n.last_step))
        return n.weight*(2.0**(-dt/self.half_life))

    def conductance(self,n:Nethra,step:int)->float:
        w=self.effective_weight(n,step)
        return 1.5*(1.0-math.exp(-max(0.0,w)/self.tau))

    def _index(self,n:Nethra)->None:
        for member in n.members:
            self.index.setdefault(member,[]).append(n.nid)

    def constructed(self,a:int,b:int)->Nethra|None:
        nid=self.by_members.get(self.member_key(a,b))
        return self.nodes.get(nid) if nid is not None else None

    def checkpoint_from_cloud(
        self,
        cloud:ResourceCloud,
        step:int,
        persistence_floor:float=0.003,
    )->dict:
        floor=float(persistence_floor)
        best:Dict[Tuple[int,int],float]={}
        for c in cloud.candidates.values():
            if c.source==c.consequence:
                continue
            key=self.member_key(c.source,c.consequence)
            ev=float(c.evidence)
            if ev>best.get(key,0.0):
                best[key]=ev

        created=0
        reinforced=0
        sampled=max(1,cloud.n)
        for key,ev in best.items():
            density=1000.0*ev/sampled
            existing=self.by_members.get(key)
            if existing is None:
                if density<floor:
                    continue
                n=Nethra(
                    nid=self.next_nid,
                    members=key,
                    weight=ev/max(cloud.sample_rate,1e-12),
                    last_step=int(step),
                    confirmations=1,
                )
                self.next_nid+=1
                self.nodes[n.nid]=n
                self.by_members[key]=n.nid
                self._index(n)
                created+=1
            else:
                n=self.nodes[existing]
                n.weight=self.effective_weight(n,step)+ev/max(cloud.sample_rate,1e-12)
                n.last_step=int(step)
                n.confirmations+=1
                reinforced+=1
        return {
            **cloud.receipt(),
            "created":created,
            "reinforced":reinforced,
            "constructed":len(self.nodes)-self.grounded_count,
            "total_nethra":len(self.nodes),
        }

    def field(self,externally_active:Mapping[int,float],step:int)->Dict[int,float]:
        """One local symmetric resonance pass with no node-type branch."""
        activation={int(nid):float(a) for nid,a in externally_active.items() if float(a)>0.0}
        touched:set[int]=set()
        for nid in activation:
            touched.update(self.index.get(nid,()))

        constructed_activation:Dict[int,float]={}
        for nid in touched:
            n=self.nodes[nid]
            g=self.conductance(n,step)
            incoming=sum(activation.get(member,0.0) for member in n.members)
            if g>0.0 and incoming>0.0:
                constructed_activation[nid]=g*incoming

        activation.update(constructed_activation)
        for nid,a in constructed_activation.items():
            n=self.nodes[nid]
            g=self.conductance(n,step)
            for member in n.members:
                activation[member]=activation.get(member,0.0)+g*a
        return activation

    def save(self,path:str|Path,*,step:int,metadata:dict|None=None)->None:
        constructed=[
            {
                "nid":n.nid,
                "members":list(n.members),
                "weight":n.weight,
                "last_step":n.last_step,
                "confirmations":n.confirmations,
            }
            for n in self.nodes.values() if n.members
        ]
        obj={
            "schema":"NETHRA_BOUNDED_V1",
            "grounded_count":self.grounded_count,
            "half_life":self.half_life,
            "tau":self.tau,
            "next_nid":self.next_nid,
            "step":int(step),
            "constructed":constructed,
            "metadata":dict(metadata or {}),
        }
        p=Path(path)
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(obj,separators=(",",":"))+"\n")

    @classmethod
    def load(cls,path:str|Path)->tuple["NethraMemory",int,dict]:
        obj=json.loads(Path(path).read_text())
        if obj.get("schema")!="NETHRA_BOUNDED_V1":
            raise RuntimeError("unsupported checkpoint schema")
        mem=cls(obj["grounded_count"],half_life=obj["half_life"],tau=obj["tau"])
        mem.next_nid=int(obj["next_nid"])
        for row in obj["constructed"]:
            n=Nethra(
                nid=int(row["nid"]),
                members=tuple(map(int,row["members"])),
                weight=float(row["weight"]),
                last_step=int(row["last_step"]),
                confirmations=int(row["confirmations"]),
            )
            mem.nodes[n.nid]=n
            mem.by_members[tuple(n.members)]=n.nid
            mem._index(n)
        return mem,int(obj["step"]),dict(obj.get("metadata",{}))
