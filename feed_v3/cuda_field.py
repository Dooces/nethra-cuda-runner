from __future__ import annotations

import math
from typing import Dict, Mapping

import numpy as np

from core import NethraMemory


class CudaFieldRuntime:
    """CUDA execution of NethraMemory.field_step with identical graph semantics.

    The Nethra graph remains canonical in NethraMemory. This class is only a compiled execution
    projection: fixed sparse incidence topology plus per-step conductances derived from the same
    weights, decay law, leakage, and timestep.
    """

    def __init__(
        self,
        memory:NethraMemory,
        *,
        execution_epsilon:float=1e-6,
        dtype:str="float32",
    ):
        import cupy as cp
        import cupyx.scipy.sparse as cps

        self.cp=cp
        self.cps=cps
        self.memory=memory
        self.execution_epsilon=float(execution_epsilon)
        self.dtype=cp.float32 if dtype=="float32" else cp.float64
        self.n=max(memory.nodes)+1 if memory.nodes else 0

        relations=[n for n in memory.nodes.values() if n.members]
        relations.sort(key=lambda n:n.nid)
        self.relation_ids=np.asarray([n.nid for n in relations],dtype=np.int64)
        self.weights=cp.asarray([n.weight for n in relations],dtype=self.dtype)
        self.last_steps=cp.asarray([n.last_step for n in relations],dtype=self.dtype)

        rows=[]
        cols=[]
        rel_index=[]
        for ri,n in enumerate(relations):
            for member in n.members:
                rows.append(n.nid);cols.append(int(member));rel_index.append(ri)
                rows.append(int(member));cols.append(n.nid);rel_index.append(ri)

        if rows:
            rows=np.asarray(rows,dtype=np.int64)
            cols=np.asarray(cols,dtype=np.int64)
            rel_index=np.asarray(rel_index,dtype=np.int64)
            order=np.lexsort((cols,rows))
            rows=rows[order]
            cols=cols[order]
            rel_index=rel_index[order]
            counts=np.bincount(rows,minlength=self.n)
            indptr=np.empty(self.n+1,dtype=np.int64)
            indptr[0]=0
            np.cumsum(counts,out=indptr[1:])
            self.rows=cp.asarray(rows)
            self.edge_relation=cp.asarray(rel_index)
            self.indptr=cp.asarray(indptr)
            self.indices=cp.asarray(cols)
            self.data=cp.zeros(len(rows),dtype=self.dtype)
            self.matrix=cps.csr_matrix(
                (self.data,self.indices,self.indptr),
                shape=(self.n,self.n),
            )
            self.data=self.matrix.data
        else:
            self.rows=cp.empty(0,dtype=cp.int64)
            self.edge_relation=cp.empty(0,dtype=cp.int64)
            self.indptr=cp.zeros(self.n+1,dtype=cp.int64)
            self.indices=cp.empty(0,dtype=cp.int64)
            self.data=cp.empty(0,dtype=self.dtype)
            self.matrix=cps.csr_matrix((self.n,self.n),dtype=self.dtype)

        self.state=cp.zeros(self.n,dtype=self.dtype)
        self.source=cp.zeros(self.n,dtype=self.dtype)
        self.inbound=cp.zeros(self.n,dtype=self.dtype)
        self.degree=cp.zeros(self.n,dtype=self.dtype)
        self._source_ids=cp.empty(0,dtype=cp.int64)

    def rebuild(self,memory:NethraMemory)->"CudaFieldRuntime":
        return CudaFieldRuntime(
            memory,
            execution_epsilon=self.execution_epsilon,
            dtype="float32" if self.dtype==self.cp.float32 else "float64",
        )

    def set_state(self,activation:Mapping[int,float])->None:
        cp=self.cp
        self.state.fill(0)
        if activation:
            ids=np.fromiter(activation.keys(),dtype=np.int64,count=len(activation))
            vals=np.fromiter(activation.values(),dtype=np.float64,count=len(activation))
            self.state[cp.asarray(ids)]=cp.asarray(vals,dtype=self.dtype)

    def _conductance(self,step:int):
        cp=self.cp
        if self.weights.size==0:
            return self.weights
        dt=cp.maximum(0.0,float(step)-self.last_steps)
        w=self.weights*cp.exp2(-dt/float(self.memory.half_life))
        return 1.5*(1.0-cp.exp(-cp.maximum(0.0,w)/float(self.memory.tau)))

    def step(
        self,
        source_current:Mapping[int,float],
        step:int,
        *,
        dt:float=0.1,
        return_active:bool=True,
    )->Dict[int,float] | None:
        cp=self.cp
        if dt<=0.0:
            raise ValueError("dt must be positive")

        if self._source_ids.size:
            self.source[self._source_ids]=0
        if source_current:
            ids=np.fromiter(source_current.keys(),dtype=np.int64,count=len(source_current))
            vals=np.fromiter(source_current.values(),dtype=np.float64,count=len(source_current))
            gpu_ids=cp.asarray(ids)
            self.source[gpu_ids]=cp.asarray(vals,dtype=self.dtype)
            self._source_ids=gpu_ids
        else:
            self._source_ids=cp.empty(0,dtype=cp.int64)

        if self.data.size:
            g=self._conductance(step)
            self.data[:]=g[self.edge_relation]
            self.inbound[:]=self.matrix@self.state
            self.degree[:]=cp.bincount(
                self.rows,
                weights=self.data,
                minlength=self.n,
            )
        else:
            self.inbound.fill(0)
            self.degree.fill(0)

        denom=1.0+float(dt)*(float(self.memory.leakage)+self.degree)
        self.state[:]=(self.state+float(dt)*(self.source+self.inbound))/denom
        self.state[self.state<=self.execution_epsilon]=0

        if return_active:
            return self.snapshot()
        return None

    def snapshot(self)->Dict[int,float]:
        active=self.cp.flatnonzero(self.state)
        ids=self.cp.asnumpy(active)
        vals=self.cp.asnumpy(self.state[active])
        return {int(i):float(v) for i,v in zip(ids,vals)}

    def motor_values(self,count:int=12)->np.ndarray:
        return self.cp.asnumpy(self.state[:int(count)])

    def active_count(self)->int:
        return int(self.cp.count_nonzero(self.state).item())

    def synchronize(self)->None:
        self.cp.cuda.Device().synchronize()
