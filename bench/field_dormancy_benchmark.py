from __future__ import annotations
import json,platform,time
from pathlib import Path
import numpy as np
from scipy import sparse

P=30732
CASES=[
    ('floor1',11485,6176),
    ('floor03',93633,8755),
    ('floor01',385071,8898),
]

def build(relations:int,seed:int=123):
    rng=np.random.default_rng(seed)
    n=P+relations
    a=rng.integers(0,P,size=relations,dtype=np.int64)
    b=rng.integers(0,P,size=relations,dtype=np.int64)
    same=a==b
    b[same]=(b[same]+1)%P
    rn=P+np.arange(relations,dtype=np.int64)
    rows=np.concatenate([np.arange(n,dtype=np.int64),rn,rn,a,b])
    cols=np.concatenate([np.arange(n,dtype=np.int64),a,b,rn,rn])
    vals=np.concatenate([
        np.full(n,-0.10,dtype=np.float64),
        np.full(relations,0.20,dtype=np.float64),
        np.full(relations,0.20,dtype=np.float64),
        np.full(relations,0.20,dtype=np.float64),
        np.full(relations,0.20,dtype=np.float64),
    ])
    return sparse.csr_matrix((vals,(rows,cols)),shape=(n,n))

def bench(relations:int,loops:int=200,repeats:int=5):
    t0=time.perf_counter();M=build(relations);build_s=time.perf_counter()-t0
    x=np.ones(M.shape[0],dtype=np.float64)
    for _ in range(10):x=M@x
    samples=[]
    for _ in range(repeats):
        x.fill(1.0);a=time.perf_counter()
        for __ in range(loops):x=M@x
        samples.append((time.perf_counter()-a)/loops)
    mem=M.data.nbytes+M.indices.nbytes+M.indptr.nbytes
    return {'relations_in_field':relations,'dimension':M.shape[0],'nnz':M.nnz,'csr_bytes':int(mem),'build_s':build_s,'matvec_us_median':1e6*float(np.median(samples)),'matvec_us_min':1e6*float(np.min(samples))}

def main():
    out={'host':{'hostname':platform.node(),'platform':platform.platform(),'python':platform.python_version(),'numpy':np.__version__},'scipy':None,'cases':[]}
    import scipy;out['scipy']=scipy.__version__
    for name,saved,active in CASES:
        print('CASE',name,'full',saved,flush=True);full=bench(saved)
        print('CASE',name,'active',active,flush=True);act=bench(active)
        out['cases'].append({'name':name,'saved_identities':saved,'active_identities':active,'full_field':full,'active_only_field':act,'speedup':full['matvec_us_median']/act['matvec_us_median'],'memory_ratio':full['csr_bytes']/act['csr_bytes']})
    Path('field_dormancy_results.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
