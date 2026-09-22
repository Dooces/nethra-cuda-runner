from __future__ import annotations
import json,platform
from pathlib import Path
from fixed_budget_persistence import run

def main():
    policies=[('full_f4_h15',4.,15000.,1.0),('full_f1_h15',1.,15000.,1.0),('full_f03_h15',.3,15000.,1.0),('full_f1_h30',1.,30000.,1.0)]
    rows=[]
    for p in policies:
        print('RUN',p[0],flush=True);rows.append(run(*p))
    out={'host':{'hostname':platform.node(),'platform':platform.platform(),'python':platform.python_version()},'runs':rows}
    Path('full_cloud_permissive.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'host':out['host'],'runs':[{k:r[k] for k in ('name','floor','half','sr','saved','motor_saved','g01','g05','g10','us','lookup_us')} for r in rows]},indent=2),flush=True)
if __name__=='__main__':main()
