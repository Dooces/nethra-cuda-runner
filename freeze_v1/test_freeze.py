from __future__ import annotations
import tempfile
from pathlib import Path
from core import NethraMemory, ResourceCloud
from world import BASE_COUNT, HandBallWorld, support

def test_singleton_competes_without_creation_threshold():
    c=ResourceCloud(1.0,1)
    for t in range(1,200):
        xs={1} if t%4==0 else {2}
        ys={100} if 1 in xs else set()
        c.observe(xs,ys,t)
    assert c.evidence.get(100,{}).get(1,0.0)>0.0

def test_checkpoint_roundtrip():
    c=ResourceCloud(1.0,2)
    for t in range(1,300):
        c.observe({1},{100},t)
    m=NethraMemory(BASE_COUNT)
    out=m.checkpoint_from_cloud(c,300,0.0)
    assert out["created"]>=1
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/"cp.json";m.save(p,step=300,metadata={"x":1})
        q,step,meta=NethraMemory.load(p)
        assert step==300 and meta["x"]==1 and len(q.by_key)==len(m.by_key)

def test_balls_are_only_raw_retinal_activity():
    w=HandBallWorld(balls=True,seed=3)
    o=w.step(set())
    assert o.ball_pixels
    assert o.ball_pixels.issubset(o.inputs)
    assert max(o.ball_pixels)<30720

if __name__=="__main__":
    test_singleton_competes_without_creation_threshold()
    test_checkpoint_roundtrip()
    test_balls_are_only_raw_retinal_activity()
    print("freeze_v1 tests passed")
