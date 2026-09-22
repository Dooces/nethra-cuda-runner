from __future__ import annotations

import resource
import tempfile
from pathlib import Path

import core
from core import Nethra,NethraMemory,ResourceCloud
from world import GROUNDED_COUNT,MOTOR_COUNT,World,grounded_activation

def rss_mb()->float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0

def test_one_nethra_type():
    assert not hasattr(core,"PrimitiveNethra")
    assert not hasattr(core,"RelationNethra")
    mem=NethraMemory(GROUNDED_COUNT)
    assert len(mem.nodes)==GROUNDED_COUNT
    assert all(type(n) is Nethra for n in mem.nodes.values())

def test_bounded_cloud_under_large_activity():
    budget=64
    capacity=4096
    cloud=ResourceCloud(
        sample_rate=1.0,
        proposal_budget=budget,
        candidate_capacity=capacity,
        seed=11,
    )
    sources={i:1.0+(i%7)*.01 for i in range(20000)}
    consequences={30000+i:1.0+(i%11)*.01 for i in range(20000)}
    before=rss_mb()
    intervals=200
    for step in range(1,intervals+1):
        cloud.observe(sources,consequences,step)
        assert len(cloud)<=capacity
    after=rss_mb()
    receipt=cloud.receipt()
    assert receipt["proposals"]<=intervals*budget
    assert receipt["candidates"]<=capacity
    assert receipt["local_candidate_touches"]<=intervals*capacity
    assert after-before<512.0
    print("bounded_cloud_rss_delta_mb",after-before)
    print("bounded_cloud_receipt",receipt)

def test_activity_width_does_not_change_proposal_bound():
    for width in (100,1000,10000,50000):
        cloud=ResourceCloud(
            sample_rate=1.0,
            proposal_budget=32,
            candidate_capacity=2048,
            seed=width,
        )
        a={i:1.0 for i in range(width)}
        b={100000+i:1.0 for i in range(width)}
        for step in range(1,11):
            cloud.observe(a,b,step)
        assert cloud.proposals<=10*32
        assert len(cloud)<=2048
        print("width",width,"candidates",len(cloud),"proposals",cloud.proposals)

def test_output_nethra_current_is_direct():
    world=World()
    obs=world.step({0:.5,3:.25})
    a=grounded_activation(obs)
    assert obs.motor_currents[0]==.5
    assert obs.motor_currents[3]==.25
    assert a[0]==.5
    assert a[3]==.25

def test_constructed_is_same_nethra_and_field_reaches_output():
    cloud=ResourceCloud(sample_rate=1.0,proposal_budget=8,candidate_capacity=128,seed=5)
    # Feed a sparse relation long enough that candidate admission and evidence can recur.
    for step in range(1,500):
        if step%3==0:
            src={100:1.0}
            dst={0:1.0}
        else:
            src={101:1.0}
            dst={102:1.0}
        cloud.observe(src,dst,step)
    mem=NethraMemory(GROUNDED_COUNT)
    mem.checkpoint_from_cloud(cloud,500,persistence_floor=0.0)
    n=mem.constructed(100,0)
    assert n is not None
    assert type(n) is Nethra
    activation=mem.field({100:1.0},500)
    assert activation.get(0,0.0)>0.0

def test_checkpoint_roundtrip():
    cloud=ResourceCloud(sample_rate=1.0,proposal_budget=8,candidate_capacity=128,seed=7)
    for step in range(1,300):
        src={1:1.0} if step%3==0 else {2:1.0}
        dst={100:1.0} if step%3==0 else {101:1.0}
        cloud.observe(src,dst,step)
    mem=NethraMemory(GROUNDED_COUNT)
    mem.checkpoint_from_cloud(cloud,300,0.0)
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/"cp.json"
        mem.save(p,step=300)
        loaded,step,_=NethraMemory.load(p)
        assert step==300
        assert len(loaded.nodes)==len(mem.nodes)
        assert all(type(n) is Nethra for n in loaded.nodes.values())

def main():
    test_one_nethra_type()
    test_bounded_cloud_under_large_activity()
    test_activity_width_does_not_change_proposal_bound()
    test_output_nethra_current_is_direct()
    test_constructed_is_same_nethra_and_field_reaches_output()
    test_checkpoint_roundtrip()
    print("feed_v3 bounded invariants passed")
    print("process_max_rss_mb",rss_mb())

if __name__=="__main__":
    main()
