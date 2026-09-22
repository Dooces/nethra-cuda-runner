from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

import core
from core import Nethra, NethraMemory, ResourceCloud
from world import PRIMITIVE_COUNT


def test_one_node_type():
    assert not hasattr(core,"PrimitiveNethra")
    assert not hasattr(core,"RelationNethra")
    mem=NethraMemory(PRIMITIVE_COUNT)
    assert len(mem.nodes)==PRIMITIVE_COUNT
    assert all(type(n) is Nethra for n in mem.nodes.values())


def test_constructed_is_same_type():
    c=ResourceCloud(1.0,1)
    for t in range(1,400):
        xs={1} if t%4==0 else {2}
        ys={100} if 1 in xs else set()
        c.observe(xs,ys,t)
    mem=NethraMemory(PRIMITIVE_COUNT)
    out=mem.checkpoint_from_cloud(c,400,0.0)
    assert out["created"]>=1
    n=mem.constructed(1,100)
    assert type(n) is Nethra
    assert n.members==(1,100)


def test_same_field_reaches_output_nethra():
    c=ResourceCloud(1.0,2)
    for t in range(1,500):
        xs={100} if t%3==0 else {101}
        ys={0} if 100 in xs else set()
        c.observe(xs,ys,t)
    mem=NethraMemory(PRIMITIVE_COUNT)
    mem.checkpoint_from_cloud(c,500,0.0)
    a=mem.field({100},500)
    assert a.get(0,0.0)>0.0


def test_checkpoint_regenerates_same_nethra_type():
    c=ResourceCloud(1.0,3)
    for t in range(1,300):
        c.observe({1} if t%3==0 else {2},{100} if t%3==0 else set(),t)
    mem=NethraMemory(PRIMITIVE_COUNT)
    mem.checkpoint_from_cloud(c,300,0.0)
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/"cp.json";mem.save(p,step=300)
        q,step,_=NethraMemory.load(p)
        assert step==300
        assert all(type(n) is Nethra for n in q.nodes.values())
        assert len(q.nodes)==len(mem.nodes)


if __name__=="__main__":
    test_one_node_type()
    test_constructed_is_same_type()
    test_same_field_reaches_output_nethra()
    test_checkpoint_regenerates_same_nethra_type()
    print("feed_v2 unified Nethra tests passed")
