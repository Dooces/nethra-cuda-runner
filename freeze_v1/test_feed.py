from __future__ import annotations
from core import NethraMemory, ResourceCloud
from run_feeding import MotorResonanceIndex
from world import BASE_COUNT, MOTOR_COUNT

def main():
    c=ResourceCloud(1.0,77)
    for t in range(1,500):
        xs={100,101}
        if t%3==0:xs.add(102)
        ys={0} if t%2==0 else {1}
        c.observe(xs,ys,t)
    m=NethraMemory(BASE_COUNT)
    m.checkpoint_from_cloud(c,500,0.0)
    active={100,101,102}
    generic=m.resonance(active,500)
    indexed=MotorResonanceIndex(m).scores(m,active,500)
    for motor in range(MOTOR_COUNT):
        assert abs(generic.get(motor,0.0)-indexed.get(motor,0.0)) < 1e-12
    print("feed execution-index equivalence passed")

if __name__=="__main__":
    main()
