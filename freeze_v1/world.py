from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import FrozenSet, Iterable, Tuple

MOTOR_COUNT = 12
JOINT_COUNT = 6
RANGES = ((0.0,1.65),(-0.9,0.9),(0.0,1.45),(0.0,1.45),(0.0,1.25),(0.0,1.35))
REST = (.75,0.0,.10,.10,.20,.10)
X_MIN,X_MAX=.28,1.24
Y_MIN,Y_MAX=-.24,.72
GRID_W,GRID_H=192,160
PIXELS=GRID_W*GRID_H
DYNAMIC_OFFSET=PIXELS
INPUT_COUNT=PIXELS+2*JOINT_COUNT
BASE_COUNT=MOTOR_COUNT+INPUT_COUNT

@dataclass
class Ball:
    x: float
    y: float
    vx: float
    vy: float

@dataclass(frozen=True)
class Obs:
    motors: FrozenSet[int]
    inputs: FrozenSet[int]
    hand_pixels: FrozenSet[int]
    ball_pixels: FrozenSet[int]
    contacts: int

class HandBallWorld:
    """Physics and raw transducers only. Evaluator fields are never fed to the learner."""

    def __init__(self, *, balls: bool = False, seed: int = 0):
        self.angle=list(REST)
        self.vel=[0.0]*JOINT_COUNT
        self.prev_points=None
        self.rng=random.Random(int(seed))
        self.balls:list[Ball]=[]
        if balls:
            self.enable_balls()

    def enable_balls(self) -> None:
        if self.balls:
            return
        self.balls=[
            Ball(.46,.66,.055,-.02),
            Ball(.77,.71,-.045,-.01),
            Ball(1.07,.63,-.025,.015),
        ]

    def _joint_step(self,j:int,pos:bool,neg:bool)->None:
        a=self.angle[j];v=self.vel[j];lo,hi=RANGES[j]
        accel=2.1*(float(pos)-float(neg))-1.9*v-.35*a
        v+=.10*accel;a+=.10*v
        if a<lo:a=lo;v=0.0
        elif a>hi:a=hi;v=0.0
        self.angle[j]=a;self.vel[j]=v

    def geometry(self)->Tuple[Tuple[float,float],...]:
        elbow_a,wrist_a,index_a,middle_a,thumb_o,thumb_f=self.angle
        def rot(t,x,y=0.0):
            c=math.cos(t);s=math.sin(t);return c*x-s*y,s*x+c*y
        def add(a,b):return a[0]+b[0],a[1]+b[1]
        elbow=(.55,0.0)
        wrist=add(elbow,rot(elbow_a,.42))
        o=elbow_a+wrist_a
        palm=add(wrist,rot(o,.06))
        ib=add(wrist,rot(o,.12,.025))
        mb=add(wrist,rot(o,.12,0))
        tb=add(wrist,rot(o,.036,-.025))
        index=add(ib,rot(o-.95*index_a,.10))
        middle=add(mb,rot(o-.95*middle_a,.11))
        thumb=add(tb,rot(o-.90+thumb_o-.55*thumb_f,.08))
        return palm,index,middle,thumb

    @staticmethod
    def _pixel(x:float,y:float)->int:
        ix=max(0,min(GRID_W-1,int(round((x-X_MIN)/(X_MAX-X_MIN)*(GRID_W-1)))))
        iy=max(0,min(GRID_H-1,int(round((y-Y_MIN)/(Y_MAX-Y_MIN)*(GRID_H-1)))))
        return iy*GRID_W+ix

    def _step_balls(self, points:Tuple[Tuple[float,float],...], prev_points)->int:
        if not self.balls:
            return 0
        dt=.10; gravity=-.32; radius=.024; hand_r=.025; contacts=0
        point_vel=[]
        if prev_points is None:
            point_vel=[(0.0,0.0)]*len(points)
        else:
            point_vel=[((p[0]-q[0])/dt,(p[1]-q[1])/dt) for p,q in zip(points,prev_points)]
        for b in self.balls:
            b.vy += gravity*dt
            b.x += b.vx*dt
            b.y += b.vy*dt
            if b.x < X_MIN+radius:
                b.x=X_MIN+radius;b.vx=abs(b.vx)*.94
            elif b.x > X_MAX-radius:
                b.x=X_MAX-radius;b.vx=-abs(b.vx)*.94
            if b.y < Y_MIN+radius:
                b.y=Y_MIN+radius;b.vy=abs(b.vy)*.92
            elif b.y > Y_MAX-radius:
                b.y=Y_MAX-radius;b.vy=-abs(b.vy)*.92

            for (px,py),(hvx,hvy) in zip(points,point_vel):
                dx=b.x-px;dy=b.y-py;d2=dx*dx+dy*dy
                limit=radius+hand_r
                if d2 <= limit*limit:
                    contacts+=1
                    d=math.sqrt(max(d2,1e-12));nx=dx/d;ny=dy/d
                    b.x=px+nx*limit;b.y=py+ny*limit
                    rvx=b.vx-hvx;rvy=b.vy-hvy
                    vn=rvx*nx+rvy*ny
                    if vn<0.0:
                        restitution=.88
                        b.vx -= (1.0+restitution)*vn*nx
                        b.vy -= (1.0+restitution)*vn*ny
                    b.vx += .18*hvx;b.vy += .18*hvy
        return contacts

    def step(self,motors:Iterable[int])->Obs:
        motors=frozenset(map(int,motors))
        for j in range(JOINT_COUNT):
            self._joint_step(j,2*j in motors,2*j+1 in motors)
        points=self.geometry()
        contacts=self._step_balls(points,self.prev_points)
        self.prev_points=points
        hand_pixels=frozenset(self._pixel(x,y) for x,y in points)
        ball_pixels=frozenset(self._pixel(b.x,b.y) for b in self.balls)
        inputs=set(hand_pixels)|set(ball_pixels)
        for j,v in enumerate(self.vel):
            if v>.020:inputs.add(DYNAMIC_OFFSET+2*j)
            elif v<-.020:inputs.add(DYNAMIC_OFFSET+2*j+1)
        return Obs(motors,frozenset(inputs),hand_pixels,ball_pixels,contacts)

def support(obs:Obs)->FrozenSet[int]:
    return frozenset(obs.motors)|frozenset(MOTOR_COUNT+i for i in obs.inputs)
