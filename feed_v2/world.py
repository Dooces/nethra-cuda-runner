from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import FrozenSet, Iterable, Tuple

MOTOR_COUNT=12
JOINT_COUNT=6
RANGES=((0.0,1.65),(-0.9,0.9),(0.0,1.45),(0.0,1.45),(0.0,1.25),(0.0,1.35))
REST=(.75,0.0,.10,.10,.20,.10)
X_MIN,X_MAX=.28,1.24
Y_MIN,Y_MAX=-.24,.72
GRID_W,GRID_H=192,160
PIXELS=GRID_W*GRID_H
DYNAMIC_OFFSET=PIXELS
INPUT_COUNT=PIXELS+2*JOINT_COUNT
PRIMITIVE_COUNT=MOTOR_COUNT+INPUT_COUNT

@dataclass
class Ball:
    x:float;y:float;vx:float;vy:float

@dataclass(frozen=True)
class Obs:
    motors:FrozenSet[int]
    inputs:FrozenSet[int]
    hand_pixels:FrozenSet[int]
    ball_pixels:FrozenSet[int]
    contacts:int
    energy:float
    source_contact:bool

class World:
    SOURCE_X=.94
    SOURCE_Y=.24
    SOURCE_RADIUS=.045

    def __init__(self,*,balls=False,source=False,energy=False,seed=0,feed_rate=.0045,
                 basal_cost=.00042,motor_cost=.000055,initial_energy=.60):
        self.angle=list(REST);self.vel=[0.0]*JOINT_COUNT;self.prev_points=None
        self.rng=random.Random(seed);self.balls=[];self.source=bool(source);self.energy_enabled=bool(energy)
        self.energy=float(initial_energy);self.feed_rate=float(feed_rate)
        self.basal_cost=float(basal_cost);self.motor_cost=float(motor_cost)
        if balls:self.enable_balls()

    def enable_balls(self):
        if self.balls:return
        self.balls=[Ball(.46,.66,.055,-.02),Ball(.77,.71,-.045,-.01),Ball(1.07,.63,-.025,.015)]

    @staticmethod
    def _pixel(x,y):
        ix=max(0,min(GRID_W-1,int(round((x-X_MIN)/(X_MAX-X_MIN)*(GRID_W-1)))))
        iy=max(0,min(GRID_H-1,int(round((y-Y_MIN)/(Y_MAX-Y_MIN)*(GRID_H-1)))))
        return iy*GRID_W+ix

    def geometry(self)->Tuple[Tuple[float,float],...]:
        e,w,ii,mm,to,tf=self.angle
        def rot(t,x,y=0.0):
            c=math.cos(t);s=math.sin(t);return c*x-s*y,s*x+c*y
        def add(a,b):return a[0]+b[0],a[1]+b[1]
        elbow=(.55,0.0);wrist=add(elbow,rot(e,.42));o=e+w
        palm=add(wrist,rot(o,.06));ib=add(wrist,rot(o,.12,.025));mb=add(wrist,rot(o,.12,0));tb=add(wrist,rot(o,.036,-.025))
        return (palm,add(ib,rot(o-.95*ii,.10)),add(mb,rot(o-.95*mm,.11)),add(tb,rot(o-.90+to-.55*tf,.08)))

    def _joint_step(self,j,pos,neg):
        a=self.angle[j];v=self.vel[j];lo,hi=RANGES[j]
        authority=(.06+.94*self.energy) if self.energy_enabled else 1.0
        accel=2.1*authority*(float(pos)-float(neg))-1.9*v-.35*a
        v+=.10*accel;a+=.10*v
        if a<lo:a=lo;v=0.0
        elif a>hi:a=hi;v=0.0
        self.angle[j]=a;self.vel[j]=v

    def _balls(self,points,prev):
        if not self.balls:return 0
        dt=.10;gravity=-.32;radius=.024;hand_r=.025;contacts=0
        pv=[(0.0,0.0)]*len(points) if prev is None else [((p[0]-q[0])/dt,(p[1]-q[1])/dt) for p,q in zip(points,prev)]
        for b in self.balls:
            b.vy+=gravity*dt;b.x+=b.vx*dt;b.y+=b.vy*dt
            if b.x<X_MIN+radius:b.x=X_MIN+radius;b.vx=abs(b.vx)*.94
            elif b.x>X_MAX-radius:b.x=X_MAX-radius;b.vx=-abs(b.vx)*.94
            if b.y<Y_MIN+radius:b.y=Y_MIN+radius;b.vy=abs(b.vy)*.92
            elif b.y>Y_MAX-radius:b.y=Y_MAX-radius;b.vy=-abs(b.vy)*.92
            for (px,py),(hvx,hvy) in zip(points,pv):
                dx=b.x-px;dy=b.y-py;d2=dx*dx+dy*dy;lim=radius+hand_r
                if d2<=lim*lim:
                    contacts+=1;d=math.sqrt(max(d2,1e-12));nx=dx/d;ny=dy/d
                    b.x=px+nx*lim;b.y=py+ny*lim
                    vn=(b.vx-hvx)*nx+(b.vy-hvy)*ny
                    if vn<0.0:
                        b.vx-=1.88*vn*nx;b.vy-=1.88*vn*ny
                    b.vx+=.18*hvx;b.vy+=.18*hvy
        return contacts

    def step(self,motors:Iterable[int])->Obs:
        motors=frozenset(map(int,motors))
        for j in range(JOINT_COUNT):self._joint_step(j,2*j in motors,2*j+1 in motors)
        points=self.geometry();contacts=self._balls(points,self.prev_points);self.prev_points=points
        hand=frozenset(self._pixel(x,y) for x,y in points);ball=frozenset(self._pixel(b.x,b.y) for b in self.balls)
        inputs=set(hand)|set(ball)
        for j,v in enumerate(self.vel):
            if v>.020:inputs.add(DYNAMIC_OFFSET+2*j)
            elif v<-.020:inputs.add(DYNAMIC_OFFSET+2*j+1)
        source_contact=False
        if self.source:
            sp=self._pixel(self.SOURCE_X,self.SOURCE_Y);inputs.add(sp)
            r2=self.SOURCE_RADIUS*self.SOURCE_RADIUS
            source_contact=any((x-self.SOURCE_X)**2+(y-self.SOURCE_Y)**2<=r2 for x,y in points)
        if self.energy_enabled:
            self.energy-=self.basal_cost+self.motor_cost*len(motors)
            if self.source and source_contact:self.energy+=self.feed_rate
            self.energy=max(0.0,min(1.0,self.energy))
        return Obs(motors,frozenset(inputs),hand,ball,contacts,self.energy,source_contact)

def support(obs:Obs)->FrozenSet[int]:
    return frozenset(obs.motors)|frozenset(MOTOR_COUNT+i for i in obs.inputs)
