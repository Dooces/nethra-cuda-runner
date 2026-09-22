from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, FrozenSet, Mapping, Tuple

MOTOR_COUNT=12
JOINT_COUNT=6
RANGES=((0.0,1.65),(-0.9,0.9),(0.0,1.45),(0.0,1.45),(0.0,1.25),(0.0,1.35))
REST=(.75,0.0,.10,.10,.20,.10)
X_MIN,X_MAX=.28,1.24
Y_MIN,Y_MAX=-.24,.72
GRID_W,GRID_H=192,160
PIXELS=GRID_W*GRID_H
DYNAMIC_OFFSET=PIXELS
SENSORY_COUNT=PIXELS+2*JOINT_COUNT
GROUNDED_COUNT=MOTOR_COUNT+SENSORY_COUNT

@dataclass(slots=True)
class Ball:
    x:float
    y:float
    vx:float
    vy:float

@dataclass(frozen=True,slots=True)
class Obs:
    motor_currents:Tuple[float,...]
    inputs:FrozenSet[int]
    hand_pixels:FrozenSet[int]
    ball_pixels:FrozenSet[int]
    contacts:int
    energy:float
    source_contact:bool

class World:
    """Physical environment only. Hidden evaluator state never enters learner activation."""

    SOURCE_X=.94
    SOURCE_Y=.24
    SOURCE_RADIUS=.045

    def __init__(
        self,
        *,
        balls:bool=False,
        source:bool=False,
        energy:bool=False,
        feed_rate:float=.0045,
        basal_cost:float=.00042,
        motor_cost:float=.000055,
        initial_energy:float=.60,
    ):
        self.angle=list(REST)
        self.vel=[0.0]*JOINT_COUNT
        self.prev_points=None
        self.balls:list[Ball]=[]
        self.source=bool(source)
        self.energy_enabled=bool(energy)
        self.energy=float(initial_energy)
        self.feed_rate=float(feed_rate)
        self.basal_cost=float(basal_cost)
        self.motor_cost=float(motor_cost)
        if balls:
            self.enable_balls()

    def enable_balls(self)->None:
        if self.balls:
            return
        self.balls=[
            Ball(.46,.66,.055,-.02),
            Ball(.77,.71,-.045,-.01),
            Ball(1.07,.63,-.025,.015),
        ]

    @staticmethod
    def _pixel(x:float,y:float)->int:
        ix=max(0,min(GRID_W-1,int(round((x-X_MIN)/(X_MAX-X_MIN)*(GRID_W-1)))))
        iy=max(0,min(GRID_H-1,int(round((y-Y_MIN)/(Y_MAX-Y_MIN)*(GRID_H-1)))))
        return iy*GRID_W+ix

    def geometry(self)->Tuple[Tuple[float,float],...]:
        elbow_a,wrist_a,index_a,middle_a,thumb_o,thumb_f=self.angle

        def rot(t:float,x:float,y:float=0.0):
            c=math.cos(t)
            s=math.sin(t)
            return c*x-s*y,s*x+c*y

        def add(a,b):
            return a[0]+b[0],a[1]+b[1]

        elbow=(.55,0.0)
        wrist=add(elbow,rot(elbow_a,.42))
        o=elbow_a+wrist_a
        palm=add(wrist,rot(o,.06))
        ib=add(wrist,rot(o,.12,.025))
        mb=add(wrist,rot(o,.12,0))
        tb=add(wrist,rot(o,.036,-.025))
        return (
            palm,
            add(ib,rot(o-.95*index_a,.10)),
            add(mb,rot(o-.95*middle_a,.11)),
            add(tb,rot(o-.90+thumb_o-.55*thumb_f,.08)),
        )

    def _joint_step(self,j:int,pos:float,neg:float)->None:
        a=self.angle[j]
        v=self.vel[j]
        lo,hi=RANGES[j]
        authority=(.06+.94*self.energy) if self.energy_enabled else 1.0
        accel=2.1*authority*(float(pos)-float(neg))-1.9*v-.35*a
        v+=.10*accel
        a+=.10*v
        if a<lo:
            a=lo
            v=0.0
        elif a>hi:
            a=hi
            v=0.0
        self.angle[j]=a
        self.vel[j]=v

    def _step_balls(self,points,prev_points)->int:
        if not self.balls:
            return 0
        dt=.10
        gravity=-.32
        radius=.024
        hand_radius=.025
        contacts=0
        if prev_points is None:
            point_vel=[(0.0,0.0)]*len(points)
        else:
            point_vel=[
                ((p[0]-q[0])/dt,(p[1]-q[1])/dt)
                for p,q in zip(points,prev_points)
            ]

        for ball in self.balls:
            ball.vy+=gravity*dt
            ball.x+=ball.vx*dt
            ball.y+=ball.vy*dt

            if ball.x<X_MIN+radius:
                ball.x=X_MIN+radius
                ball.vx=abs(ball.vx)*.94
            elif ball.x>X_MAX-radius:
                ball.x=X_MAX-radius
                ball.vx=-abs(ball.vx)*.94

            if ball.y<Y_MIN+radius:
                ball.y=Y_MIN+radius
                ball.vy=abs(ball.vy)*.92
            elif ball.y>Y_MAX-radius:
                ball.y=Y_MAX-radius
                ball.vy=-abs(ball.vy)*.92

            for (px,py),(hvx,hvy) in zip(points,point_vel):
                dx=ball.x-px
                dy=ball.y-py
                d2=dx*dx+dy*dy
                limit=radius+hand_radius
                if d2<=limit*limit:
                    contacts+=1
                    d=math.sqrt(max(d2,1e-12))
                    nx=dx/d
                    ny=dy/d
                    ball.x=px+nx*limit
                    ball.y=py+ny*limit
                    vn=(ball.vx-hvx)*nx+(ball.vy-hvy)*ny
                    if vn<0.0:
                        ball.vx-=1.88*vn*nx
                        ball.vy-=1.88*vn*ny
                    ball.vx+=.18*hvx
                    ball.vy+=.18*hvy

        return contacts

    def step(self,motor_currents:Mapping[int,float])->Obs:
        currents=tuple(
            max(0.0,min(1.0,float(motor_currents.get(m,0.0))))
            for m in range(MOTOR_COUNT)
        )

        for j in range(JOINT_COUNT):
            self._joint_step(j,currents[2*j],currents[2*j+1])

        points=self.geometry()
        contacts=self._step_balls(points,self.prev_points)
        self.prev_points=points

        hand=frozenset(self._pixel(x,y) for x,y in points)
        ball=frozenset(self._pixel(b.x,b.y) for b in self.balls)
        inputs=set(hand)|set(ball)

        for j,v in enumerate(self.vel):
            if v>.020:
                inputs.add(DYNAMIC_OFFSET+2*j)
            elif v<-.020:
                inputs.add(DYNAMIC_OFFSET+2*j+1)

        source_contact=False
        if self.source:
            source_pixel=self._pixel(self.SOURCE_X,self.SOURCE_Y)
            inputs.add(source_pixel)
            r2=self.SOURCE_RADIUS*self.SOURCE_RADIUS
            source_contact=any(
                (x-self.SOURCE_X)**2+(y-self.SOURCE_Y)**2<=r2
                for x,y in points
            )

        if self.energy_enabled:
            self.energy-=self.basal_cost+self.motor_cost*sum(currents)
            if self.source and source_contact:
                self.energy+=self.feed_rate
            self.energy=max(0.0,min(1.0,self.energy))

        return Obs(
            motor_currents=currents,
            inputs=frozenset(inputs),
            hand_pixels=hand,
            ball_pixels=ball,
            contacts=contacts,
            energy=self.energy,
            source_contact=source_contact,
        )

def grounded_activation(obs:Obs)->Dict[int,float]:
    """Physical transducer/actuator current expressed directly as Nethra activation."""
    out={
        m:float(current)
        for m,current in enumerate(obs.motor_currents)
        if float(current)>0.0
    }
    for sensor in obs.inputs:
        out[MOTOR_COUNT+int(sensor)]=1.0
    return out

def sensory_current(obs:Obs)->Dict[int,float]:
    """Physical sensory transducer current only; output Nethra are driven by the field/babbling."""
    return {
        MOTOR_COUNT+int(sensor):1.0
        for sensor in obs.inputs
    }
