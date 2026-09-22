from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable

from world import HandBallWorld, Obs


@dataclass(frozen=True)
class EnergyObs:
    motors: FrozenSet[int]
    inputs: FrozenSet[int]
    hand_pixels: FrozenSet[int]
    ball_pixels: FrozenSet[int]
    contacts: int
    energy: float
    source_contact: bool
    source_pixel: int


class EnergyHandBallWorld(HandBallWorld):
    """Same hand/ball physics with one visible physical energy source.

    Energy is never emitted as a learner input. Contact with the source changes only body physics:
    it restores energy; lower energy reduces motor authority on subsequent intervals.
    """

    SOURCE_X = .94
    SOURCE_Y = .24
    SOURCE_RADIUS = .040

    def __init__(
        self,
        *,
        balls: bool = True,
        seed: int = 0,
        initial_energy: float = .55,
        basal_cost: float = .00042,
        motor_cost: float = .000055,
        feed_rate: float = .0045,
    ):
        super().__init__(balls=balls, seed=seed)
        self.energy = float(initial_energy)
        self.basal_cost = float(basal_cost)
        self.motor_cost = float(motor_cost)
        self.feed_rate = float(feed_rate)
        self.source_pixel = self._pixel(self.SOURCE_X, self.SOURCE_Y)

    def _joint_step(self, j: int, pos: bool, neg: bool) -> None:
        a = self.angle[j]
        v = self.vel[j]
        lo, hi = __import__("world").RANGES[j]
        authority = .06 + .94 * max(0.0, min(1.0, self.energy))
        accel = 2.1 * authority * (float(pos) - float(neg)) - 1.9 * v - .35 * a
        v += .10 * accel
        a += .10 * v
        if a < lo:
            a = lo
            v = 0.0
        elif a > hi:
            a = hi
            v = 0.0
        self.angle[j] = a
        self.vel[j] = v

    def step(self, motors: Iterable[int]) -> EnergyObs:
        motors = frozenset(map(int, motors))
        base: Obs = super().step(motors)
        points = self.geometry()
        r2 = self.SOURCE_RADIUS * self.SOURCE_RADIUS
        contact = any((x - self.SOURCE_X) ** 2 + (y - self.SOURCE_Y) ** 2 <= r2 for x, y in points)

        self.energy -= self.basal_cost + self.motor_cost * len(motors)
        if contact:
            self.energy += self.feed_rate
        self.energy = max(0.0, min(1.0, self.energy))

        inputs = set(base.inputs)
        inputs.add(self.source_pixel)
        return EnergyObs(
            base.motors,
            frozenset(inputs),
            base.hand_pixels,
            base.ball_pixels,
            base.contacts,
            self.energy,
            contact,
            self.source_pixel,
        )
