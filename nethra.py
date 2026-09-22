from math import exp, sqrt


class Nethra:
    def __init__(self, incidence=None):
        self.incidence = dict(incidence or {})
        self.evidence = 0
        self.activation = 0.0
        self.external = 0.0

    @property
    def members(self):
        return frozenset(self.incidence)

    def push(self, amount):
        self.external += float(amount)

    def read(self):
        return self.activation


class Field:
    def __init__(self, *, g_min=.20, g_max=1.50, tau=100.0,
                 capacitance=1.0, leakage=1.0, recurrence=2):
        self.g_min = float(g_min)
        self.g_max = float(g_max)
        self.tau = float(tau)
        self.capacitance = float(capacitance)
        self.leakage = float(leakage)
        self.recurrence = int(recurrence)
        self.nethra = []
        self.pending = {}
        self.by_members = {}

    def add(self):
        n = Nethra()
        self.nethra.append(n)
        return n

    def conductance(self, relation):
        e = max(0.0, float(relation.evidence))
        return self.g_min + (self.g_max - self.g_min) * (1.0 - exp(-e / self.tau))

    @staticmethod
    def _unit(values):
        norm = sqrt(sum(x*x for x in values.values()))
        return {n: x / norm for n, x in values.items()}

    @staticmethod
    def _alignment(a, b):
        return sum(a[n] * b[n] for n in a)

    @staticmethod
    def _same_axis(a, b):
        return abs(abs(Field._alignment(a, b)) - 1.0) <= 1e-12

    @staticmethod
    def _passive_shape(direction):
        signs = {1 if x > 0.0 else -1 for x in direction.values() if x != 0.0}
        if len(signs) != 1:
            return None
        peak = max(abs(x) for x in direction.values())
        return {n: abs(x) / peak for n, x in direction.items()}

    @staticmethod
    def _relation_axis(relation):
        return Field._unit({n: x for n, x in relation.incidence.items()})

    def consider(self, residual):
        """Consider unexplained interval delta after existing Nethra have been subtracted."""
        values = {n: float(change) for n, change in residual.items() if change != 0.0}
        members = frozenset(values)
        if len(members) < 2:
            return None

        direction = self._unit(values)
        passive_shape = self._passive_shape(direction)

        if passive_shape is not None:
            target_axis = self._unit(passive_shape)
            for relation in self.by_members.get(members, ()):
                if self._same_axis(self._relation_axis(relation), target_axis):
                    relation.evidence += 1
                    return relation

        bucket = self.pending.setdefault(members, [])
        for candidate in bucket:
            if self._same_axis(candidate["direction"], direction):
                candidate["count"] += 1
                if candidate["count"] < self.recurrence or passive_shape is None:
                    return None
                relation = Nethra(passive_shape)
                relation.evidence = candidate["count"]
                self.nethra.append(relation)
                self.by_members.setdefault(members, []).append(relation)
                bucket.remove(candidate)
                return relation

        bucket.append({"direction": direction, "count": 1})
        return None

    def step(self, dt=.1):
        current = {n: n.external - self.leakage * n.activation for n in self.nethra}
        for relation in self.nethra:
            if not relation.incidence:
                continue
            base = self.conductance(relation)
            for member, relative in relation.incidence.items():
                g = base * relative
                flow = g * (relation.activation - member.activation)
                current[relation] -= flow
                current[member] += flow

        delta = {}
        scale = float(dt) / self.capacitance
        for n in self.nethra:
            old = n.activation
            n.activation += scale * current[n]
            delta[n] = n.activation - old
            n.external = 0.0
        return delta
