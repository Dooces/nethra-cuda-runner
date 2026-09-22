from math import exp


class Nethra:
    """The only persistent participant."""
    def __init__(self, members=()):
        self.members = frozenset(members)
        self.evidence = 0
        self.activation = 0.0
        self.external = 0.0

    def push(self, amount):
        self.external += float(amount)

    def read(self):
        return self.activation


class Field:
    """Nethra field plus recurrence-driven Nethra construction."""
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

    def observe(self, delta):
        """Repeated joint Nethra change may crystallize as another Nethra."""
        members = frozenset(n for n, change in delta.items() if change != 0.0)
        if len(members) < 2:
            return None

        relation = self.by_members.get(members)
        if relation is not None:
            relation.evidence += 1
            return relation

        count = self.pending.get(members, 0) + 1
        if count < self.recurrence:
            self.pending[members] = count
            return None

        relation = Nethra(members)
        relation.evidence = count
        self.nethra.append(relation)
        self.by_members[members] = relation
        self.pending.pop(members, None)
        return relation

    def step(self, dt=.1):
        """C da/dt = J - leakage*a + symmetric conductive incidence current."""
        current = {
            n: n.external - self.leakage * n.activation
            for n in self.nethra
        }

        for relation in self.nethra:
            if not relation.members:
                continue
            g = self.conductance(relation)
            for member in relation.members:
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
