class Nethra:
    def __init__(self, relation=None):
        self.relation = dict(relation or {})
        self.activation = 0.0
        self.external = 0.0

    def push(self, amount):
        self.external += float(amount)

    def read(self):
        return self.activation


def advance(nodes):
    incoming = {n: n.external for n in nodes}

    for relation in nodes:
        if not relation.relation:
            continue

        incoming[relation] += sum(
            member.activation for member in relation.relation
        )

        for member in relation.relation:
            incoming[member] += relation.activation

    delta = {}

    for node in nodes:
        next_activation = incoming[node]
        delta[node] = next_activation - node.activation
        node.activation = next_activation
        node.external = 0.0

    return delta


def learn(delta, evidence, nodes):
    pattern = {
        node: change
        for node, change in delta.items()
        if change != 0.0
    }

    if not pattern:
        return None

    for relation in nodes:
        if relation.relation == pattern:
            return relation

    for seen in evidence:
        if seen["pattern"] == pattern:
            seen["count"] += 1

            if seen["count"] == 2:
                relation = Nethra(pattern)
                nodes.append(relation)
                evidence.remove(seen)
                return relation

            return None

    evidence.append({"pattern": pattern, "count": 1})
    return None
