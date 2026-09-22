"""Shadow test for transient dyadic range descriptions over the frozen Nethra core.

Nothing in this file is a persistent Nethra or a semantic matcher.  A scalar observation emits one
purely numerical containing interval at every represented depth.  Statistics are accumulated for
those transient interval descriptions.  The held-out scoring below is evaluator-only: it tests
whether additional resolution actually carries prospective information; it is not available to the
learner and is not a proposed promotion rule.
"""

from collections import defaultdict
from math import log
import random


class DyadicEvidence:
    def __init__(self, lo=0.0, hi=1.0, max_depth=12):
        if not hi > lo:
            raise ValueError("hi must exceed lo")
        self.lo = float(lo)
        self.hi = float(hi)
        self.max_depth = int(max_depth)
        self.counts = [defaultdict(lambda: [0, 0]) for _ in range(self.max_depth + 1)]
        self.total = [0, 0]
        self.updates = 0

    def key(self, value, depth):
        """Arithmetic containing interval index; it makes no equality/meaning claim."""
        x = (float(value) - self.lo) / (self.hi - self.lo)
        if x < 0.0 or x > 1.0:
            raise ValueError("value outside externally supplied physical domain")
        cells = 1 << int(depth)
        index = min(int(x * cells), cells - 1)
        return index

    def describe(self, value):
        """Emit one nested transient range coordinate per depth, O(max_depth)."""
        return tuple((depth, self.key(value, depth)) for depth in range(self.max_depth + 1))

    def observe(self, value, consequence):
        """Accumulate consequence counts for every containing range; no range becomes persistent."""
        y = int(bool(consequence))
        self.total[y] += 1
        for depth, index in self.describe(value):
            cell = self.counts[depth][index]
            cell[y] += 1
            self.updates += 1

    def probability(self, value, depth):
        """Empirical P(consequence=1 | containing range), falling back only when unseen."""
        index = self.key(value, depth)
        zero, one = self.counts[depth].get(index, (0, 0))
        n = zero + one
        if n:
            return one / n
        total = self.total[0] + self.total[1]
        return self.total[1] / total if total else 0.5

    def incremental_information(self, depth):
        """Audit statistic: child-range consequence distribution relative to its parent range."""
        if depth <= 0:
            return 0.0
        total = self.total[0] + self.total[1]
        if not total:
            return 0.0
        score = 0.0
        for child, (c0, c1) in self.counts[depth].items():
            n = c0 + c1
            if not n:
                continue
            p = c1 / n
            p0, p1 = self.counts[depth - 1][child // 2]
            pn = p0 + p1
            q = p1 / pn if pn else 0.5
            score += (n / total) * bernoulli_kl(p, q)
        return score


def bernoulli_kl(p, q):
    eps = 1e-12
    p = min(max(float(p), eps), 1.0 - eps)
    q = min(max(float(q), eps), 1.0 - eps)
    return p * log(p / q) + (1.0 - p) * log((1.0 - p) / (1.0 - q))


def brier(model, rows, depth):
    """Outside-observer score only; the tested evidence layer never receives this result."""
    return sum((y - model.probability(x, depth)) ** 2 for x, y in rows) / len(rows)


def exact_float_brier(train, test):
    """Control showing why exact continuous-value recurrence is useless on fresh samples."""
    table = {}
    ones = 0
    for x, y in train:
        table.setdefault(x, [0, 0])
        table[x][y] += 1
        ones += y
    fallback = ones / len(train)
    err = 0.0
    repeats = 0
    for x, y in test:
        counts = table.get(x)
        if counts:
            repeats += 1
            p = counts[1] / sum(counts)
        else:
            p = fallback
        err += (y - p) ** 2
    return err / len(test), repeats


def make_rows(kind, n, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        x = rng.random()
        if kind == "irrelevant":
            p = 0.50
        elif kind == "narrow":
            p = 0.95 if 0.497 <= x < 0.503 else 0.05
        elif kind == "islands":
            p = 0.90 if (0.15 <= x < 0.35 or 0.65 <= x < 0.85) else 0.10
        elif kind == "smooth":
            p = 0.05 + 0.90 * x
        else:
            raise ValueError(kind)
        out.append((x, int(rng.random() < p)))
    return out


def run_case(kind, n=120_000, max_depth=12):
    train = make_rows(kind, n, 1000 + len(kind))
    test = make_rows(kind, n, 2000 + len(kind))
    model = DyadicEvidence(max_depth=max_depth)
    for x, y in train:
        model.observe(x, y)
    scores = [brier(model, test, d) for d in range(max_depth + 1)]
    info = [model.incremental_information(d) for d in range(max_depth + 1)]
    exact_score, exact_repeats = exact_float_brier(train, test)
    return model, scores, info, exact_score, exact_repeats


def main():
    results = {}
    for kind in ("irrelevant", "narrow", "islands", "smooth"):
        model, scores, info, exact_score, exact_repeats = run_case(kind)
        results[kind] = (model, scores, info, exact_score, exact_repeats)

    irrelevant = results["irrelevant"]
    narrow = results["narrow"]
    islands = results["islands"]
    smooth = results["smooth"]

    # Purely continuous exact keys do not recur between independent streams.
    assert irrelevant[4] == narrow[4] == islands[4] == smooth[4] == 0
    assert abs(irrelevant[3] - irrelevant[1][0]) < 1e-12

    # Irrelevant scalar variation gains no durable held-out predictive value from refinement.
    assert min(irrelevant[1][:9]) > irrelevant[1][0] - 0.001

    # A narrow consequential band is recoverable without ever comparing two float values.
    assert min(narrow[1][6:11]) < narrow[1][0] - 0.002

    # Greedy binary refinement is invalid: the first split cancels, deeper ranges still matter.
    assert abs(islands[1][1] - islands[1][0]) < 0.002
    assert islands[1][3] < islands[1][0] - 0.08
    assert islands[2][1] < 0.001
    assert islands[2][3] > 0.10

    # A smooth consequence relation finds useful finite resolution; exact matching again collapses.
    assert smooth[1][5] < smooth[1][0] - 0.05
    assert abs(smooth[3] - smooth[1][0]) < 1e-12

    # Updating all represented resolutions is linear in represented depth, not exponential in bins.
    model = irrelevant[0]
    assert model.updates == 120_000 * 13

    for kind, (_model, scores, info, exact_score, exact_repeats) in results.items():
        best_depth = min(range(len(scores)), key=scores.__getitem__)
        print(kind)
        print("  brier depth0   ", f"{scores[0]:.9f}")
        print("  brier depth1   ", f"{scores[1]:.9f}")
        print("  brier depth3   ", f"{scores[3]:.9f}")
        print("  best depth     ", best_depth, f"{scores[best_depth]:.9f}")
        print("  exact-float    ", f"{exact_score:.9f}", "heldout repeats", exact_repeats)
        print("  incr-info d1   ", f"{info[1]:.9f}")
        print("  incr-info d3   ", f"{info[3]:.9f}")
    print("updates_per_scalar", irrelevant[0].updates // 120_000)
    print("all_assertions_passed")


if __name__ == "__main__":
    main()
