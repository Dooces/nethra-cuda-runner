import unittest

from nethra import Nethra, NethraSpace


class CleanSlateContractTests(unittest.TestCase):
    def test_every_registered_entity_is_the_same_nethra_type(self):
        space = NethraSpace()
        nodes = [space.create() for _ in range(7)]
        self.assertTrue(all(type(node) is Nethra for node in nodes))

    def test_observation_requires_time_extent(self):
        space = NethraSpace()
        node = space.create()
        with self.assertRaises(ValueError):
            space.interval_delta(start=1.0, end=1.0, delta_by_nethra={node.nid: 1.0})

    def test_interval_cardinality_has_no_special_case(self):
        space = NethraSpace()
        nodes = [space.create() for _ in range(7)]
        one = space.interval_delta(
            start=0.0,
            end=1.0,
            delta_by_nethra={nodes[0].nid: 0.25},
        )
        many = space.interval_delta(
            start=1.0,
            end=2.0,
            delta_by_nethra={node.nid: float(i + 1) for i, node in enumerate(nodes)},
        )
        self.assertEqual(len(one["delta"]), 1)
        self.assertEqual(len(many["delta"]), 7)

    def test_zero_deltas_are_execution_sparsity_only(self):
        space = NethraSpace()
        a = space.create()
        b = space.create()
        record = space.interval_delta(
            start=0.0,
            end=2.0,
            delta_by_nethra={a.nid: 0.0, b.nid: -0.5},
        )
        self.assertEqual(dict(record["delta"]), {b.nid: -0.5})


if __name__ == "__main__":
    unittest.main()
