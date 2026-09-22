# Nethra clean slate

This branch discards the previous experimental implementation and keeps only the smallest established invariants.

## Stage zero

1. Every entity that can participate in the learned system is a Nethra.
2. External input and output eventually attach to Nethra; they do not introduce another graph-node ontology.
3. Observation exists as change over an explicit non-zero interval.
4. An interval may contain any number of changing Nethra.
5. No pair, tuple, fixed arity, binary construction tree, source/consequence object, candidate object, reward, selector, semantic label, or task-specific learner exists here.
6. Relation construction is intentionally absent until its rule is derived and tested without choosing a representation in advance.
7. Field dynamics are intentionally absent until the relation representation is settled.
8. Persistence, decay, consequence, temporal extension, attention, action and feeding are intentionally absent.

The previous branches remain available as experimental evidence. Nothing from them is imported into this branch.

The only implemented operation at this stage is registration of Nethra identities and validation of an interval delta. The interval record is returned to the caller and is not retained as a dense timeline.
