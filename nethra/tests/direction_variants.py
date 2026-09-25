"""PROTOTYPE, NOT PART OF THE CORE.  Variants of direction="split" evidence change (docs/HANDOFF.md 0h).
env VAR: "+"-joined flags saying which terms of the evidence change also move the OTHER direction:
  on / op : negative / positive outgoing term (relation -> member flow, then member's residual)
            also moves member -> relation evidence;
  in / ip : negative / positive incoming term (tension share) also moves relation -> member evidence;
  neg = on+in;  persist: members lying in more than one route of the relation get both terms on both;
  all four = the shared evidence (with separate stores).  UNI=1: member -> relation evidence from the
  relation's own manifestation minus all member inflow (collapses, 0h).
Usage: import direction_variants; direction_variants.install(nethra)  (before creating fields), or
  VAR=op DIR=split python3 with_direction_variant.py script.py args"""
import os
import numpy as np
from collections import Counter

def _move_evidence_split(self, source_current, manifestation, current_closed, current_description,
                         current_source_event, residual_neighbors, physical, compiled):
    """direction="split": the same evidence change, each term on its own direction.

    Prior flow along an incidence ran relation -> member when A_rel > A_mem (through g) and
    member -> relation otherwise (through g_in).  The outgoing term moves the relation -> member
    evidence, the incoming term the member -> relation evidence, with the same formulas."""
    A = self.current_interval_integral
    order = self._order
    rows = [(r, m, row) for (r, m), row in physical.items() if row["g"] > 0.0 or row["g_in"] > 0.0]
    R = len(rows)
    nodes = self.nethra
    size = len(nodes)
    rel_i = np.fromiter((order[r] for r, _m, _row in rows), dtype=np.intp, count=R)
    mem_i = np.fromiter((order[m] for _r, m, _row in rows), dtype=np.intp, count=R)
    g_out = np.fromiter((float(row["g"]) for _r, _m, row in rows), dtype=float, count=R)
    g_in = np.fromiter((float(row["g_in"]) for _r, _m, row in rows), dtype=float, count=R)
    a_rel = np.fromiter((float(A.get(r, 0.0)) for r, _m, _row in rows), dtype=float, count=R)
    a_mem = np.fromiter((float(A.get(m, 0.0)) for _r, m, _row in rows), dtype=float, count=R)
    d = a_rel - a_mem
    q = np.where(d > 0.0, g_out, g_in) * d
    out_pos = np.flatnonzero(q > 0.0)
    in_pos = np.flatnonzero(q < 0.0)
    prior_sum = np.bincount(mem_i[out_pos], weights=q[out_pos], minlength=size)
    _u, first = np.unique(mem_i[out_pos], return_index=True)
    flow_members = [nodes[i] for i in mem_i[out_pos][np.sort(first)].tolist()]
    prior_flow = {n: float(prior_sum[order[n]]) for n in flow_members}
    keys = list(manifestation)
    keys += [n for n in prior_flow if n not in manifestation]
    manifest_minus_prior = {
        n: float(manifestation.get(n, 0.0)) - float(prior_flow.get(n, 0.0)) for n in keys
    }
    self.update_residuals(manifest_minus_prior, neighbors=residual_neighbors, compiled=compiled)
    mmp_arr = np.zeros(size)
    for n, value in manifest_minus_prior.items():
        mmp_arr[order[n]] = value
    p_out = q[out_pos]
    mmp_out = mmp_arr[mem_i[out_pos]]
    tension = np.bincount(rel_i[out_pos], weights=p_out * mmp_out, minlength=size)
    q_in = -q[in_pos]
    total_in = np.bincount(rel_i[in_pos], weights=q_in, minlength=size)
    rel_in = rel_i[in_pos]
    d_out = self.outgoing_evidence_per_flow * p_out * mmp_out
    d_in = self.incoming_evidence_per_tension * tension[rel_in] * (q_in / total_in[rel_in])
    if os.environ.get("UNI", "") == "1":
        # one rule for every directed incidence: flow into a Nethra, then that Nethra's own
        # manifestation minus everything that flowed into it (members -> relation flows here)
        p_in_sum = np.bincount(rel_in, weights=q_in, minlength=size)
        man = np.zeros(size)
        for n, value in manifestation.items():
            man[order[n]] = value
        d_in = self.outgoing_evidence_per_flow * q_in * (man[rel_in] - p_in_sum[rel_in])
    touched = {}
    VAR = os.environ.get("VAR", "")
    jobs = [(out_pos, d_out, "receipts", self.incidence_evidence),
            (in_pos, d_in, "receipts_in", self.incidence_evidence_in)]
    flags = set(VAR.split("+")) if VAR else set()
    if VAR == "neg":
        flags = {"on", "in"}
    # on/op: negative/positive outgoing term also moves member -> relation evidence;
    # in/ip: negative/positive incoming term also moves relation -> member evidence
    if "on" in flags:
        jobs.append((out_pos[d_out < 0], d_out[d_out < 0], "receipts_in", self.incidence_evidence_in))
    if "op" in flags:
        jobs.append((out_pos[d_out > 0], d_out[d_out > 0], "receipts_in", self.incidence_evidence_in))
    if "in" in flags:
        jobs.append((in_pos[d_in < 0], d_in[d_in < 0], "receipts", self.incidence_evidence))
    if "ip" in flags:
        jobs.append((in_pos[d_in > 0], d_in[d_in > 0], "receipts", self.incidence_evidence))
    if "persist" in flags:
        # a member lying in more than one route of the relation (present on both sides) has
        # no order relative to it: both terms move both directions of its incidence
        both = np.fromiter((sum(1 for r in rel.routes if mem in r) > 1 for rel, mem, _ in rows),
                           dtype=bool, count=R)
        po = out_pos[both[out_pos]]; pi = in_pos[both[in_pos]]
        jobs.append((po, d_out[both[out_pos]], "receipts_in", self.incidence_evidence_in))
        jobs.append((pi, d_in[both[in_pos]], "receipts", self.incidence_evidence))
    for positions, deltas, rk, store in jobs:
        for k, delta in zip(positions.tolist(), deltas.tolist()):
            relation, member, row = rows[k]
            receipts = row[rk]
            if not receipts:
                continue
            share = float(delta) / len(receipts)
            for route, signature in receipts:
                bucket = store.get((relation, route, member))
                if bucket is None:
                    bucket = store[(relation, route, member)] = Counter()
                bucket[signature] = max(0.0, float(bucket.get(signature, 0.0)) + share)
                touched[(relation, route, signature)] = None
    for relation, route, signature in touched:
        self._route_summary(relation, route, signature)
    self._construct(source_current, manifest_minus_prior, current_closed, current_description,
                    current_source_event)
    return manifest_minus_prior



def install(core):
    core.NethraField._move_evidence_split = _move_evidence_split
