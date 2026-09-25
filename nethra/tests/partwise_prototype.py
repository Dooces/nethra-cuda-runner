"""PROTOTYPE, NOT PART OF THE CORE (subclasses of NethraField used by binocular_multi.py and
context_partwise.py).  Scratch prototype (core untouched): structural subtraction per part before construction.
Handles = constructed Nethra with a route contained in the before-side closure and a route contained in
the after-side closure.  Source Nethra inside those routes are accounted.  Construction runs only on the
unaccounted source Nethra of each side (before = previous explicit, after = current explicit)."""
import nethra_presence as core

class PartwiseField(core.NethraField):
    subtraction_log = None
    def _admit_whole_support(self, current_closed, current_description, unresolved, current_source_event):
        if unresolved <= self.admission_threshold or not self.previous_closure:
            return ()
        before_side = frozenset(self.closure(self.previous_explicit, self.current_source_event))
        after_side = frozenset(current_closed)
        prev_src = self._event_members(self.current_source_event)
        cur_src = self._event_members(current_source_event)
        touched = set()
        for m in prev_src:
            for rel, _r in self.member_to_routeuses.get(m, ()): touched.add(rel)
        acc_prev, acc_cur, handles = set(), set(), []
        for h in self._ordered(touched):
            rb = [R for R in h.routes if R <= before_side]
            ra = [R for R in h.routes if R <= after_side]
            if not rb or not ra: continue
            cb = {m for R in rb for m in R if m in prev_src}
            ca = {m for R in ra for m in R if m in cur_src}
            if not cb or not ca: continue
            handles.append(h); acc_prev |= cb; acc_cur |= ca
        rest_prev = prev_src - acc_prev
        rest_cur = cur_src - acc_cur
        if self.subtraction_log is not None:
            self.subtraction_log.append((len(handles), len(rest_prev), len(rest_cur)))
        if not rest_prev and not rest_cur:
            return tuple(handles)                      # everything accounted: nothing built
        if not handles:
            return super()._admit_whole_support(current_closed, current_description, unresolved, current_source_event)
        if not rest_prev or not rest_cur:
            return tuple(handles)                      # remainder only on one side: nothing joins it across the interval
        # remainder only: re-describe each side from its unaccounted source Nethra, then the ordinary procedure
        sub_prev = frozenset((n, v) for n, v in self.current_source_event if n in rest_prev)
        sub_cur = frozenset((n, v) for n, v in current_source_event if n in rest_cur)
        saved = (self.previous_explicit, self.current_source_event, self.current_event)
        try:
            self.previous_explicit = rest_prev; self.current_source_event = sub_prev; self.current_event = frozenset()
            closed = self.closure(rest_cur, sub_cur)
            return super()._admit_whole_support(closed, frozenset(), unresolved, sub_cur)
        finally:
            self.previous_explicit, self.current_source_event, self.current_event = saved


class ExactSourcePartwiseField(PartwiseField):
    """V1: the structural source event is the pushed pattern itself (no substitution by a stored
    pattern).  Graded values stay in the event; nothing is stored for cosine matching."""
    def _canonical_source_event(self, source_current):
        return frozenset((n, float(v)) for n, v in source_current.items() if float(v) != 0.0)
