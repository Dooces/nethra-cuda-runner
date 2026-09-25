"""Conduction rule variants (PROTOTYPE, 2026-09-25 comparison, docs/HANDOFF.md section 0f).
COND = top | all | topleaves (= core default "top_and_leaves") | leaves | beforeall | afterall |
leavesbefore | leavesafter.  Requires conduction="top" as the base rule.
Installs the rule on nethra.NethraField (the class the user scripts use)."""
import os
def install(nethra):
    C=nethra.NethraField; base=C._covered_members
    cond=os.environ.get("COND","top")
    def leaves(self, nethra_, route):
        cons=[m for m in route if m.routes]
        if len(cons)==len(route):          # no primitive member: keep top rule
            return base(self, nethra_, route)
        return cons
    def topleaves(self, nethra_, route):
        return [m for m in base(self, nethra_, route) if m.routes]
    def beforeall(self, nethra_, route):
        return [] if getattr(self,"_registering_before",False) else base(self, nethra_, route)
    def afterall(self, nethra_, route):
        return base(self, nethra_, route) if getattr(self,"_registering_before",False) else []
    def leavesbefore(self, nethra_, route):
        cov=base(self, nethra_, route)
        return [m for m in cov if m.routes] if getattr(self,"_registering_before",False) else cov
    def leavesafter(self, nethra_, route):
        cov=base(self, nethra_, route)
        return cov if getattr(self,"_registering_before",False) else [m for m in cov if m.routes]
    rule={"leavesbefore":leavesbefore,"leavesafter":leavesafter,"top":base,"all":base,"leaves":leaves,"topleaves":topleaves,"beforeall":beforeall,"afterall":afterall}[cond]
    C._covered_members=rule
    init=C.__init__
    def patched(self,*a,**k):
        k.setdefault("conduction", "all" if cond=="all" else "top"); k.setdefault("leakage", float(os.environ.get("LEAK","1")))
        init(self,*a,**k)
    C.__init__=patched
