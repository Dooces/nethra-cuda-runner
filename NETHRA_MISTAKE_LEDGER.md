# Nethra mistake ledger

This file is cumulative. Never rewrite history to make prior work look cleaner.

## Mandatory maintenance rule

Whenever an assistant or developer reads `nethra.py` for interpretation, debugging, modification,
benchmarking, or explanation, they must also read this ledger first. Before making a semantic change,
append any newly discovered mistake, regression, contamination, false claim, or shortcut introduced
since the last ledger entry. Do not repeat a listed mistake under a new name.

## 2026-09-23 — stale branch promoted over the frozen interval boundary

- Native-plasticity investigation work branched from `34e2cf5c...` before the later commits that
  detached `_consider_completed_interval_provisional()` from live `step()`.
- The divergent investigation lineage was later promoted to branches named
  `nethra-current-baseline` / `nethra-closure-fixed-freeze`, despite still calling the retired
  provisional probability/counting learner from live execution.
- This silently reversed an explicitly frozen decision.
- Prevention: semantic baseline changes must be traced to their actual ancestor, never inferred from
  a branch name.

## 2026-09-23 — benchmarks reattached retired learning machinery

- `StableDepthAuditField.step()` and related harnesses explicitly called
  `_consider_completed_interval_provisional()`, then results were described as evidence for native
  Nethra learning.
- Wide-library/depth results therefore measured the retired history learner plus the field, not the
  native construction mechanism.
- Prevention: a harness that replaces `step()`, `_edges()`, construction, or learning semantics
  is a probe of that harness, never evidence that the core learned autonomously.

## 2026-09-23 — provisional learner modified while accidentally live

- `bc81934...` changed subtraction logic inside
  `_consider_completed_interval_provisional()` while the dirty lineage still invoked it live.
- This changed actual learning despite the function being described as regression-only machinery.
- Prevention: retired/provisional semantic machinery must not remain callable from the native live
  path.

## 2026-09-23 — stale current-event bug in recursive AAPL learner

- The learner asked closure to interpret the current observation using `self.f.current_event` from
  the preceding interval.
- Very deep recursive results were reported before this was corrected.
- `bbc529c2...` later fixed refinding to construct the current source event first.
- Prevention: observation t may never be interpreted with event state from t-1.

## 2026-09-23 — conceptual state confused with one-file integration state

- Previously settled persistence, recursive bootstrap, subtraction-before-construction, permissive
  admission, continuous per-incidence plasticity, and g(0)=0 were repeatedly described as open
  because the stripped one-file core had not integrated them.
- This caused already-settled questions to be reopened instead of reading prior contracts/reports.
- Prevention: distinguish “not integrated in this file” from “not settled conceptually.”

## 2026-09-23 — temporary candidate requirement resurrected after permissive admission was settled

- After the user had accepted permissive weak admission plus continuous plasticity, later analysis
  incorrectly reinstated a separate temporary-candidate species/process as mandatory.
- That contradicted the accepted route: admit an ordinary weak Nethra from unresolved support, then
  let incidence evidence determine field relevance while retaining the object.
- Prevention: do not reintroduce a second semantic object when an ordinary Nethra already serves as
  the hypothesis.

## 2026-09-23 — whole-support recruitment repeatedly reopened

- The already-developed answer was whole unresolved active support after recursive refinding and
  subtraction, with no subset enumeration; incidence-local plasticity separates useful members from
  nuisance members.
- Later replies again treated initial support-route formation as an unanswered design question.
- Prevention: use the settled whole-support mechanism unless new evidence falsifies it.

## 2026-09-23 — tests used as reassurance despite failing to guard semantic provenance

- Behavioral regression suites repeatedly passed while the active branch used stale ancestry,
  reattached provisional learning, and harness-level semantic overrides.
- Passing those suites was presented as confidence that the architecture remained clean.
- Prevention: behavioral tests are not evidence of semantic provenance. Read ancestry, live call
  paths, and exact diffs directly.

## 2026-09-23 — improvised recovery integration substituted a new residual coordinate

- A recovery branch attempted to integrate plasticity using raw source current/charge as the generic
  consequence coordinate and reached only shallow recursive depth.
- That implementation was not the previously settled whole-support mechanism and was not promoted.
- Prevention: recover the prior agreed equations and support semantics before writing replacement
  learning code.

## 2026-09-23 — broad “bootstrap unresolved” claim contradicted earlier frozen work

- Earlier bootstrap audits had already shown higher relations forming only after prerequisite Nethra
  existed, including irreducible multi-layer cases with zero premature construction.
- Later summaries incorrectly called representational bootstrap generally unresolved.
- Prevention: before declaring a core mechanism open, search the frozen contracts and prior reports
  for an explicit pass/fail conclusion.

## 2026-09-23 — first native whole-support integration omitted structural reuse

- The first integration pass correctly subtracted existing prospective field prediction before
  admission, but admission only checked for an exact already-stored whole-support route.
- It did not also call the already-settled structural `_accounted(before, after)` check, so a weak
  existing relation that already described both sides could have been duplicated.
- Prevention: native admission must perform both kinds of subtraction before minting: field
  prediction residual and structural accounted/reuse.

## 2026-09-23 — re-derived and nearly re-broke already-fixed self-description handling

- The native whole-support integration reached an `_accounted()` reuse case where the reused
  relation could already be present in the recursive whole-support description.
- This had already been solved earlier: self-containing recursive descriptions are tautological and
  must be skipped; refound self-presence can subtract/account for structure but contributes no fresh
  support route into itself.
- The first integration pass nevertheless attempted to reason about adding that route again.
- Prevention: preserve the established `if relation in description: skip route registration`
  rule wherever recursive descriptions are promoted or reused.

## 2026-09-23 — reintroduced the already-rejected source-only consequence residual

- The first native whole-support integration set the consequence target from current external source
  charge alone. That repeats the exact defect isolated by `82c0b1eb...`: an internally manifested
  recursive Nethra can have zero external source and positive real manifestation, so source-only
  residual falsely punishes a correct prediction.
- The prior native replay correction used an identical zero-input counterfactual from the same
  pre-outcome state: `target = C * (actual - baseline)`, clamped only at zero for positive
  manifestation. This preserves original-source provenance separately while allowing internally
  manifested Nethra to count as consequences.
- Prevention: external source support is a provenance/construction coordinate, never a universal
  consequence coordinate.
