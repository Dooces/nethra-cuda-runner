# Blocking: how Nethra deals with it

Read this before touching blocking, the residual, evidence attribution or construction order.
Every item below has already been settled with receipts. Do not re-derive it.

## 1. The approach (settled)

Blocking is not a module or a rule of its own. It falls out of one ordering that the core already enforces:

1. **Complete / refind existing structure first.**
2. **Subtract what existing structure already accounts for.**
3. **Only the unresolved remainder earns new evidence or new construction.**

Kamin blocking is the case where step 2 removes the consequence: A already accounts for A -> O, so when X is
added (AX -> O), X should gain no durable consequence evidence, because nothing is left over for it.

Established, not open: NETHRA_TODO.md "Subtraction-before-construction — ESTABLISHED ORDERING" (branch
`nethra-field-tension-investigation`, commit 4df45a8, Fedora regression run 35795735967). Said there:
**DO NOT reopen whether subtraction occurs before construction.** The spec test from the same record, item 4:
*add an irrelevant X to an already-accounted recurrence and verify X does not acquire durable consequence
evidence.*

## 2. Where it lives in the current core (`nethra/nethra.py`)

- **Evidence side**, `_move_evidence_and_construct`: evidence moves by the residual epsilon = M - P
  (M = C [a_actual - a_zero_source]+, P = prior flow toward the Nethra). A consequence already carried by
  existing structure leaves a smaller residual, so less tension reaches the relations X feeds.
  X's incoming evidence is its flow share of that tension:
  `delta e_Xr = incoming_evidence_per_tension * (q_Xr / sum_j q_jr) * T_r`, with
  `T_r = sum_m p_rm (M_m - P_m)`.
- **Structure side**, `_construct` -> `_admit_whole_support` / `_admit_by_parts` -> `_accounted(before, after)`:
  existing Nethra whose routes match both sides account for the transition, so no new Nethra is minted
  for it. Recurring transitions go by whole support; the first occurrence goes by parts
  (`join_on_recurrence`).
- **Conduction**: `conduction="top_and_leaves"` (default) keeps blocking. Top-only loses it (0.89), and so
  does `d_in=0` under split (0.98). HANDOFF_LOG §0f.

## 3. Words that caused confusion (use these meanings)

- **Evidence attribution is not the field law.** The field law is the current-flow equation. Correcting how
  evidence is attributed is an accounting correction, not a "field-law change".
- **Expectation is live activation**, not the P flow statistic. A ratio P/M is the shortfall of that
  statistic, not a physical shortfall.
- **Accounting invariant:** existing support accounts for its share of the confirmed manifestation before the
  remainder justifies more evidence or construction. Expected and observed must describe the same Nethra, the
  same magnitude coordinate and the same temporal span; each contribution is counted once.
- `_accounted` certifies that stored structure matches both sides. It returns *which* Nethra, not *how much*
  of M they explain.

## 4. Do not do these (each was tried or proposed and dropped)

- Reopen the construction order.
- Raise `g_max` or change `tau` to get blocking: g_max 3 or 6 gives 0.82-1.06 (everything learns faster, X too).
- Add a scale, cap or gate on g, P or M in the evidence rule.
- Call a P/M ratio a physical shortfall.
- Build a blocking test with the cue present in the outcome interval: X then co-occurs with O itself and it
  is no longer the Kamin design (ratio 1.03 at 30 trials).
- Compare a blocked cue and a control cue inside one stream through different outcome Nethra; use the
  `human.py` design (separate fields, same outcome, control pretrained on another cue).

## 5. Receipts (default core, `a41200d`)

| test | blocked / control |
|---|---|
| `human.py 14`, 60 pretrain + 30 compound, live read | 0.18 |
| same, P read | 0.09 |
| NETHRA_OPERATING_NOTES §7 (baselines comparison) | 0.40 |
| `human.py`, admission seed 5 | 1.05 |
| 60 pretrain + 120 compound, live / P | 0.90 / 1.02 |
| 60 pretrain + 400 compound, live / P | 0.91 / 1.03 |

Longer compound exposure (2026-09-26, `nethra/experiments/blocking/scan.py`, `prospect.py`, `bl.py`):
- X's incoming evidence stays at seed (14) through ~30 compound trials and then grows: 21.5 at 60, 1137 at 120.
  `_accounted` returns the pretrained A -> O structure on the same steps.
- The complete zero-source state at O is 0.2 of the observation's own response at the interval end and 0.4 over
  the interval integral, with A fully trained. So redistributing the measured prospective state cannot zero the
  residual.
- This corresponds to the record's remaining work (NETHRA_TODO): the temporary candidate that carries only the
  remainder after subtraction. It is not a reason to reopen items 1-4.

## 6. How to measure blocking

`human.py` design: two fresh fields. Blocked: pretrain A -> O. Control: pretrain K -> O. Then AX -> O in
both, with O in its own interval after the cue and a filler between trials. Read O after X alone (3 quiet
intervals, push X). Report blocked / control, live and P. Report the compound-trial count; the ratio depends on it.
