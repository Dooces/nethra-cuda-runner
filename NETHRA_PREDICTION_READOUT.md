# Passive one-step prediction readout

Status: **available diagnostic/readout workflow**.

This note documents the prediction setup established by the passive top-k audits. It does not add
a predictor to Nethra and it does not change core learning semantics.

## What prediction means

Nethra is asked for one prospective state at a requested future TIME from everything genuinely
known at the present.

The live model is never advanced with its own prediction.

For one prediction:

1. Preserve the complete live Nethra state at the present.
2. Clone that state into a disposable shadow.
3. Supply only information genuinely known for the requested horizon, normally TIME and any other
   known exogenous inputs.
4. Evolve the shadow to that horizon.
5. Read the externally bound input Nethra from the prospective field and rank them.
6. Record top-k and confidence/separation.
7. Discard the shadow.
8. Reveal the real next observation to the untouched live Nethra and continue ordinary
   plasticity/construction.

The future observation being scored is never supplied to the shadow.

## Information leakage boundary

Allowed before prediction:
- every real observation already seen;
- all persistent Nethra and evidence learned from those observations;
- the complete current decaying field state;
- elapsed TIME to the requested horizon;
- other future inputs that are genuinely already known.

Forbidden before scoring:
- the actual future target;
- unknown intermediate future observations;
- any feature computed from those future observations.

The auditor may compare the completed prediction against reality afterward.

## Current native readouts

For an externally bound input Nethra i, useful passive diagnostics include:

    immediate tendency:
        d a_i / dt

    finite-horizon support above passive leakage:
        r_i(T) = a_i^shadow(T) - a_i^live(0) exp(-lambda T / C)

The finite-horizon residual is currently the primary top-k readout in the passive prediction
audits. It removes the trivial advantage caused by different starting activation while leaving the
prospective field evolution itself untouched.

For candidates sorted by residual score:

    top-k = highest k candidate scores

    margin = score(top1) - score(top2)

A normalized separation may also be reported:

    confidence = |top1 - top2| / (|top1| + |top2| + epsilon)

This is a diagnostic separation in [0,1], **not a calibrated probability**.

## Multi-horizon prediction

A prediction at a later horizon is still one prediction from the current real state:

    live state now
      -> shadow
      -> advance known TIME to requested horizon
      -> read field once

Do not recursively feed a predicted intermediate observation back as though it were real.

If an imagined trajectory is desired, use a separate disposable/shadow simulation and preserve
provenance so hypothetical inputs cannot become observed evidence.

## Established behavior

The passive auditor has already demonstrated:
- deterministic 8-symbol sequence: 16/16 one-step top-1 predictions;
- deterministic ordered contextual sequences: context-specific continuations remain readable;
- top-k exposes competing expectations without forcing a false single answer;
- top1-top2 separation is available as a native confidence-like diagnostic;
- prediction shadows do not feed back into learning.

The readout is an auditor/view of the existing field. Nethra does not need a second internal
classifier that says "I predicted X".

## AAPL use

For AAPL, the externally grounded price-direction Nethra are P+ and P-. Price-move magnitude remains
continuous in external current; it is not binned into symbolic magnitude classes. TIME is driven for
the actual elapsed wall-clock gap.

A passive AAPL prediction therefore ranks P+ and P- before the next adjusted OPEN/CLOSE price move is
revealed. Top-k has two candidates, and the separation between them is the confidence-like quantity.
The same prospective field may also be translated back through the fixed price-current transduction
as a diagnostic expected-return/price proxy.

Training/establishment must be kept separate from prediction auditing when testing whether a mature
Nethra has learned the stream: establish first with ordinary learning/construction only, then turn
on the disposable shadow auditor for unseen chronological observations.
