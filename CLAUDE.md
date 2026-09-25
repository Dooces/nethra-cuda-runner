# Instructions for working on Nethra

Read this whole file before doing anything. Then read `nethra/NETHRA_OPERATING_NOTES.md`.
Every rule here exists because it was broken before.

## 0. Before running anything: predict it from the code

1. **Predict first.** Write down, from `nethra.py`, what construction will build for the stream:
   - which routes;
   - how many Nethra;
   - what closure will refind;
   - when each is read (§3).

   Key facts:
   - `_admit_whole_support` joins the complete closure of consecutive intervals.
   - Every co-present Nethra enters the route. No subsets are enumerated.
   - Existing structure is reused before anything new is built.
2. **Run it tiny** (tens of intervals).
3. **Compare.** If the numbers differ from the prediction, stop and find out why in the code
   before running anything larger or on the runner.

Every mistake made so far would have been caught here. Example: three cycles always pushed
together can only build one Nethra per joint state (input Nethra + lcm of the periods). No cycle
gets structure of its own, because no cycle is ever present alone.

## 1. What Nethra is not

Nethra is not machine learning, not a neural network, not a model, not a learner, not a predictor.
Do not map it onto anything you know from ML. If a thought starts with "like a network / like
training / like attention / like a classifier", stop. It is wrong.

Never use these words about Nethra, in code, comments, docs or replies:

| Don't say | Say instead |
|---|---|
| input layer, output, hidden state, unit, neuron | Nethra (every one is the same type) |
| weight | incidence evidence, conductance `g` |
| training, epoch, dataset, sample | stream, intervals, exposure |
| prediction, predicted | prior flow `P`, live activation (the expectation) |
| target, label, error, loss | manifestation `M`, residual `M - P` |
| accuracy, score, argmax, winner | activation, share of activation, closure |
| triple, token, item, unit (for a group of intervals) | "Q then R then S in consecutive intervals" |
| learned representation, embedding | constructed Nethra, routes, closure |
| confidence calibration, surprise metric | closure refinding, activation (only what the field has) |
| capacity, hyperparameter tuning | parameter (see notes §4), only with a stated reason |

The field has no grouping and no units. A harness pushes numbers onto Nethra, one interval at a
time. Whether several intervals become one structure is decided by construction and shown by
closure. Never declare it in the harness.

## 2. Hard rules

1. **Do not change the field law.**
   - All earned incidences conduct.
   - No gate or multiplier on `g`.
   - No selector or winner.
   - No new participant type.
   - `g(0) = 0`.
   - Departure is not a negative push.
2. **Do not open a problem without an observed failure, a contradiction or an explicit request.**
   - A number looking low is not a failure.
   - Do not invent a fix, a metric or a mechanism.
3. **Do not tune parameters to move a number.**
   - `g_max`, `tau`, `outgoing_evidence_per_flow`, `incoming_evidence_per_tension`, `admission_seed` etc. change only with an explicit request.
4. **Do not rerun the user's test scripts** (`nethra/tests/*.py`) unless asked.
   - When unsure how something works, write your own small script in the scratchpad.
   - Measure the specific thing.
   - Report numbers.
5. **No standing test suites, regression harnesses or CI workflows** unless explicitly asked.
6. **Report measurements, not conclusions.**
   - Say what was pushed, what was read, and the numbers.
   - Don't turn an observation into a diagnosis and then act on it.
7. **Check claims in code before repeating them.** Including your own earlier claims and other
   models' claims.
8. **Comments and docs describe Nethra in its own terms** (table above). Changing wording in
   `nethra.py` must not change code: compare the AST without docstrings before committing.
9. **`nethra/NETHRA_MISTAKE_LEDGER.md` is an old copy, for reference only.** Do not append to it.

## 3. How to feed and read (short form; details in the notes)

- **Feed:** push a number onto each Nethra present in this interval, then `step(1.0)`.
  - One interval per observation.
  - Nothing else: no labels, no history window, no targets.
- **Co-present context:** push it in the same interval as what it accompanies.
- **Read:**
  - live activation right after an interval (that is the expectation);
  - closure: which constructed Nethra are refound.
    - `previous_closure` after `step` was computed **before** that interval's construction, so it
      misses Nethra built at that boundary (a t-1 read).
    - Read the completed interval under current topology instead:
      `f.closure(f.previous_explicit, f.current_source_event)`, as `_admit_whole_support` does.
  - topology (routes, incidence evidence).
- **Probe without disturbing the field:** copy it and turn evidence change and construction off on the copy.

      g = NethraField.from_checkpoint_dict(f.checkpoint_dict())
      g.topology_and_evidence_change = False

## 4. How structure builds

- **Small factors first.** Construction only joins what closure refinds.
  - Build the small factors first; larger structure then builds up from them.
  - Only then can later experience factor the small ones back out.
- **Design every stream so the small factors are built first.**
  - A stream that skips them is not a test of Nethra.
  - Its results are not properties of Nethra. Don't record them as facts.
- **Input goes in as the notes say** (notes §3; graded values through overlapping receptive
  Nethra). Results from input fed any other way are also not properties of Nethra.

## 5. Current work

- **Roadmap:** `docs/NETHRA_ROADMAP.md`.
- **Current item:** sub-pattern transfer (`nethra/tests/transfer.py`).
  - Its stream does not build the small factors first (see §4).
  - Fix the stream before reading anything from it.

## 6. Practical

- **Container:** 4 cores. Runs over about 10 minutes go in the background.
- **Killing processes:** don't use `pkill -f <pattern>`. The pattern also matches the shell
  running the kill, which kills your own command. Kill by PID.
- **Files:**
  - disposable scripts go in the scratchpad, not the repo;
  - repo layout: core, notes and old ledger in `nethra/`, the user's test scripts in
    `nethra/tests/`.
- **Replies:** short and plain. No filler.
