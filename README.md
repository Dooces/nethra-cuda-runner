# Nethra

One field of ordinary Nethra. Every Nethra has an activation; every earned incidence conducts both ways with
conductance `g(e) = g_max (1 - exp(-e / tau))`; structure is built from what existing structure does not already
account for. The whole core is one file: `nethra/nethra.py`.

## Start here (in this order)

1. `CLAUDE.md` — rules for working on Nethra. Every rule exists because it was broken before.
2. `nethra/NETHRA_OPERATING_NOTES.md` — how to feed it, how to read it, what it is and is not.
3. `docs/HANDOFF.md` — current state and open work. `docs/HANDOFF_LOG.md` holds every measurement behind it.
4. Settled topics — read before touching them, do not re-derive:
   - `docs/BLOCKING.md` — blocking is the established order *refind -> subtract what is accounted for ->
     only the remainder earns evidence or construction*; receipts and a do-not list.
   - `nethra/NETHRA_MISTAKE_LEDGER.md` — mistakes already made once.
5. `docs/NETHRA_ROADMAP.md` — longer-range plan. `docs/CPU_PROFILE_2026-09-26.md` — where the time goes.

## Layout

| path | what |
|---|---|
| `nethra/nethra.py` | the core (field, evidence, construction, closure) |
| `nethra/tests/` | user tests and traces referred to by the handoff |
| `nethra/experiments/` | measurement harnesses (blocking scan, extension tests, parallel runner, prototypes) |
| `nethra/SCRIPTS.md` | one line per script |
| `docs/` | handoff, log, roadmap, settled topics, profile |
| `.github/workflows/` | self-hosted runner jobs (CUDA smoke, GPU bench, cycles) and runner utilities |

## Working flow

- **Health check first:** `nethra/tests/smoke.sh` (~40 s, one core). It checks determinism and checkpoint round
  trip (`bitcheck_cores.py`), blocking at the `human.py` design (0.18), context switching, cycles with a shared
  symbol, and a common factor with context held. Non-zero exit = the default core changed.
- **One numeric thread per field** (`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`) and a `timeout` on
  every run. A field is serial by construction (interval t needs t-1).
- **Parallelism = independent runs:** put one command per line in a file and run
  `python3 nethra/experiments/par.py JOBS [WORKERS]` — seeds, conditions and parameter points run in separate
  processes and scale with cores.
- **Keep streams compact.** Cost grows with field size: ~10 ms/interval at 115 Nethra, ~107 ms at 356
  (`docs/CPU_PROFILE_2026-09-26.md`). Most extension tests finish in 1-60 s.
- **Probe on a copy** with evidence change off (`NethraField.from_checkpoint_dict(f.checkpoint_dict())`), so reads
  never disturb the field.
- **Any execution speedup must be bit-identical:** `bitcheck_cores.py REF.py NEW.py` must print `ALL IDENTICAL`.

## Branches and history

`main` is the only branch. Every earlier branch is preserved as a tag `archive/<branch-name>`
(`git tag -l 'archive/*'`; restore with `git switch -c NAME archive/NAME`). Removed from `main` in the 2026-09-26
cleanup (still in `archive/main-before-cleanup-2026-09-26`): `bench/` (standalone persistence/decay benchmarks that
do not use the core) with its 13 workflows, and 6 feed/freeze workflows whose files no longer exist.
