# CPU profile, default core a41200d (2026-09-26)

Container: 1 core, no GPU. All runs one thread (OMP/BLAS = 1). cProfile, 28-regime stream, fields grown to size.

| Nethra | evidence keys | ms/interval | ETD integration | of which `_etd_prepare` / `eigh` | evidence move | of which `update_residuals` |
|---|---|---|---|---|---|---|
| 159 | 1,084 | 22 | 7.6 | 3.8 / 1.9 | 8.1 | 2.4 |
| 257 | 2,626 | 54 | 20.8 | 11.1 / 5.2 | 19.5 | 7.7 |
| 356 | 4,772 | 107 | 44.0 | 25.2 / 11.1 | 39.9 | 19.5 |

Above `etd_max_nodes` (400) the integrator switches to RK4; growing from 350 to 450 Nethra took ~130 ms/interval.

- ETD (~40%): one eigendecomposition + 7 formed N x N operators per interval, O(N^3). Multithreaded BLAS or a GPU
  applies here only. `_etd_prepare` is already cached between the zero-source and actual runs.
- Evidence move + residual pair statistics + incidence compile (~55-60%): per-incidence Python dict work and
  co-supplier pairs (degree^2). Not GPU work, not threadable under the GIL. The structural fix is array storage of
  incidence evidence (roadmap), which is also what would let CUDA pay off.
- Parallel runs: `nethra/experiments/par.py` runs independent jobs (seeds, conditions) in processes; scales with cores.

Candidate needing a decision (not bit-identical, rounding-level):
`nethra/experiments/etd_operator_free_prototype.py` applies the ETD operators as Q (f * (Q^T x)) instead of forming 7
N x N matrices. Max activation difference 5e-16 after 30 intervals; same Nethra and evidence counts.
250 Nethra: 39.9 -> 36.5 ms/interval (-9%). 350 Nethra: 90.7 -> 73.1 (-19%). Saving grows with N.

Checked and not worth it: skipping zero-delta evidence writes (the check costs as much as the write); de-duplicating
pair codes (codes repeat across receivers, the sort is needed).
