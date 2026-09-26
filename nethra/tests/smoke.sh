#!/bin/bash
# One-command health check for main. One numeric thread, a timeout on every step. ~1 minute on one core.
# Expected values are from the default core; a change in any of them means the core changed.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
fail=0
step() { local name=$1 want=$2; shift 2; local out line; out=$(timeout 120 "$@" 2>&1)
  line=$(echo "$out" | grep -F -- "$want" | head -1)
  if [ -n "$line" ]; then echo "ok   $name: $line"; else echo "FAIL $name (want '$want'): $(echo "$out" | tail -1)"; fail=1; fi; }
step "determinism + checkpoint round trip" "ALL IDENTICAL" env N=60 python3 bitcheck_cores.py ../nethra.py ../nethra.py
step "blocking (human.py design, 30 compound)" "live 0.0050/0.0285 = 0.18" python3 ../experiments/blocking/bl.py
step "context switching, 12 cues" "switch K=12" python3 ../experiments/ext.py switch 12 20
step "cycles sharing a symbol" "cycles L=6" python3 ../experiments/ext.py cycles 6 80
step "common factor, context held" "0.87 1.00 1.00 1.00" env HOLD=1 python3 ../experiments/ext.py common 6 30
exit $fail
