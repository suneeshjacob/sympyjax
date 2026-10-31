#!/usr/bin/env bash
set -euo pipefail
# Run after `python -m pip install -e .` from the repository root.
# Includes all paper experiments; results and timings depend on environment.
python -m three_rpr_sympy_jax \
  --n-samples 300 --seed 7 \
  --runtime-repeats 100 --batch-size 512 \
  --tracking-time 4.0 \
  --outdir results/paper
