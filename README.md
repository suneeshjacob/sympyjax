# SymPy–JAX forward dynamics for a planar 3-RPR parallel manipulator

Research implementation accompanying **“A SymPy–JAX–based Implementation of Forward Dynamics for a Planar 3-RPR Parallel Manipulator”**, by Akkarapakam Suneesh Jacob and Sivasundar Manisankar.

This package reorganizes two author-provided Python scripts into a reusable, installable Python project. It retains their mathematical definitions while offering reproducible command-line experiments, CSV tables, figures, and mathematical consistency tests.

> **Reproducibility note:** The code's built-in parameter values match the parameters printed in the paper, but **some computed numerical values do not match the paper's tables**. The author-provided scripts appear to depend optionally on a `robot_properties.py` that was not supplied. See [detailed numerical comparison](docs/reproducibility.md) before claiming exact reproduction. Published benchmark timings are system-dependent.

## What the code implements

1. SymPy loop-closure vector `Phi(q)` and constraint Jacobian `A(q)` for 3-RPR kinematics.
2. Symbolic kinetic-energy Lagrangian `L(q, qdot)` using leg and platform inertias.
3. Geometric reduction to the platform centroid pose `x=[xc,yc,psi]` and velocity `xdot`.
4. JAX automatic differentiation for reduced mass matrix `Mr`, mixed derivatives `AL`, and position gradient `CL`.
5. Forward dynamics `xddot = solve(Mr, DqaDx.T @ tau + CL - AL @ xdot)`.
6. Constraint consistency, conditioning statistics, forward-dynamics examples, JIT and batched runtime evaluation, PD trajectory tracking, and optional open-loop and condition-map experiments.

The model assumes a rigid, **horizontal** three-DOF 3-RPR mechanism; its results depend on working away from singular configurations.

## Setup

Requires Python 3.10+; CPU mode is sufficient on Linux, macOS (including Apple Silicon), and Windows.

```bash
git clone https://github.com/YOUR-USERNAME/3rpr-sympy-jax.git
cd 3rpr-sympy-jax
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Replace `YOUR-USERNAME` with the actual repository account once it is uploaded. JAX CPU installation with `pip` is sufficient; no CUDA is necessary. On GPU systems, consult the official JAX platform-specific installation instructions.

## Run

**Fast first check** (single state + five random samples; does not run the full paper):

```bash
python -m three_rpr_sympy_jax --only single workspace forward --n-samples 5 --outdir results/smoke
```

**Full paper workflow** (300 samples, 100 repetitions, 512 batched states, 4-second PD trajectory):

```bash
python -m three_rpr_sympy_jax --outdir results/paper
# or: bash scripts/reproduce_paper.sh
```

**Individual experiments:**

```bash
python -m three_rpr_sympy_jax --only benchmark --runtime-repeats 100 --batch-size 512
python -m three_rpr_sympy_jax --only tracking --tracking-time 4.0
python -m three_rpr_sympy_jax --only condition-map --map-nx 15 --map-ny 15
python -m three_rpr_sympy_jax --only open-loop --open-loop-time 0.5
python -m three_rpr_sympy_jax --only single --params config/paper_parameters.json
```

`--only` chooses one or more named experiments. Omitting it runs the five paper experiments; `--condition-map` and `--open-loop` add optional experiments. The `three-rpr-experiments` console command is equivalent to `python -m three_rpr_sympy_jax`.

### Outputs

By default, the CLI writes into `results/paper/` (ignored by Git). Each run includes `run_metadata.json` that records parameters, software versions, backend and seed. Outputs include:

| File | Content |
| --- | --- |
| `single_test_summary.csv` | Single-state constraints, condition numbers and state derivative |
| `random_workspace_results.csv` / `random_workspace_summary.csv` | 300-sample residuals and condition-number statistics |
| `forward_dynamics_samples.csv` | Actuator inputs and resulting platform accelerations |
| `runtime_benchmark.csv` | Uncompiled, first-compiled, repeated JIT and batched timings |
| `closed_loop_tracking_trajectory.csv` / `_summary.csv` | Controller trajectory and errors |
| `closed_loop_xy_tracking.png`, `closed_loop_error_norm.png` | Paper-equivalent trajectory and tracking-error figures |
| `condition_map_*.png` | Optional position-dependent conditioning maps |

See [docs/reproducibility.md](docs/reproducibility.md) for a paper-section-to-command map and interpretation limits.

## Reuse as a Python library

```python
import jax
import jax.numpy as jnp
from three_rpr_sympy_jax import RobotParams, create_model

q_from_x, L_reduced, dynamics, force_to_tau, diagnostics = create_model(RobotParams())
state = jnp.array([0.45, 0.45, 0.10, 0.10, -0.05, 0.02])
tau = jnp.array([0.5, -0.25, 0.15])
state_dot = jax.jit(dynamics)(state, tau)
print(state_dot)
```

See [examples/basic_usage.py](examples/basic_usage.py). Parameter values are in [config/paper_parameters.json](config/paper_parameters.json). Passing `--params path.json` uses exactly that JSON configuration; unlike the original scripts, the package does **not** silently import an unspecified `robot_properties.py`.

## Tests

```bash
python -m pytest -q
```

Tests verify geometric consistency, constraints, velocity closure, mass-matrix symmetry/positive definiteness at a nonsingular pose, and the zero-input zero-velocity equilibrium. These tests are not independent experimental validations.

## Repository structure

```text
src/three_rpr_sympy_jax/   package: parameters, symbolic model, dynamics, experiments, CLI
config/                   paper's printed parameter set in JSON
examples/                 minimal Python integration example
scripts/                  full-experiment reproduction command
docs/                     paper mapping and numerical discrepancies
tests/                    scientific regression and smoke tests
legacy/                   verbatim copies of both user-provided scripts
.github/workflows/        automated Python test workflow
```

## Citation

When using this code, cite the accompanying manuscript. Its publication venue, year, DOI, and bibliographic record were **not** provided, so the included `CITATION.cff` identifies it as an unpublished manuscript. Update that record when publication metadata becomes available.

## Publishing on GitHub

Create an empty GitHub repository (for example, `3rpr-sympy-jax`), extract this ZIP into its root, and push with:

```bash
git init
git add .
git commit -m "Initial research code release"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/3rpr-sympy-jax.git
git push -u origin main
```

If the GitHub repository was created with its own README or license, pull/reconcile those commits before pushing. The attached paper PDF is intentionally not included because public distribution rights were not specified.

## License

**No open-source license has been selected for this repository.** Before publishing, choose and add an appropriate `LICENSE` file, after confirming ownership and any coauthor/publisher conditions. Absence of a license means GitHub viewers do not automatically receive permission to reuse, modify, or redistribute the code.
