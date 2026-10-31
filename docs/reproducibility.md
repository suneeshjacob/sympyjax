# Reproducibility and numerical interpretation

## Model and assumptions

The package directly follows the two Python scripts supplied with the manuscript **“A SymPy–JAX–based Implementation of Forward Dynamics for a Planar 3-RPR Parallel Manipulator.”** The generalized coordinates are `q = [d1,d2,d3,theta1,theta2,theta3,phi12,phi23,phi31]`; the reduced coordinates are the triangular platform centroid and heading `x = [xc,yc,psi]`. The source scripts use `L=T`, for ideal horizontal motion, with no gravity, friction, damping, clearance, contact, motor/electrical dynamics, or actuator limits.

The default values match the *printed parameters* of Tables I and II. The original scripts optionally attempt to import an external `robot_properties.py`, silently reverting to built-in values on import errors. **That external file was not supplied.** This refactor uses the printed defaults unless a JSON parameter file is explicitly provided; it never silently imports third-party or local parameter values.

## Paper-to-command mapping

| Paper | Reproduction command | Main outputs |
| --- | --- | --- |
| Eq. 40–43 single-state diagnostics | `--only single` | `single_test_summary.csv` |
| Table III random samples and conditioning | `--only workspace --n-samples 300 --seed 7` | `random_workspace_results.csv`, `random_workspace_summary.csv` |
| Table V forward-dynamics samples | `--only forward` | `forward_dynamics_samples.csv` |
| Table IV runtime/JIT/batch | `--only benchmark --runtime-repeats 100 --batch-size 512` | `runtime_benchmark.csv` |
| Fig. 2–3 / Table VI PD tracking | `--only tracking --tracking-time 4.0` | `closed_loop_*.csv`, `closed_loop_*.png` |
| Optional diagnostic maps | `--only condition-map --map-nx 30 --map-ny 30` | `condition_map_*.csv/png` |
| Optional open-loop simulation | `--only open-loop` | `open_loop_trajectory.csv` |

Every CLI run writes `run_metadata.json` with geometric/inertial settings, random seed, dependency versions, JAX backend, and relevant run options.

## IMPORTANT: reported numerical values are **not** reproduced by the supplied scripts under their default parameters

We evaluated the *unchanged* supplied mathematical core with its explicit built-in defaults, on Python/JAX in the execution environment. At the paper's test state `s = [.45,.45,.10,.10,-.05,.02]` with `tau=[.5,-.25,.15]`, we obtained:

| Diagnostic | Paper | Supplied implementation with built-in defaults |
| --- | ---: | ---: |
| `||Phi||_2` | `5.00e-16` | `5.04e-16` |
| `||A qdot||_2` | `4.22e-17` | `7.14e-17` |
| `cond(Ap)` | `50.22` | `127.12` |
| `cond(J)` | `45.98` | `59.79` |
| `cond(Mr)` | `73.85` | `140.46` |
| `ddot(x), ddot(y), ddot(psi)` | `[.1016,.0105,-.2970]` | `[.1346,.0125,-.1616]` |

For the 4-second PD tracking experiment, the paper reports an RMS tracking-error norm of `7.92e-3` and maximum actuator-force norm `2.214`. The supplied default-parameter implementation evaluated in this environment produced `5.63e-3` and `0.954` respectively. These are further discrepancies that should be investigated; they do not indicate a validated improvement.

At *zero velocity* with `tau=[1,0,0]`, Table V reports `[.1404,.1492,-.4472]`; the supplied implementation yields approximately `[.1904,.1978,-.3537]` with built-in defaults.

This discrepancy **exists in the original provided implementation** and was not introduced by restructuring. Possible explanations include a missing original `robot_properties.py`, alternate geometry/inertia choices, or a different code revision. We **cannot establish which explanation is correct** from the supplied material. Do not describe the included code as an exact reproduction of the paper's numerical tables until the underlying source of the differences is resolved. Providing the precise parameter source or the script/revision that produced the manuscript's tables would permit further reconciliation.

The repository's numerical tests check the mathematical pipeline for internal consistency and expected behavior; they intentionally **do not assert** the paper's currently unreproduced numeric values.

## Conditioning and interpretation

All inverse solves in the supplied formulation assume a nonsingular configuration. The script computes condition numbers of `Ap`, the task-space Jacobian `J = (Dqa/Dx)^(-1)`, and the reduced inertia `Mr`. Near singularities, direct solves can be unstable; outputs should not be interpreted as valid across the whole joint or Cartesian workspace. Workspace sampling in the paper uses fixed rectangular coordinate bounds, not a physical workspace feasibility filter; some random states may be badly conditioned.

For rigid horizontal free-space motion, the mass matrix is obtained via a velocity Hessian of the reduced Lagrangian. The paper's PD tracking is a model-in-the-loop diagnostic only; it is not hardware validation, a robust stability guarantee, or a general-purpose controller.

## Runtime fairness

Elapsed times depend on CPU/GPU, JAX/XLA version, thread scheduling and cache state. For accurate JAX timing, the packaged benchmark forces synchronization by calling `.block_until_ready()` and constructs device-side batch inputs before starting the batch timer. The original script converted every output to NumPy during timing; the packaged version times device execution more directly. Therefore **published times are not expected to match numerically**, even when dynamics agree.

For a full environment snapshot, keep `run_metadata.json` and the generated result CSVs with any submitted reproduction package.
