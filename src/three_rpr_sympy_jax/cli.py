"""Command-line experiment runner for the paper's numerical sections."""
from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import scipy
import sympy

from . import __version__
from .model import create_model
from .parameters import RobotParams, parameters_from_json, parameters_to_dict
from .experiments import (
    closed_loop_tracking, condition_maps, forward_dynamics_samples,
    open_loop_simulation, random_workspace_tests, runtime_benchmark,
)

EXPERIMENTS = ("single", "workspace", "forward", "benchmark", "tracking", "open-loop", "condition-map")
DEFAULT = ("single", "workspace", "forward", "benchmark", "tracking")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Reproduce/inspect the paper's planar 3-RPR SymPy--JAX workflow."
    )
    parser.add_argument("--only", nargs="+", choices=(*EXPERIMENTS, "all"), default=["all"],
                        help="Select experiments. 'all' runs paper defaults; optional tests require --condition-map/--open-loop.")
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--open-loop-time", type=float, default=2.0,
                        help="Duration of optional open-loop simulation (default 2 s).")
    parser.add_argument("--tracking-time", type=float, default=4.0,
                        help="Integration duration and desired trajectory period (paper: 4 s).")
    parser.add_argument("--runtime-repeats", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--outdir", type=Path, default=Path("results/paper"))
    parser.add_argument("--params", type=Path, default=None,
                        help="Optional JSON parameter overrides. Default = Tables I and II.")
    parser.add_argument("--open-loop", action="store_true", help="Also run optional open-loop simulation.")
    parser.add_argument("--condition-map", action="store_true", help="Also generate optional conditioning maps.")
    parser.add_argument("--map-nx", type=int, default=30)
    parser.add_argument("--map-ny", type=int, default=30)
    args = parser.parse_args(argv)
    if args.n_samples < 1 or args.runtime_repeats < 1 or args.batch_size < 1:
        parser.error("--n-samples, --runtime-repeats, and --batch-size must be positive")
    if args.tracking_time <= 0 or args.open_loop_time <= 0 or args.map_nx < 2 or args.map_ny < 2:
        parser.error("--tracking-time must be positive, map dimensions >= 2")
    return args


def _metadata(args, params, selected):
    return {
        "source": "Provided scripts for 'A SymPy-JAX-based Implementation of Forward Dynamics for a Planar 3-RPR Parallel Manipulator'",
        "package_version": __version__,
        "experiment_names": sorted(selected),
        "seed": args.seed,
        "random_samples": args.n_samples,
        "tracking_time_s": args.tracking_time,
        "open_loop_time_s": args.open_loop_time,
        "runtime_repeats": args.runtime_repeats,
        "benchmark_batch_size": args.batch_size,
        "parameters": parameters_to_dict(params),
        "python": sys.version.split()[0],
        "numpy": np.__version__, "jax": jax.__version__,
        "sympy": sympy.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(), "jax_backend": jax.default_backend(),
        "jax_x64": bool(jax.config.jax_enable_x64),
        "environment_note": "Benchmark values are machine-, backend-, and software-version-dependent.",
    }


def run(argv=None):
    args = parse_args(argv)
    params = parameters_from_json(args.params) if args.params else RobotParams()
    selected = set(DEFAULT if "all" in args.only else args.only)
    if args.open_loop:
        selected.add("open-loop")
    if args.condition_map:
        selected.add("condition-map")
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "run_metadata.json").write_text(
        json.dumps(_metadata(args, params, selected), indent=2) + "\n", encoding="utf-8"
    )

    print("Constructing symbolic model and JAX reduced dynamics...", flush=True)
    q_from_x, L_reduced, state_derivative, platform_force_to_tau, diagnostics = create_model(params)

    if "single" in selected:
        s = jnp.array([.45, .45, .10, .10, -.05, .02], dtype=jnp.float64)
        tau = jnp.array([.5, -.25, .15], dtype=jnp.float64)
        d = diagnostics(s)
        sd = np.asarray(state_derivative(s, tau))
        print("\n=== SINGLE-STATE DIAGNOSTICS ===")
        for key in ("phi_residual", "vel_residual", "cond_Ap", "cond_J", "cond_B"):
            print(f"{key}: {d[key]}")
        print("state derivative:", sd)
        with (args.outdir / "single_test_summary.csv").open("w", encoding="utf-8") as f:
            f.write("quantity,value\n")
            for key in ("phi_residual", "vel_residual", "cond_Ap", "cond_J", "cond_B"):
                f.write(f"{key},{d[key]:.17g}\n")
            for i, value in enumerate(sd):
                f.write(f"sdot_{i},{value:.17g}\n")

    if "workspace" in selected:
        random_workspace_tests(diagnostics, args.n_samples, args.seed, args.outdir)
    if "forward" in selected:
        forward_dynamics_samples(state_derivative, args.outdir)
    if "benchmark" in selected:
        runtime_benchmark(state_derivative, args.outdir, args.runtime_repeats, args.batch_size)
    if "tracking" in selected:
        closed_loop_tracking(state_derivative, platform_force_to_tau,
                             args.outdir, t_final=args.tracking_time)
    if "open-loop" in selected:
        open_loop_simulation(state_derivative, args.outdir, t_final=args.open_loop_time)
    if "condition-map" in selected:
        condition_maps(diagnostics, args.outdir, nx=args.map_nx, ny=args.map_ny)

    print("\nFinished. Outputs:", args.outdir.resolve(), flush=True)
    return 0


def main():
    raise SystemExit(run())


if __name__ == "__main__":
    main()
