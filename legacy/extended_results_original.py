#!/usr/bin/env python3
"""
run_sympy_jax_3rpr_extended_paper_results.py

Extended result generator for the SymPy--JAX reduced-coordinate dynamics paper.

It generates:
1. Constraint and velocity consistency over random workspace samples.
2. Conditioning statistics for A_p, J, and B.
3. Forward-dynamics sample table.
4. Runtime benchmarking:
   - non-JIT calls
   - JIT compile+first call
   - JIT repeated calls
   - batched JIT calls
5. Closed-loop task-space PD trajectory tracking:
   - desired and actual trajectory
   - tracking error
   - actuator forces
   - CSV summaries and PNG plots
6. Optional open-loop simulation.

Install:
    conda activate robotics
    cd /Users/apple/Downloads
    conda install -c conda-forge numpy scipy sympy matplotlib -y
    pip install -U jax jaxlib

Run:
    python run_sympy_jax_3rpr_extended_paper_results.py --n-samples 300 --condition-map

Outputs:
    paper_results_sympy_jax_extended/
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

os.environ.setdefault("JAX_ENABLE_X64", "True")

import numpy as np
import sympy as sp
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


@dataclass
class RobotParams:
    rA: Tuple[float, float] = (0.0, 0.0)
    rB: Tuple[float, float] = (1.0, 0.0)
    rC: Tuple[float, float] = (1.0, 2.0)
    a: Tuple[float, float, float] = (0.1, 0.1, 0.1)
    b: Tuple[float, float, float] = (0.3, 0.3, 0.3)
    c12: float = 0.15
    c23: float = 0.15
    c31: float = 0.15
    ma: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    mb: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    mG: float = 1.0
    Ia: Tuple[float, float, float] = (0.01, 0.01, 0.01)
    Ib: Tuple[float, float, float] = (0.01, 0.01, 0.01)
    IG: float = 0.01
    la: Tuple[float, float, float] = (0.05, 0.05, 0.05)
    lb: Tuple[float, float, float] = (0.15, 0.15, 0.15)


def load_params_from_robot_properties() -> RobotParams:
    try:
        import robot_properties as rp  # type: ignore
        raw = rp.robot_parameters()
        inert = rp.inertial_and_geometrical_parameters(raw)

        def get_any(d: Dict, *names, default=None):
            for n in names:
                if n in d:
                    return d[n]
            return default

        rA = (float(get_any(raw, "rax", "rAx", "r_Ax", default=0.0)),
              float(get_any(raw, "ray", "rAy", "r_Ay", default=0.0)))
        rB = (float(get_any(raw, "rbx", "rBx", "r_Bx", default=1.0)),
              float(get_any(raw, "rby", "rBy", "r_By", default=0.0)))
        rC = (float(get_any(raw, "rcx", "rCx", "r_Cx", default=1.0)),
              float(get_any(raw, "rcy", "rCy", "r_Cy", default=2.0)))

        a = tuple(float(get_any(raw, f"a{i}", default=0.1)) for i in [1, 2, 3])
        b = tuple(float(get_any(raw, f"b{i}", default=0.3)) for i in [1, 2, 3])
        c12 = float(get_any(raw, "c12", default=0.15))
        c23 = float(get_any(raw, "c23", default=0.15))
        c31 = float(get_any(raw, "c31", default=0.15))

        ma = tuple(float(get_any(inert, f"ma{i}", default=1.0)) for i in [1, 2, 3])
        mb = tuple(float(get_any(inert, f"mb{i}", default=1.0)) for i in [1, 2, 3])
        mG = float(get_any(inert, "mG", "mg", default=1.0))
        Ia = tuple(float(get_any(inert, f"Ia{i}", default=0.01)) for i in [1, 2, 3])
        Ib = tuple(float(get_any(inert, f"Ib{i}", default=0.01)) for i in [1, 2, 3])
        IG = float(get_any(inert, "IG", "Ig", default=0.01))
        la = tuple(float(get_any(inert, f"la{i}", default=0.05)) for i in [1, 2, 3])
        lb = tuple(float(get_any(inert, f"lb{i}", default=0.15)) for i in [1, 2, 3])

        print("Loaded parameters from robot_properties.py")
        return RobotParams(rA=rA, rB=rB, rC=rC, a=a, b=b,
                           c12=c12, c23=c23, c31=c31,
                           ma=ma, mb=mb, mG=mG,
                           Ia=Ia, Ib=Ib, IG=IG, la=la, lb=lb)
    except Exception as e:
        print("Could not load robot_properties.py. Using fallback parameters.")
        print("Reason:", repr(e))
        return RobotParams()


def platform_local_points_from_sides(c12: float, c23: float, c31: float):
    u = (c31**2 + c12**2 - c23**2) / (2.0 * c12)
    v = math.sqrt(max(c31**2 - u**2, 0.0))
    P = np.array([[0.0, 0.0], [c12, 0.0], [u, v]], dtype=float)
    P -= np.mean(P, axis=0)
    return P[0], P[1], P[2]


def build_sympy_model(params: RobotParams):
    d1, d2, d3 = sp.symbols("d1 d2 d3")
    th1, th2, th3 = sp.symbols("theta1 theta2 theta3")
    ph12, ph23, ph31 = sp.symbols("phi12 phi23 phi31")
    q_syms = [d1, d2, d3, th1, th2, th3, ph12, ph23, ph31]

    d1d, d2d, d3d = sp.symbols("d1_dot d2_dot d3_dot")
    th1d, th2d, th3d = sp.symbols("theta1_dot theta2_dot theta3_dot")
    ph12d, ph23d, ph31d = sp.symbols("phi12_dot phi23_dot phi31_dot")
    qd_syms = [d1d, d2d, d3d, th1d, th2d, th3d, ph12d, ph23d, ph31d]

    def e(alpha):
        return sp.Matrix([sp.cos(alpha), sp.sin(alpha)])

    rA = sp.Matrix(params.rA)
    rB = sp.Matrix(params.rB)
    rC = sp.Matrix(params.rC)

    a1, a2, a3 = params.a
    b1, b2, b3 = params.b
    c12, c23, c31 = params.c12, params.c23, params.c31

    f1 = (a1 + d1 + b1) * e(th1) + c12 * e(th1 + ph12) - (a2 + d2 + b2) * e(th2) - (rB - rA)
    f2 = (a2 + d2 + b2) * e(th2) + c23 * e(th2 + ph23) - (a3 + d3 + b3) * e(th3) - (rC - rB)
    f3 = (a3 + d3 + b3) * e(th3) + c31 * e(th3 + ph31) - (a1 + d1 + b1) * e(th1) - (rA - rC)

    Phi = sp.Matrix.vstack(f1, f2, f3)
    A = Phi.jacobian(sp.Matrix(q_syms))

    q_vec = sp.Matrix(q_syms)
    qd_vec = sp.Matrix(qd_syms)

    def time_derivative(vec):
        return vec.jacobian(q_vec) * qd_vec

    r_base = [rA, rB, rC]
    d = [d1, d2, d3]
    th = [th1, th2, th3]
    thd = [th1d, th2d, th3d]

    T = 0
    for i in range(3):
        r_ai = r_base[i] + params.la[i] * e(th[i])
        v_ai = time_derivative(r_ai)
        T += sp.Rational(1, 2) * params.ma[i] * v_ai.dot(v_ai) + sp.Rational(1, 2) * params.Ia[i] * thd[i]**2

        r_bi = r_base[i] + (params.a[i] + d[i] + params.lb[i]) * e(th[i])
        v_bi = time_derivative(r_bi)
        T += sp.Rational(1, 2) * params.mb[i] * v_bi.dot(v_bi) + sp.Rational(1, 2) * params.Ib[i] * thd[i]**2

    P1 = rA + (a1 + d1 + b1) * e(th1)
    P2 = rB + (a2 + d2 + b2) * e(th2)
    P3 = rC + (a3 + d3 + b3) * e(th3)
    G = (P1 + P2 + P3) / 3
    vG = time_derivative(G)
    psi_dot = th1d + ph12d
    T += sp.Rational(1, 2) * params.mG * vG.dot(vG) + sp.Rational(1, 2) * params.IG * psi_dot**2
    L = T

    modules = [{"sin": jnp.sin, "cos": jnp.cos, "sqrt": jnp.sqrt, "atan2": jnp.arctan2}, "jax"]
    Phi_fun = sp.lambdify(q_syms, Phi, modules=modules)
    A_fun = sp.lambdify(q_syms, A, modules=modules)
    L_fun = sp.lambdify(q_syms + qd_syms, L, modules=modules)
    return Phi_fun, A_fun, L_fun


def make_reduced_dynamics(params: RobotParams, Phi_fun, A_fun, L_fun):
    p1, p2, p3 = platform_local_points_from_sides(params.c12, params.c23, params.c31)
    p_local = jnp.array(np.stack([p1, p2, p3]), dtype=jnp.float64)
    base = jnp.array([params.rA, params.rB, params.rC], dtype=jnp.float64)
    a = jnp.array(params.a, dtype=jnp.float64)
    b = jnp.array(params.b, dtype=jnp.float64)

    def rot(psi):
        c = jnp.cos(psi)
        s = jnp.sin(psi)
        return jnp.array([[c, -s], [s, c]], dtype=jnp.float64)

    def q_from_x(x):
        xc, yc, psi = x
        R = rot(psi)
        center = jnp.array([xc, yc], dtype=jnp.float64)
        P = center[None, :] + (R @ p_local.T).T
        V = P - base

        lengths = jnp.linalg.norm(V, axis=1)
        d = lengths - a - b
        theta = jnp.arctan2(V[:, 1], V[:, 0])

        side12 = P[1] - P[0]
        side23 = P[2] - P[1]
        side31 = P[0] - P[2]

        alpha12 = jnp.arctan2(side12[1], side12[0])
        alpha23 = jnp.arctan2(side23[1], side23[0])
        alpha31 = jnp.arctan2(side31[1], side31[0])

        phi12 = alpha12 - theta[0]
        phi23 = alpha23 - theta[1]
        phi31 = alpha31 - theta[2]
        return jnp.concatenate([d, theta, jnp.array([phi12, phi23, phi31], dtype=jnp.float64)])

    def L_full(q, qd):
        return jnp.squeeze(jnp.array(L_fun(*(tuple(q) + tuple(qd))), dtype=jnp.float64))

    def L_reduced(x, xd):
        q = q_from_x(x)
        DqDx = jax.jacfwd(q_from_x)(x)
        qd = DqDx @ xd
        return L_full(q, qd)

    def DqaDx_fun(x):
        return jax.jacfwd(q_from_x)(x)[:3, :]

    def reduced_matrices(x, xd):
        B = jax.hessian(lambda vv: L_reduced(x, vv))(xd)
        C_L = jax.grad(lambda xx: L_reduced(xx, xd))(x)

        def dL_dxdot(xx):
            return jax.grad(lambda vv: L_reduced(xx, vv))(xd)

        A_L = jax.jacfwd(dL_dxdot)(x)
        return B, A_L, C_L

    def state_derivative(s, tau):
        x = s[:3]
        xd = s[3:]
        tau = jnp.asarray(tau, dtype=jnp.float64)
        B, A_L, C_L = reduced_matrices(x, xd)
        DqaDx = DqaDx_fun(x)
        Qx = DqaDx.T @ tau
        xdd = jnp.linalg.solve(B, Qx + C_L - A_L @ xd)
        return jnp.concatenate([xd, xdd])

    def platform_force_to_tau(x, Qx):
        DqaDx = DqaDx_fun(x)
        return jnp.linalg.solve(DqaDx.T, Qx)

    def diagnostics(s):
        x = jnp.asarray(s[:3], dtype=jnp.float64)
        xd = jnp.asarray(s[3:], dtype=jnp.float64)
        q = q_from_x(x)
        DqDx = jax.jacfwd(q_from_x)(x)
        qd = DqDx @ xd

        Phi = jnp.ravel(jnp.array(Phi_fun(*tuple(q)), dtype=jnp.float64))
        A = jnp.array(A_fun(*tuple(q)), dtype=jnp.float64)
        Ap = A[:, 3:]
        DqaDx = DqDx[:3, :]
        J = jnp.linalg.inv(DqaDx)
        B, _, _ = reduced_matrices(x, xd)

        return {
            "q": np.array(q),
            "qd": np.array(qd),
            "phi_residual": float(jnp.linalg.norm(Phi)),
            "vel_residual": float(jnp.linalg.norm(A @ qd)),
            "cond_Ap": float(jnp.linalg.cond(Ap)),
            "cond_J": float(jnp.linalg.cond(J)),
            "cond_B": float(jnp.linalg.cond(B)),
        }

    return q_from_x, L_reduced, state_derivative, platform_force_to_tau, diagnostics


def summarize(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return np.min(values), np.mean(values), np.median(values), np.max(values)


def random_workspace_tests(diagnostics, n_samples, seed, outdir):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_samples):
        s = np.array([rng.uniform(0.35, 0.75), rng.uniform(0.35, 0.75), rng.uniform(-0.5, 0.5),
                      rng.normal(0.0, 0.15), rng.normal(0.0, 0.15), rng.normal(0.0, 0.15)], dtype=float)
        try:
            d = diagnostics(jnp.array(s, dtype=jnp.float64))
            rows.append([d["phi_residual"], d["vel_residual"], d["cond_Ap"], d["cond_J"], d["cond_B"]])
        except Exception:
            rows.append([np.nan, np.nan, np.inf, np.inf, np.inf])

    data = np.array(rows, dtype=float)
    header = "phi_residual,velocity_residual,cond_Ap,cond_J,cond_B"
    np.savetxt(outdir / "random_workspace_results.csv", data, delimiter=",", header=header, comments="")

    print("\n===== RANDOM WORKSPACE TESTS =====")
    print(f"Samples: {n_samples}")
    with open(outdir / "random_workspace_summary.csv", "w") as f:
        f.write("quantity,min,mean,median,max\n")
        for i, name in enumerate(header.split(",")):
            mn, mean, med, mx = summarize(data[:, i])
            print(f"{name:20s}: min={mn:.4e}, mean={mean:.4e}, median={med:.4e}, max={mx:.4e}")
            f.write(f"{name},{mn},{mean},{med},{mx}\n")
    return data


def forward_dynamics_samples(state_derivative, outdir):
    s0 = jnp.array([0.45, 0.45, 0.10, 0.0, 0.0, 0.0], dtype=jnp.float64)
    inputs = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, -0.25, 0.15]]
    rows = []
    print("\n===== FORWARD DYNAMICS SAMPLES =====")
    for u in inputs:
        sd = np.array(state_derivative(s0, jnp.array(u, dtype=jnp.float64)))
        rows.append(np.r_[u, sd])
        print(f"tau={u} -> sdot={sd}")
    rows = np.array(rows)
    np.savetxt(outdir / "forward_dynamics_samples.csv", rows, delimiter=",",
               header="tau1,tau2,tau3,xdot,ydot,psidot,xddot,yddot,psiddot", comments="")
    return rows


def runtime_benchmark(state_derivative, outdir, n_repeats=100, batch_size=512):
    print("\n===== RUNTIME BENCHMARK =====")
    s = jnp.array([0.45, 0.45, 0.10, 0.1, -0.05, 0.02], dtype=jnp.float64)
    tau = jnp.array([0.5, -0.25, 0.15], dtype=jnp.float64)

    _ = np.array(state_derivative(s, tau))

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        _ = np.array(state_derivative(s, tau))
    t1 = time.perf_counter()
    nonjit_ms = 1000.0 * (t1 - t0) / n_repeats

    jitted = jax.jit(state_derivative)
    t0 = time.perf_counter()
    _ = np.array(jitted(s, tau))
    t1 = time.perf_counter()
    jit_first_ms = 1000.0 * (t1 - t0)

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        _ = np.array(jitted(s, tau))
    t1 = time.perf_counter()
    jit_ms = 1000.0 * (t1 - t0) / n_repeats

    rng = np.random.default_rng(0)
    S = np.zeros((batch_size, 6), dtype=float)
    S[:, 0] = rng.uniform(0.35, 0.75, batch_size)
    S[:, 1] = rng.uniform(0.35, 0.75, batch_size)
    S[:, 2] = rng.uniform(-0.3, 0.3, batch_size)
    S[:, 3:] = rng.normal(0, 0.1, (batch_size, 3))
    Tau = rng.normal(0, 0.3, (batch_size, 3))
    batched = jax.jit(jax.vmap(state_derivative, in_axes=(0, 0)))

    t0 = time.perf_counter()
    _ = np.array(batched(jnp.array(S, dtype=jnp.float64), jnp.array(Tau, dtype=jnp.float64)))
    t1 = time.perf_counter()
    batch_first_ms_total = 1000.0 * (t1 - t0)

    t0 = time.perf_counter()
    _ = np.array(batched(jnp.array(S, dtype=jnp.float64), jnp.array(Tau, dtype=jnp.float64)))
    t1 = time.perf_counter()
    batch_ms_total = 1000.0 * (t1 - t0)
    batch_ms_per_sample = batch_ms_total / batch_size

    print(f"Non-JIT repeated call       : {nonjit_ms:.6f} ms/call")
    print(f"JIT compile + first call    : {jit_first_ms:.6f} ms")
    print(f"JIT repeated call           : {jit_ms:.6f} ms/call")
    print(f"Batched JIT first total     : {batch_first_ms_total:.6f} ms for {batch_size} samples")
    print(f"Batched JIT repeated total  : {batch_ms_total:.6f} ms for {batch_size} samples")
    print(f"Batched JIT repeated/sample : {batch_ms_per_sample:.6f} ms/sample")

    with open(outdir / "runtime_benchmark.csv", "w") as f:
        f.write("mode,time_ms\n")
        f.write(f"nonjit_repeated_call,{nonjit_ms}\n")
        f.write(f"jit_compile_plus_first_call,{jit_first_ms}\n")
        f.write(f"jit_repeated_call,{jit_ms}\n")
        f.write(f"batched_jit_first_total_{batch_size},{batch_first_ms_total}\n")
        f.write(f"batched_jit_repeated_total_{batch_size},{batch_ms_total}\n")
        f.write(f"batched_jit_repeated_per_sample_{batch_size},{batch_ms_per_sample}\n")

    return nonjit_ms, jit_first_ms, jit_ms, batch_ms_per_sample


def reference_trajectory(t, T=4.0):
    w = 2.0 * np.pi / T
    x = 0.50 + 0.05 * np.cos(w * t)
    y = 0.50 + 0.05 * np.sin(w * t)
    psi = 0.10 + 0.05 * np.sin(w * t)
    xd = -0.05 * w * np.sin(w * t)
    yd = 0.05 * w * np.cos(w * t)
    psid = 0.05 * w * np.cos(w * t)
    return np.array([x, y, psi], dtype=float), np.array([xd, yd, psid], dtype=float)


def closed_loop_tracking(state_derivative, platform_force_to_tau, outdir, t_final=4.0,
                         kp=(80.0, 80.0, 20.0), kd=(16.0, 16.0, 8.0)):
    print("\n===== CLOSED-LOOP PD TRACKING =====")
    x0, xd0 = reference_trajectory(0.0, T=t_final)
    s0 = np.r_[x0, xd0]
    Kp = np.diag(kp)
    Kd = np.diag(kd)

    def controller(t, s):
        x = s[:3]
        xd = s[3:]
        xr, xdr = reference_trajectory(t, T=t_final)
        e = xr - x
        ed = xdr - xd
        Qx = Kp @ e + Kd @ ed
        tau = np.array(platform_force_to_tau(jnp.array(x, dtype=jnp.float64), jnp.array(Qx, dtype=jnp.float64)))
        return tau, e, ed, xr, xdr

    def rhs(t, s):
        tau, *_ = controller(t, s)
        return np.array(state_derivative(jnp.array(s, dtype=jnp.float64), jnp.array(tau, dtype=jnp.float64)))

    sol = solve_ivp(rhs, (0.0, t_final), s0, method="RK45", max_step=0.01, rtol=1e-6, atol=1e-8)

    times = sol.t
    Y = sol.y.T
    refs, refds, taus, errs, err_norms = [], [], [], [], []
    for t, s in zip(times, Y):
        tau, e, ed, xr, xdr = controller(float(t), s)
        refs.append(xr)
        refds.append(xdr)
        taus.append(tau)
        errs.append(e)
        err_norms.append(np.linalg.norm(e))

    refs = np.array(refs)
    refds = np.array(refds)
    taus = np.array(taus)
    errs = np.array(errs)
    err_norms = np.array(err_norms)

    rms_error = float(np.sqrt(np.mean(err_norms**2)))
    max_error = float(np.max(err_norms))
    rms_component = np.sqrt(np.mean(errs**2, axis=0))
    max_tau = np.max(np.abs(taus), axis=0)
    max_tau_norm = float(np.max(np.linalg.norm(taus, axis=1)))

    print("success:", sol.success)
    print("message:", sol.message)
    print("RMS tracking error norm:", rms_error)
    print("Max tracking error norm:", max_error)
    print("RMS component errors [x,y,psi]:", rms_component)
    print("Max |tau_i|:", max_tau)
    print("Max ||tau||:", max_tau_norm)

    all_data = np.column_stack([times, Y, refs, refds, errs, err_norms, taus])
    header = "t,x,y,psi,xdot,ydot,psidot,x_ref,y_ref,psi_ref,xdot_ref,ydot_ref,psidot_ref,e_x,e_y,e_psi,error_norm,tau1,tau2,tau3"
    np.savetxt(outdir / "closed_loop_tracking_trajectory.csv", all_data, delimiter=",", header=header, comments="")

    with open(outdir / "closed_loop_tracking_summary.csv", "w") as f:
        f.write("metric,value\n")
        f.write(f"success,{sol.success}\n")
        f.write(f"rms_error_norm,{rms_error}\n")
        f.write(f"max_error_norm,{max_error}\n")
        f.write(f"rms_error_x,{rms_component[0]}\n")
        f.write(f"rms_error_y,{rms_component[1]}\n")
        f.write(f"rms_error_psi,{rms_component[2]}\n")
        f.write(f"max_abs_tau1,{max_tau[0]}\n")
        f.write(f"max_abs_tau2,{max_tau[1]}\n")
        f.write(f"max_abs_tau3,{max_tau[2]}\n")
        f.write(f"max_tau_norm,{max_tau_norm}\n")

    plt.figure()
    plt.plot(refs[:, 0], refs[:, 1], "--", label="desired")
    plt.plot(Y[:, 0], Y[:, 1], label="actual")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Closed-loop x-y tracking")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "closed_loop_xy_tracking.png", dpi=300)
    plt.close()

    labels = ["x", "y", "psi"]
    for i, label in enumerate(labels):
        plt.figure()
        plt.plot(times, refs[:, i], "--", label=f"{label} desired")
        plt.plot(times, Y[:, i], label=f"{label} actual")
        plt.xlabel("Time [s]")
        plt.ylabel(label)
        plt.title(f"Closed-loop tracking: {label}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(outdir / f"closed_loop_{label}_tracking.png", dpi=300)
        plt.close()

    plt.figure()
    plt.plot(times, err_norms)
    plt.xlabel("Time [s]")
    plt.ylabel(r"$||x_d-x||_2$")
    plt.title("Closed-loop tracking error norm")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "closed_loop_error_norm.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(times, taus[:, 0], label=r"$\tau_1$")
    plt.plot(times, taus[:, 1], label=r"$\tau_2$")
    plt.plot(times, taus[:, 2], label=r"$\tau_3$")
    plt.xlabel("Time [s]")
    plt.ylabel("Actuator force")
    plt.title("Closed-loop actuator forces")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "closed_loop_actuator_forces.png", dpi=300)
    plt.close()

    return sol


def open_loop_simulation(state_derivative, outdir, t_final=2.0):
    print("\n===== OPEN-LOOP STATE-SPACE SIMULATION =====")
    s0 = np.array([0.45, 0.45, 0.10, 0.0, 0.0, 0.0], dtype=float)
    tau = np.array([0.5, -0.25, 0.15], dtype=float)

    def rhs(t, s):
        return np.array(state_derivative(jnp.array(s, dtype=jnp.float64), jnp.array(tau, dtype=jnp.float64)))

    sol = solve_ivp(rhs, (0.0, t_final), s0, method="RK45", max_step=0.02, rtol=1e-6, atol=1e-8)
    print("success:", sol.success)
    print("message:", sol.message)
    print("final state:", sol.y[:, -1])
    np.savetxt(outdir / "open_loop_trajectory.csv", np.column_stack([sol.t, sol.y.T]), delimiter=",",
               header="t,x,y,psi,xdot,ydot,psidot", comments="")
    return sol


def condition_maps(diagnostics, outdir, nx=30, ny=30):
    print("\n===== CONDITION MAPS =====")
    xs = np.linspace(0.35, 0.75, nx)
    ys = np.linspace(0.35, 0.75, ny)
    maps = {name: np.full((ny, nx), np.nan) for name in ["Ap", "J", "B"]}

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            s = jnp.array([x, y, 0.1, 0.0, 0.0, 0.0], dtype=jnp.float64)
            try:
                d = diagnostics(s)
                maps["Ap"][iy, ix] = d["cond_Ap"]
                maps["J"][iy, ix] = d["cond_J"]
                maps["B"][iy, ix] = d["cond_B"]
            except Exception:
                pass

    for name, arr in maps.items():
        np.savetxt(outdir / f"condition_map_{name}.csv", arr, delimiter=",")
        plt.figure()
        plt.imshow(np.log10(np.clip(arr, 1.0, 1e8)), extent=[xs[0], xs[-1], ys[0], ys[-1]], origin="lower", aspect="auto")
        plt.colorbar(label=f"log10 cond({name})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Condition number map of {name}")
        plt.tight_layout()
        plt.savefig(outdir / f"condition_map_{name}.png", dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--outdir", type=str, default="paper_results_sympy_jax_extended")
    parser.add_argument("--condition-map", action="store_true")
    parser.add_argument("--open-loop", action="store_true")
    parser.add_argument("--tracking-time", type=float, default=4.0)
    parser.add_argument("--runtime-repeats", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Building SymPy--JAX model...")
    params = load_params_from_robot_properties()
    Phi_fun, A_fun, L_fun = build_sympy_model(params)
    _, _, state_derivative, platform_force_to_tau, diagnostics = make_reduced_dynamics(params, Phi_fun, A_fun, L_fun)

    s_test = jnp.array([0.45, 0.45, 0.10, 0.10, -0.05, 0.02], dtype=jnp.float64)
    tau_test = jnp.array([0.5, -0.25, 0.15], dtype=jnp.float64)
    d = diagnostics(s_test)
    sd = state_derivative(s_test, tau_test)

    print("\n===== SINGLE TEST =====")
    print("||Phi(q)||    =", d["phi_residual"])
    print("||A(q) qdot|| =", d["vel_residual"])
    print("cond(Ap)      =", d["cond_Ap"])
    print("cond(J)       =", d["cond_J"])
    print("cond(B)       =", d["cond_B"])
    print("sdot          =", np.array(sd))

    with open(outdir / "single_test_summary.csv", "w") as f:
        f.write("quantity,value\n")
        f.write(f"phi_residual,{d['phi_residual']}\n")
        f.write(f"velocity_residual,{d['vel_residual']}\n")
        f.write(f"cond_Ap,{d['cond_Ap']}\n")
        f.write(f"cond_J,{d['cond_J']}\n")
        f.write(f"cond_B,{d['cond_B']}\n")
        for i, val in enumerate(np.array(sd)):
            f.write(f"sdot_{i},{val}\n")

    random_workspace_tests(diagnostics, args.n_samples, args.seed, outdir)
    forward_dynamics_samples(state_derivative, outdir)
    runtime_benchmark(state_derivative, outdir, n_repeats=args.runtime_repeats, batch_size=args.batch_size)
    closed_loop_tracking(state_derivative, platform_force_to_tau, outdir, t_final=args.tracking_time)

    if args.open_loop:
        open_loop_simulation(state_derivative, outdir, t_final=2.0)

    if args.condition_map:
        condition_maps(diagnostics, outdir)

    print("\nDone. Outputs saved in:", outdir.resolve())


if __name__ == "__main__":
    main()
