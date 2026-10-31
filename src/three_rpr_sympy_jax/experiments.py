"""Paper experiments (Tables III--VI, Figs. 2--3) and optional diagnostics."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def summarize(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan, np.nan
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
    print(f"Samples: {n_samples}; failed/nonfinite rows: {int(np.count_nonzero(~np.isfinite(data).all(axis=1)))}")
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

    jax.block_until_ready(state_derivative(s, tau))

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        jax.block_until_ready(state_derivative(s, tau))
    t1 = time.perf_counter()
    nonjit_ms = 1000.0 * (t1 - t0) / n_repeats

    jitted = jax.jit(state_derivative)
    t0 = time.perf_counter()
    jitted(s, tau).block_until_ready()
    t1 = time.perf_counter()
    jit_first_ms = 1000.0 * (t1 - t0)

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        jitted(s, tau).block_until_ready()
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

    S_device = jnp.array(S, dtype=jnp.float64)
    Tau_device = jnp.array(Tau, dtype=jnp.float64)
    t0 = time.perf_counter()
    batched(S_device, Tau_device).block_until_ready()
    t1 = time.perf_counter()
    batch_first_ms_total = 1000.0 * (t1 - t0)

    t0 = time.perf_counter()
    batched(S_device, Tau_device).block_until_ready()
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
    state_derivative = jax.jit(state_derivative)
    platform_force_to_tau = jax.jit(platform_force_to_tau)

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

    if not sol.success:
        raise RuntimeError(f"Closed-loop integration failed: {sol.message}")
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
    state_derivative = jax.jit(state_derivative)
    s0 = np.array([0.45, 0.45, 0.10, 0.0, 0.0, 0.0], dtype=float)
    tau = np.array([0.5, -0.25, 0.15], dtype=float)

    def rhs(t, s):
        return np.array(state_derivative(jnp.array(s, dtype=jnp.float64), jnp.array(tau, dtype=jnp.float64)))

    sol = solve_ivp(rhs, (0.0, t_final), s0, method="RK45", max_step=0.02, rtol=1e-6, atol=1e-8)
    print("success:", sol.success)
    print("message:", sol.message)
    print("final state:", sol.y[:, -1])
    if not sol.success:
        raise RuntimeError(f"Open-loop integration failed: {sol.message}")
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
