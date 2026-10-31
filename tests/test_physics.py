"""Internal scientific sanity checks independent of unreproduced paper tables."""
import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose


def test_loop_closure_and_velocity(model):
    q, lagrangian, state_derivative, to_tau, diagnostics = model
    state = jnp.array([.45, .45, .10, .10, -.05, .02])
    d = diagnostics(state)
    assert d["phi_residual"] < 1e-10
    assert d["vel_residual"] < 1e-10
    assert np.all(np.isfinite(d["q"]))
    assert np.all(np.isfinite(d["qd"]))


def test_zero_force_zero_velocity_is_equilibrium(model):
    _, _, state_derivative, _, _ = model
    state = jnp.array([.45, .45, .10, 0., 0., 0.])
    assert_allclose(np.asarray(state_derivative(state, jnp.zeros(3))), 0, atol=1e-10)


def test_mass_matrix_symmetry_and_positive_definiteness(model):
    _, lagrangian, _, _, _ = model
    x = jnp.array([.45, .45, .10])
    M = np.asarray(jax.hessian(lambda v: lagrangian(x, v))(jnp.zeros(3)))
    assert_allclose(M, M.T, atol=1e-10)
    assert np.linalg.eigvalsh(M).min() > 0


def test_state_derivative_first_half_equals_velocity(model):
    _, _, state_derivative, _, _ = model
    state = jnp.array([.45, .45, .10, .10, -.05, .02])
    out = np.asarray(state_derivative(state, jnp.array([.5, -.25, .15])))
    assert_allclose(out[:3], np.asarray(state[3:]), atol=1e-12)
    assert np.isfinite(out).all()


def test_virtual_work_force_mapping(model):
    q_from_x, _, _, to_tau, _ = model
    x = jnp.array([.45, .45, .10])
    Q = jnp.array([1.0, -.3, .2])
    DqaDx = np.asarray(jax.jacfwd(q_from_x)(x))[:3]
    tau = np.asarray(to_tau(x, Q))
    assert_allclose(DqaDx.T @ tau, np.asarray(Q), atol=1e-9)
