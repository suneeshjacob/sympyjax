"""Use the 3-RPR dynamics as a callable function in another Python project."""
import jax
import jax.numpy as jnp
import numpy as np
from three_rpr_sympy_jax import RobotParams, create_model

params = RobotParams()
q_from_x, L_reduced, state_derivative, platform_force_to_tau, diagnostics = create_model(params)
s = jnp.array([.45, .45, .10, .10, -.05, .02], dtype=jnp.float64)
tau = jnp.array([.5, -.25, .15], dtype=jnp.float64)
print("Generalized coordinates q:", np.asarray(q_from_x(s[:3])))
print("Constraint/condition diagnostics:", diagnostics(s))
print("State derivative sdot:", np.asarray(jax.jit(state_derivative)(s, tau)))
print("Actuator loads for Qx=[1,0,0]:",
      np.asarray(platform_force_to_tau(s[:3], jnp.array([1., 0., 0.]))))
