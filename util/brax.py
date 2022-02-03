import jax
from jax import numpy as jnp

import brax
from brax.envs import env

from util.types import *

from functools import partial


def get_n_dmp(env: env.Env):
    dummy_key = jax.random.PRNGKey(0)
    dummy_state = env.reset(dummy_key)
    dummy_state_dmp = brax_state_to_dmp_state(env.sys, dummy_state)
    n_dmp = dummy_state_dmp.y.shape[-1]
    return n_dmp
    if cfg.DMP.N_DMP != n_dmp:
        warnings.warn(
            "cfg.DMP.N_DMP is incorrect; raised due to cfg.DMP.INFER_STATE=True."
        )


@partial(jax.vmap, in_axes=(0, None, 0))
def brax_state_to_dmp_state(sys: brax.physics.system.System,
                            state: env.State, x: float = 1.) -> StateDMP:
    """Estimates changes in qp.pos and qp.rot under Euler integration
    """
    qp = state.qp
    qp_next = sys.integrator.kinetic(qp)
    qp_vel = (qp_next.pos - qp.pos) / sys.integrator.dt #[#, 3] as qp.pos
    qp_ang = (qp_next.rot - qp.rot) / sys.integrator.dt #[#, 4] as qp.rot

    return StateDMP(
        y=jnp.concatenate(
            [qp.pos.flatten(),
             qp.rot.flatten()]),
        yd=jnp.concatenate(
            [qp_vel.flatten(),
             qp_ang.flatten()]),
        x=x,
    )
