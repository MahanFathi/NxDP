import jax
from jax import numpy as jnp

import brax
from brax.envs import env

from util.types import *


def get_n_dmp(env: env.Env):
    dummy_key = jax.random.PRNGKey(0)
    dummy_state = env.reset(dummy_key)
    dummy_state_dmp = self.brax_state_to_dmp_state(env.sys, dummy_state)
    n_dmp = dummy_state_dmp.y.shape[-1]
    return n_dmp
    if cfg.DMP.N_DMP != n_dmp:
        warnings.warn(
            "cfg.DMP.N_DMP is incorrect; raised due to cfg.DMP.INFER_STATE=True."
        )


def brax_state_to_dmp_state(sys: brax.physics.system.System,
                            state: env.State, x: float = 1.) -> StateDMP:
    """Estimates changes in qp.pos and qp.rot under Euler integration
    """
    qp = state.qp
    qp_next = sys.integrator.kinetic(qp)
    qp_vel = (qp_next.pos - qp.pos) / self.brax_sys.integrator.dt #[#, 3] as qp.pos
    qp_ang = (qp_next.rot - qp.rot) / self.brax_sys.integrator.dt #[#, 4] as qp.rot

    batch_size = qp.pos.shape[0]

    return StateDMP(
        y=jnp.concatenate(
            [qp.pos.reshape(batch_size, -1),
             qp_vel.reshape(batch_size, -1)], axis=-1)
        yd=jnp.concatenate(
            [qp.rot.reshape(batch_size, -1),
             qp_ang.reshape(batch_size, -1)], axis=-1)
        x=x,
    )
