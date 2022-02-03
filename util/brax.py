import jax
from jax import numpy as jnp

import brax
from brax.envs import env

from util.types import *

from functools import partial


def get_n_dmp(env: env.Env):
    dummy_key = jax.random.PRNGKey(0)
    dummy_state = env.reset(dummy_key)
    qp = dummy_state.qp
    if jnp.ndim(qp.pos) < 3:
        qp = jax.tree_map(lambda x: jnp.expand_dims(x, 0), qp)
    dummy_state_dmp = brax_qp_to_dmp_state(env.sys, qp)
    n_dmp = dummy_state_dmp.y.shape[-1]
    return n_dmp


def brax_qp_to_dmp_state(sys: brax.physics.system.System,
                         qp: brax.physics.base.QP, x: float = 1.) -> StateDMP:
    """Estimates changes in qp.pos and qp.rot under Euler integration
    """
    # jitting the integrator is cruical, see https://github.com/google/brax/issues/102
    qp_next = jax.jit(jax.vmap(sys.integrator.kinetic))(qp) # assuming batched qp
    qp_vel = (qp_next.pos - qp.pos) / sys.integrator.dt #[#, 3] as qp.pos
    qp_ang = (qp_next.rot - qp.rot) / sys.integrator.dt #[#, 4] as qp.rot

    batch_size = qp.pos.shape[0]

    return StateDMP(
        y=jnp.concatenate(
            [qp.pos.reshape([batch_size, -1]),
             qp.rot.reshape([batch_size, -1])], axis=-1),
        yd=jnp.concatenate(
            [qp_vel.reshape([batch_size, -1]),
             qp_ang.reshape([batch_size, -1])], axis=-1),
        x=x,
    )
