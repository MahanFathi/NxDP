from typing import Sequence
from functools import partial

import jax
from jax import numpy as jnp
from util.types import PRNGKey, Params
from yacs.config import CfgNode

from util.net import make_model
from util.types import *


class OmegaNet(object):

    def __init__(self, cfg: CfgNode, target_action_size: int, n_dmp: int = None):

        self.n_dmp = n_dmp if n_dmp else cfg.DMP.N_DMP
        self._omega_net = make_model(
            cfg.OMEGA_NET.FEATURES + [target_action_size],
            4 * self.n_dmp,
        )


    def init(self, key: PRNGKey):
        return self._omega_net.init(key)


    @staticmethod
    def concat_dmp_states(states: Sequence[StateDMP]):
        return jnp.concatenate(
            [
                states[0].y, states[0].yd,
                states[1].y, states[1].yd,
            ],
            axis=-1
        )


    @partial(jax.jit, static_argnums=(0,))
    def _apply_step(self, carry, dmp_state_next: StateDMP):
        params, dmp_state = carry
        x = self.concat_dmp_states([dmp_state, dmp_state_next])
        logits = self._omega_net.apply(params, x)
        return (params, dmp_state_next), logits


    @partial(jax.jit, static_argnums=(0,))
    def apply(self, params: Params, dmp_states: StateDMP) -> jnp.ndarray:
        """Calculate policy logits on consecutive pairs
        # TODO: this could be also calculated by taking the state you've landed
        on as the first input and try to derive the system to the next desired
        dmp state. Note that this does not contradict planning in the latent space.
        """
        first_param_state = StateDMP(
            y=dmp_states.y[0],
            yd=dmp_states.yd[0],
            x=dmp_states.x[0],
        )
        _, logits = jax.lax.scan(
            self._apply_step,
            (params, first_param_state),
            jax.tree_map(lambda x: x[1:], dmp_states),
        )
        return logits # (unroll_length, batch_size, target_action_size)
