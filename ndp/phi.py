from functools import partial

import jax
from jax import numpy as jnp
from brax.envs import env
from yacs.config import CfgNode

from util.net import make_model
from util.types import *
from util.brax import Qp2Dmp

import warnings


class PhiNet(object):

    def __init__(self, cfg: CfgNode, env: env.Env, qp2dmp: Qp2Dmp = None, n_dmp: int = None):

        self.qp2dmp = qp2dmp

        # figure out n_dmp
        self.dmp_state_is_inferred = cfg.DMP.INFER_STATE
        self.n_dmp = n_dmp if n_dmp else cfg.DMP.N_DMP

        self.n_bfs = cfg.DMP.N_BFS
        output_size = (
            self.n_bfs + # each dmp has n_bfs parameters (w) for each basis function
            1 + # goal
            2 * self.dmp_state_is_inferred # y, yd
        ) * self.n_dmp
        self._phi_net = make_model(
            cfg.PHI_NET.FEATURES + [output_size],
            env.observation_size,
        )


    def init(self, key: PRNGKey):
        return self._phi_net.init(key)


    @partial(jax.jit, static_argnums=(0,))
    def apply(self, params: Params, qp: brax.physics.base.QP, observations: jnp.ndarray) -> ParamsDMP:
        dmp_params = self._phi_net.apply(params, observations)
        dmp_params = jnp.reshape(
            dmp_params,
            [dmp_params.shape[0], self.n_dmp, -1]
        )

        if self.dmp_state_is_inferred:
            state_dmp = StateDMP(
                y=dmp_params[:, :, -2],  # (batch_size, n_dmp)
                yd=dmp_params[:, :, -1], # (batch_size, n_dmp)
                x=1.0,
            )
        else:
            state_dmp = self.qp2dmp(qp)

        inferred_state_index = -2 * self.dmp_state_is_inferred
        return ParamsDMP(
            w=dmp_params[:, :, :(inferred_state_index - 1)], # (batch_size, n_dmp, n_bfs)
            g=dmp_params[:, :,  (inferred_state_index - 1)], # (batch_size, n_dmp)
            s=state_dmp,
        )
