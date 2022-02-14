from typing import Dict
from functools import partial

import jax
from jax import numpy as jnp
import brax
from brax.envs import env
from util.types import PRNGKey, Params
from yacs.config import CfgNode

from ndp.dmp import DMP
from ndp.phi import PhiNet
from ndp.omega import OmegaNet
from util.types import *
from util.brax import Qp2Dmp

import warnings


class NDP(object):

    def __init__(
            self,
            cfg: CfgNode,
            env: env.Env,
            target_action_size: int,
            timestep: float,
    ):
        n_dmp = None
        self.qp2dmp = Qp2Dmp(env)
        if cfg.DMP.INFER_STATE:
            n_dmp = cfg.DMP.N_DMP
        else:
            n_dmp = qp2dmp.get_n_dmp()
            if cfg.DMP.N_DMP != n_dmp:
                warnings.warn(
                    "cfg.DMP.N_DMP is incorrect; raised due to cfg.DMP.INFER_STATE=True. "
                    "For {} environment, set it to {}.".format(cfg.ENV.ENV_NAME, n_dmp)
                )

        self.dmp = DMP(cfg, timestep, n_dmp)
        self.phi_net = PhiNet(cfg, env, self.qp2dmp, n_dmp)
        self.omega_net = OmegaNet(cfg, target_action_size, n_dmp)


    def init(self, key: PRNGKey) -> Dict[str, Params]:
        key_phi, key_omega = jax.random.split(key, 2)
        return {
            'phi': self.phi_net.init(key_phi),
            'omega': self.omega_net.init(key_omega),
        }


    @partial(jax.jit, static_argnums=(0,))
    def apply(
            self,
            params: Dict[str, Params],
            qps: brax.physics.base.QP,
            observations: jnp.ndarray, # these might be normalized
    ) -> jnp.ndarray:
        """ Takes the parameters of phi and omega, + a batch of
            observations as input, spits out the next
            `unroll_length` actions.
        """
        phi_params, omega_params = params['phi'], params['omega']
        dmp_params = self.phi_net.apply(phi_params, qps, observations)
        dmp_states = self.dmp.do_dmp_unroll(dmp_params)
        actions = self.omega_net.apply(omega_params, dmp_states)
        return actions
