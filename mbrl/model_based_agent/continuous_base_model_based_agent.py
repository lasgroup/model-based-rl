from abc import ABC, abstractmethod
from functools import partial
from typing import Tuple
import os
import pickle

import chex
import jax.numpy as jnp
import jax.random as jr
import wandb
from brax.envs import Env as BraxEnv
from brax.training.replay_buffers import UniformSamplingQueue, ReplayBufferState
from brax.training.types import Transition
from bsm.statistical_model import StatisticalModel
from bsm.utils import StatisticalModelState
from bsm.utils.normalization import Data
from jax import jit
from mbpo.optimizers.base_optimizer import BaseOptimizer
from mbpo.systems.rewards.base_rewards import Reward
from mbpo.utils.type_aliases import OptimizerState

from .base_model_based_agent import BaseModelBasedAgent

from mbrl.model_based_agent.optimizer_wrapper import Actor
from mbrl.utils.brax_utils import EnvInteractor

class ContinuousBaseModelBasedAgent(BaseModelBasedAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (self.predict_difference == False), f'self.predict_difference must be False for continuous agent, got: {self.predict_difference}.'

    def _collected_buffer_to_train_data(self, collected_buffer_state: ReplayBufferState):
        idx = jnp.arange(start=collected_buffer_state.sample_position, stop=collected_buffer_state.insert_position)
        all_data = jnp.take(collected_buffer_state.data, idx, axis=0, mode='wrap')
        all_transitions: Transition = self.collected_data_buffer._unflatten_fn(all_data)
        obs = all_transitions.observation
        actions = all_transitions.action
        inputs = jnp.concatenate([obs, actions], axis=-1)
        derivatives = all_transitions.extras['state_extras']['derivative']
        if self.predict_difference:
            raise NotImplementedError
        else:
            outputs = derivatives
        return Data(inputs=inputs, outputs=outputs)