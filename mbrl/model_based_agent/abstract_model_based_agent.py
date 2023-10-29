from abc import ABC, abstractmethod
from typing import Generic, Union
from gym import Env as GymEnv
from brax.envs import Env as BraxEnv
from jax import vmap
import jax.numpy as jnp
import chex
from bsm.statistical_model import StatisticalModel
from bsm.utils.type_aliases import ModelState
from mbpo.utils.type_aliases import OptimizerState, OptimizerTrainingOutPut
from mbpo.optimizers.base_optimizer import BaseOptimizer
import jax.random as jr
from brax.training.replay_buffers import UniformSamplingQueue, ReplayBufferState
from brax.training.types import Transition
import wandb


class ModelBasedAgent(ABC, Generic[ModelState, OptimizerState, OptimizerTrainingOutPut]):
    def __init__(self,
                 env: Union[BraxEnv, GymEnv],
                 model: StatisticalModel[ModelState],
                 optimizer: BaseOptimizer[OptimizerState, OptimizerTrainingOutPut],
                 offline_data: Transition = None,
                 predict_difference: bool = True,
                 reset_bnn: bool = True,
                 return_best_bnn: bool = True,
                 bnn_training_test_ratio: float = 0.2,
                 return_best_optimizer: bool = True,
                 discounting: float = 0.99,
                 max_replay_size_true_data_buffer: int = 10 ** 4,
                 key: chex.PRNGKey = jr.PRNGKey(0),
                 ):
        self.env = env
        self.model = model
        self.optimzier = optimizer
        self.bnn_training_test_ratio = bnn_training_test_ratio
        self.predict_difference = predict_difference
        self.return_best_optimizer = return_best_optimizer
        self.return_best_bnn = return_best_bnn
        self.reset_bnn = reset_bnn
        self.discounting = discounting
        self.key = key
        self.state_dim, self.action_dim = self.env.observation_space.shape[0], self.env.action_space.shape[0]

        dummy_obs = jnp.zeros(shape=(self.state_dim,))
        self.dummy_sample = Transition(observation=dummy_obs,
                                       action=jnp.zeros(shape=(self.action_dim,)),
                                       reward=jnp.array(0.0),
                                       discount=jnp.array(0.99),
                                       next_observation=dummy_obs)

        self.true_data_buffer = UniformSamplingQueue(
            max_replay_size=max_replay_size_true_data_buffer,
            dummy_data_sample=self.dummy_sample,
            sample_batch_size=1)

        # We now insert data into the true data buffer
        self.key, key_bf_init = jr.split(self.key)
        true_data_bf_state = self.true_data_buffer.init(key_bf_init)

        if offline_data:
            assert offline_data.observation.shape[-1] == self.state_dim and \
                   offline_data.action.shape[-1] == self.action_dim
            self.true_data_buffer_state = self.add_data_to_buffer(data=offline_data,
                                                                  true_data_buffer_state=true_data_bf_state)
        else:
            self.true_data_buffer_state = true_data_bf_state

        self.init_states_buffer = UniformSamplingQueue(
            max_replay_size=max_replay_size_true_data_buffer,  # Should be larger than the number of episodes we run
            dummy_data_sample=self.dummy_sample,
            sample_batch_size=1)

    def add_data_to_buffer(self, data: Transition, true_data_buffer_state: ReplayBufferState) -> ReplayBufferState:
        return self.true_data_buffer.insert(true_data_buffer_state, data)

    def define_wandb_metrics(self):
        wandb.define_metric('x_axis/episode')
