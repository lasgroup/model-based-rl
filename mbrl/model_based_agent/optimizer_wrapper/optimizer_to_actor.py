from abc import ABC, abstractmethod
from typing import Tuple

import chex
from brax.training.replay_buffers import ReplayBufferState
from mbpo.optimizers.base_optimizer import BaseOptimizer
from mbpo.systems.base_systems import System
from mbpo.systems.dynamics.base_dynamics import DynamicsParams
from mbpo.systems.rewards.base_rewards import RewardParams
from mbpo.utils.type_aliases import OptimizerState, OptimizerTrainingOutPut


class Actor(ABC):
    def __init__(self,
                 env_observation_size: int,
                 env_action_size: int,
                 optimizer: BaseOptimizer):
        self.optimizer = optimizer
        self.env_observation_size = env_observation_size
        self.env_action_size = env_action_size

    def set_system(self, system: System):
        self.optimizer.set_system(system)

    @abstractmethod
    def act(self,
            obs: chex.Array,
            opt_state: OptimizerState[RewardParams, DynamicsParams],
            evaluate: bool = True
            ) -> Tuple[chex.Array, OptimizerState]:
        pass

    def optimizer_act(self,
                      obs: chex.Array,
                      opt_state: OptimizerState[RewardParams, DynamicsParams],
                      evaluate: bool = True) -> Tuple[chex.Array, OptimizerState]:
        return self.optimizer.act(obs, opt_state, evaluate)

    def train(self,
              opt_state: OptimizerState[RewardParams, DynamicsParams]) -> OptimizerTrainingOutPut[
        RewardParams, DynamicsParams]:
        return self.optimizer.train(opt_state)

    def init(self,
             key: chex.PRNGKey,
             true_buffer_state: ReplayBufferState | None = None) -> OptimizerState:
        return self.optimizer.init(key, true_buffer_state)

    @property
    def can_act_in_batches(self):
        return self.optimizer.can_act_in_batches

    def dummy_true_buffer_state(self,
                                key: chex.Array) -> ReplayBufferState:
        return self.optimizer.dummy_true_buffer_state(key)

    @property
    def system(self):
        return self.optimizer.system


class PetsActor(Actor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def act(self,
            obs: chex.Array,
            opt_state: OptimizerState[RewardParams, DynamicsParams],
            evaluate: bool = True
            ) -> Tuple[chex.Array, OptimizerState]:
        return self.optimizer.act(obs, opt_state, evaluate)


class MeanActor(Actor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def act(self,
            obs: chex.Array,
            opt_state: OptimizerState[RewardParams, DynamicsParams],
            evaluate: bool = True
            ) -> Tuple[chex.Array, OptimizerState]:
        return self.optimizer.act(obs, opt_state, evaluate)


class OptimisticActor(Actor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def act(self,
            obs: chex.Array,
            opt_state: OptimizerState[RewardParams, DynamicsParams],
            evaluate: bool = True
            ) -> Tuple[chex.Array, OptimizerState]:
        # Return only the part of the augmented_action that takes care of action (without etas)
        augmented_action, opt_state = self.optimizer.act(obs, opt_state)
        return augmented_action[..., :self.env_action_size], opt_state
