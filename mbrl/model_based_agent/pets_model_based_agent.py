from abc import ABC
from typing import Generic, Union
from gym import Env as GymEnv
from brax.envs import Env as BraxEnv
import jax.numpy as jnp
import chex
from bsm.statistical_model import StatisticalModel
from bsm.utils.type_aliases import ModelState
from mbpo.utils.type_aliases import OptimizerState
from mbpo.optimizers.base_optimizer import BaseOptimizer
import jax.random as jr
from brax.training.replay_buffers import UniformSamplingQueue, ReplayBufferState
from brax.training.types import Transition
import wandb
from mbrl.model_based_agent.system_wrapper import LearnedModelSystem, LearnedDynamics, SystemParams, DynamicsParams
from mbpo.systems.rewards.base_rewards import Reward, RewardParams
from bsm.utils.normalization import Data


@chex.dataclass
class ModelBasedAgentState(Generic[ModelState]):
    buffer_state: ReplayBufferState
    optimizer_state: OptimizerState[ModelState]


class PetsModelBasedAgent(ABC, Generic[ModelState, RewardParams]):
    def __init__(self,
                 env: Union[BraxEnv, GymEnv],
                 model: StatisticalModel[ModelState],
                 reward_model: Reward[RewardParams],
                 optimizer: BaseOptimizer[RewardParams],
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
        self.reward_model = reward_model
        self.optimizer = optimizer
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
            true_data_buffer_state = self.add_data_to_buffer(data=offline_data,
                                                             true_data_buffer_state=true_data_bf_state)
        else:
            true_data_buffer_state = true_data_bf_state

        self.init_states_buffer = UniformSamplingQueue(
            max_replay_size=max_replay_size_true_data_buffer,  # Should be larger than the number of episodes we run
            dummy_data_sample=self.dummy_sample,
            sample_batch_size=1)
        self.dynamics = LearnedDynamics(self.model)
        self.key, optimizer_init_key = jr.split(self.key, 2)
        self.system = LearnedModelSystem(
            dynamics=self.dynamics,
            reward=self.reward_model,
        )
        self.optimizer.set_system(system=self.system)
        init_optimizer_state = self.optimizer.init(optimizer_init_key)
        self.init_agent_state = ModelBasedAgentState(buffer_state=true_data_buffer_state,
                                                     optimizer_state=init_optimizer_state,
                                                     )

    @property
    def init_optimizer_state(self) -> OptimizerState:
        return self.init_agent_state.optimizer_state

    @property
    def init_system_params(self) -> SystemParams:
        return self.init_optimizer_state.system_params

    @property
    def init_dynamics_params(self) -> DynamicsParams:
        return self.init_system_params.dynamics_params

    @property
    def init_model_state(self) -> ModelState:
        return self.init_dynamics_params.model_state

    def add_data_to_buffer(self, data: Transition, agent_state: ModelBasedAgentState) -> ModelBasedAgentState:
        new_buffer_state = self.true_data_buffer.insert(agent_state.buffer_state, data)
        new_agent_state = agent_state.replace(buffer_state=new_buffer_state)
        return new_agent_state

    @staticmethod
    def define_wandb_metrics():
        wandb.define_metric('x_axis/episode')

    def get_data_for_model_training_from_buffer_state(self, buffer_state: ReplayBufferState):
        idx = jnp.arange(start=buffer_state.sample_position, stop=buffer_state.insert_position)
        all_data = jnp.take(buffer_state.data, idx, axis=0, mode='wrap')
        all_transitions = self.true_data_buffer._unflatten_fn(all_data)
        obs = all_transitions.observation
        actions = all_transitions.action
        inputs = jnp.concatenate([obs, actions], axis=-1)
        next_obs = all_transitions.next_observation
        if self.predict_difference:
            outputs = next_obs - obs
        else:
            outputs = next_obs
        return Data(inputs=inputs, outputs=outputs)

    def train_dynamics_model(self,
                             agent_state: ModelBasedAgentState,
                             ) -> ModelBasedAgentState:
        # Prepare data
        buffer_state = agent_state.buffer_state
        data = self.get_data_for_model_training_from_buffer_state(buffer_state)
        new_model_state = self.model.update(
            statistical_model_state=agent_state.optimizer_state.system_params.dynamics_params.model_state,
            data=data)
        new_dynamics_params = agent_state.optimizer_state.system_params.dynamics_params.replace(
            model_state=new_model_state)
        new_system_params = agent_state.optimizer_state.system_params.replace(
            dynamics_params=new_dynamics_params)
        new_optimizer_state = agent_state.optimizer_state.replace(system_params=new_system_params)
        new_agent_state = agent_state.replace(optimizer_state=new_optimizer_state)
        return new_agent_state

    def train_policy(self, agent_state: ModelBasedAgentState) -> ModelBasedAgentState:
        optimizer_state = agent_state.optimizer_state
        optimizer_state = optimizer_state.replace(true_buffer_state=agent_state.buffer_state)

        optimizer_output = self.optimizer.train(
            opt_state=optimizer_state
        )
        new_agent_state = agent_state.replace(optimizer_state=optimizer_output.optimizer_state)
        return new_agent_state
