from abc import ABC, abstractmethod
from functools import partial

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

from mbrl.model_based_agent.optimizer_wrapper import Actor
from mbrl.utils.brax_utils import EnvInteractor


@chex.dataclass
class ModelBasedAgentState:
    optimizer_state: OptimizerState
    env_steps: chex.Array
    key: chex.Array


class BaseModelBasedAgent(ABC):

    def __init__(self,
                 env: BraxEnv,
                 statistical_model: StatisticalModel,
                 reward_model: Reward,
                 optimizer: BaseOptimizer,
                 episode_length: int,
                 eval_env: BraxEnv,
                 num_envs: int = 1,
                 num_eval_envs: int = 1,
                 action_repeat: int = 1,
                 offline_data: Transition | None = None,
                 max_collected_data_in_buffer: int = 10 ** 4,
                 max_episodes: int = 100,
                 predict_difference: bool = True,
                 reset_statistical_model: bool = True,
                 key: chex.PRNGKey = jr.PRNGKey(0),
                 log_to_wandb: bool = False,
                 deterministic_policy_for_data_collection: bool = False,
                 ):
        self.env = env
        self.statistical_model = statistical_model
        self.reward_model = reward_model
        self.episode_length = episode_length
        self.eval_env = eval_env
        self.num_envs = num_envs
        self.num_eval_envs = num_eval_envs
        self.action_repeat = action_repeat
        self.offline_data = offline_data
        self.max_collected_data_in_buffer = max_collected_data_in_buffer
        self.max_episodes = max_episodes
        self.predict_difference = predict_difference
        self.reset_statistical_model = reset_statistical_model
        self.key = key
        self.log_to_wandb = log_to_wandb
        self.deterministic_policy_for_data_collection = deterministic_policy_for_data_collection

        self.key, subkey = jr.split(self.key)
        self.env_interactor = EnvInteractor(
            env=self.env,
            eval_env=self.eval_env,
            num_envs=self.num_envs,
            num_eval_envs=self.num_eval_envs,
            episode_length=self.episode_length,
            action_repeat=self.action_repeat,
            key=subkey,
            deterministic_policy_for_data_collection=deterministic_policy_for_data_collection)

        self.collected_data_buffer = self.prepare_data_buffers()
        self.actor = self.prepare_actor(optimizer)

    def prepare_data_buffers(self) -> UniformSamplingQueue:
        dummy_sample = Transition(observation=jnp.zeros(shape=(self.env.observation_size,)),
                                  action=jnp.zeros(shape=(self.env.action_size,)),
                                  reward=jnp.array(0.0),
                                  discount=jnp.array(0.99),
                                  next_observation=jnp.zeros(shape=(self.env.observation_size,)),
                                  )

        collected_data_buffer = UniformSamplingQueue(
            max_replay_size=self.max_collected_data_in_buffer,
            dummy_data_sample=dummy_sample,
            sample_batch_size=1)

        return collected_data_buffer

    @abstractmethod
    def prepare_actor(self,
                      optimizer: BaseOptimizer,
                      ) -> Actor:
        raise NotImplementedError

    def _init_data_buffer_states(self,
                                 key: chex.PRNGKey) -> ReplayBufferState:
        key_collected_data_bs, key_init_state_bs = jr.split(key)
        collected_data_buffer_state = self.collected_data_buffer.init(key_collected_data_bs)

        # If we have offline data we insert in the collected data buffer
        if self.offline_data:
            assert self.offline_data.observation.shape[-1] == self.env.observation_size
            assert self.offline_data.action.shape[-1] == self.env.action_size
            collected_data_buffer_state = self.collected_data_buffer.insert(collected_data_buffer_state,
                                                                            self.offline_data)

        return collected_data_buffer_state

    def init(self, key: chex.Array):
        key_state, key_data_buffers, key_optimizer = jr.split(key, 3)
        collected_data_buffer_state = self._init_data_buffer_states(key_data_buffers)
        init_optimizer_state = self.actor.init(key=key_optimizer,
                                               true_buffer_state=collected_data_buffer_state)
        return ModelBasedAgentState(optimizer_state=init_optimizer_state,
                                    env_steps=0,
                                    key=key_state)

    def train_policy(self,
                     agent_state: ModelBasedAgentState,
                     episode_idx: int) -> ModelBasedAgentState:
        optimizer_state = agent_state.optimizer_state
        # TODO: here we always start training the optimizer from scratch, we might want to just continue training after
        #  the data buffer surpasses certain margin
        optimizer_output = self.actor.train(opt_state=optimizer_state)
        new_agent_state = agent_state.replace(optimizer_state=optimizer_output.optimizer_state)
        return new_agent_state

    def train_dynamics_model(self,
                             agent_state: ModelBasedAgentState,
                             episode_idx: int, ) -> ModelBasedAgentState:
        statistical_model_state = agent_state.optimizer_state.system_params.dynamics_params.statistical_model_state
        collected_data_buffer_state = agent_state.optimizer_state.true_buffer_state
        key, key_sms = jr.split(agent_state.key)

        statistical_model_state = self._update_statistical_model(
            statistical_model_state=statistical_model_state,
            collected_data_buffer_state=collected_data_buffer_state,
            key=key_sms)

        new_dynamics_params = agent_state.optimizer_state.system_params.dynamics_params.replace(
            statistical_model_state=statistical_model_state)
        new_system_params = agent_state.optimizer_state.system_params.replace(
            dynamics_params=new_dynamics_params)
        new_optimizer_state = agent_state.optimizer_state.replace(system_params=new_system_params)
        new_agent_state = agent_state.replace(optimizer_state=new_optimizer_state,
                                              key=key)
        return new_agent_state

    def _collected_buffer_to_train_data(self,
                                        collected_buffer_state: ReplayBufferState):
        idx = jnp.arange(start=collected_buffer_state.sample_position, stop=collected_buffer_state.insert_position)
        all_data = jnp.take(collected_buffer_state.data, idx, axis=0, mode='wrap')
        all_transitions = self.collected_data_buffer._unflatten_fn(all_data)
        obs = all_transitions.observation
        actions = all_transitions.action
        inputs = jnp.concatenate([obs, actions], axis=-1)
        next_obs = all_transitions.next_observation
        if self.predict_difference:
            outputs = next_obs - obs
        else:
            outputs = next_obs
        return Data(inputs=inputs, outputs=outputs)

    def _update_statistical_model(self,
                                  statistical_model_state: StatisticalModelState,
                                  collected_data_buffer_state: ReplayBufferState,
                                  key: chex.PRNGKey):
        # We prepare data to train from the collected_data_buffer
        data = self._collected_buffer_to_train_data(collected_data_buffer_state)
        if self.reset_statistical_model:
            statistical_model_state = self.statistical_model.init(key=key)
        new_statistical_model_state = self.statistical_model.update(
            stats_model_state=statistical_model_state,
            data=data)
        return new_statistical_model_state

    @partial(jit, static_argnums=0)
    def simulate_on_true_env(self,
                             agent_state: ModelBasedAgentState,
                             ) -> ModelBasedAgentState:
        key_agent, key_reset = jr.split(agent_state.key)
        env_state = self.env_interactor.reset(key=key_reset)
        optimizer_state = agent_state.optimizer_state
        interaction = self.env_interactor.generate_rollouts(env_state=env_state,
                                                            actor_state=optimizer_state,
                                                            actor=self.actor
                                                            )
        final_state, optimizer_state, transitions = interaction
        collected_data_buffer_state = agent_state.optimizer_state.true_buffer_state
        collected_data_buffer_state = self.collected_data_buffer.insert(buffer_state=collected_data_buffer_state,
                                                                        samples=transitions)
        optimizer_state = optimizer_state.replace(true_buffer_state=collected_data_buffer_state)
        env_steps = agent_state.env_steps + self.num_envs * self.episode_length
        return ModelBasedAgentState(optimizer_state=optimizer_state,
                                    env_steps=env_steps,
                                    key=key_agent, )

    def do_episode(self,
                   agent_state: ModelBasedAgentState,
                   episode_idx: int,
                   ) -> ModelBasedAgentState:
        if episode_idx > 0 or self.offline_data:
            # If we collected some data already then we train dynamics model and the policy
            print(f'Start of dynamics training')
            agent_state = self.train_dynamics_model(agent_state=agent_state,
                                                    episode_idx=episode_idx)
            print(f'End of dynamics training')
            print(f'Start of policy training')
            agent_state = self.train_policy(agent_state=agent_state,
                                            episode_idx=episode_idx)
            print(f'End of policy training')
        # We collect new data with the current policy
        print(f'Start of data collection')
        agent_state = self.simulate_on_true_env(agent_state=agent_state)
        print(f'End of data collection')
        print(f'Start with evaluation of the policy')
        metrics = self.env_interactor.run_evaluation(actor=self.actor,
                                                     actor_state=agent_state.optimizer_state)
        if self.log_to_wandb:
            wandb.log(metrics)
        else:
            print(metrics)
        print(f'End with evaluation of the policy')
        return agent_state

    def run_episodes(self,
                     num_episodes: int,
                     start_from_scratch: bool = True,
                     key: chex.PRNGKey = jr.PRNGKey(0),
                     agent_state: ModelBasedAgentState | None = None) -> ModelBasedAgentState:
        if start_from_scratch:
            # If we start collecting the data and need to initialize the agent state
            agent_state = self.init(key)
        for episode_idx in range(num_episodes):
            print(f'Starting with Episode {episode_idx}')
            agent_state = self.do_episode(agent_state=agent_state,
                                          episode_idx=episode_idx)
            print(f'End of Episode {episode_idx}')
        return agent_state

