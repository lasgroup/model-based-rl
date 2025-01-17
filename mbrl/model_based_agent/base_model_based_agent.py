from abc import ABC, abstractmethod
from functools import partial
from typing import Tuple, Sequence
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

from mbrl.model_based_agent.optimizer_wrapper import Actor
from mbrl.utils.brax_utils import EnvInteractor

from datetime import datetime


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
                 predict_difference: bool = False,
                 reset_statistical_model: bool = True,
                 key: chex.PRNGKey = jr.PRNGKey(0),
                 log_to_wandb: bool = False,
                 deterministic_policy_for_data_collection: bool = False,
                 first_episode_for_policy_training: int = -1,
                 save_trajectory_transitions: bool = False,
                 dt: float = 0.05,
                 state_extras_ref: dict = {},
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
        self.first_episode_for_policy_training = first_episode_for_policy_training
        self.save_trajectory_transitions = save_trajectory_transitions
        self.dt = dt
        self.state_extras_ref = state_extras_ref

        self.key, subkey = jr.split(self.key)
        self.env_interactor = EnvInteractor(
            env=self.env,
            eval_env=self.eval_env,
            num_envs=self.num_envs,
            num_eval_envs=self.num_eval_envs,
            episode_length=self.episode_length,
            action_repeat=self.action_repeat,
            key=subkey,
            deterministic_policy_for_data_collection=deterministic_policy_for_data_collection,
            extra_fields=list(self.state_extras_ref.keys()))

        self.collected_data_buffer = self.prepare_data_buffers()
        self.actor = self.prepare_actor(optimizer)

    def prepare_data_buffers(self) -> UniformSamplingQueue:
        dummy_sample = Transition(observation=jnp.zeros(shape=(self.env.observation_size,)),
                                  action=jnp.zeros(shape=(self.env.action_size,)),
                                  reward=jnp.array(0.0),
                                  discount=jnp.array(0.99),
                                  next_observation=jnp.zeros(shape=(self.env.observation_size,)),
                                  extras={'state_extras': self.state_extras_ref}
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
        all_transitions: Transition = self.collected_data_buffer._unflatten_fn(all_data)
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
                             ) -> Tuple[ModelBasedAgentState, Transition]:
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
        new_agent_state = ModelBasedAgentState(optimizer_state=optimizer_state, env_steps=env_steps, key=key_agent)
        return new_agent_state, transitions # TODO: Remove new_agent_state (Debugging)

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
            if episode_idx >= self.first_episode_for_policy_training:
                print(f'Start of policy training')
                agent_state = self.train_policy(agent_state=agent_state,
                                                episode_idx=episode_idx)
                print(f'End of policy training')
        # We collect new data with the current policy
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Start with data collection") # TODO: Remove line (Debugging)
        print(f'Start of data collection')
        agent_state, trajectory_transitions = self.simulate_on_true_env(agent_state=agent_state)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Finished simulating on true env") # TODO: Remove line (Debugging)
        if self.save_trajectory_transitions and self.log_to_wandb:
            directory = os.path.join(wandb.run.dir, 'results')
            if not os.path.exists(directory):
                os.makedirs(directory)
            model_path = os.path.join(directory, f'episode_{episode_idx}_trajectory.pkl')
            with open(model_path, 'wb') as handle:
                pickle.dump(trajectory_transitions, handle)
            wandb.save(model_path, wandb.run.dir)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - End of data collection") # TODO: Remove line (Debugging)
        print(f'End of data collection')
        print(f'Start with evaluation of the policy')
        metrics, data = self.env_interactor.run_evaluation(actor=self.actor,
                                                     actor_state=agent_state.optimizer_state)
        
        # Log epistemic uncertainty
        statistical_model_state = agent_state.optimizer_state.system_params.dynamics_params.statistical_model_state # obviously bizarre, fix
        output = self.statistical_model.predict_batch(self.states, statistical_model_state=statistical_model_state)
        epistemic_magnitude = jnp.sqrt(jnp.sum(output.epistemic_std**2, axis=1))
        augmented_epistemic_std = jnp.hstack([output.epistemic_std, epistemic_magnitude[:, None]])

        ep_uncert_metrics = {}
        for prefix, fn in [('mean', jnp.mean),('max', jnp.max),('min', jnp.min)]:
            ep_uncert_metrics.update(
                {
                    f'{prefix}_ep_uncert/episode_uncert_dim_{dim}': (
                        value
                    )
                    for dim, value in enumerate(fn(augmented_epistemic_std, axis=0))
                }
            )
        ## Until here

        if self.log_to_wandb:
            wandb.log(ep_uncert_metrics | metrics | {'episode_idx': episode_idx})
        else:
            print(ep_uncert_metrics | metrics)
        print(f'End with evaluation of the policy')
        return agent_state

    def run_episodes(self,
                     num_episodes: int,
                     start_from_scratch: bool = True,
                     key: chex.PRNGKey = jr.PRNGKey(0),
                     agent_state: ModelBasedAgentState | None = None) -> ModelBasedAgentState:
        if start_from_scratch:
            # If we start collecting the data and need to initialize the agent state
            key, subkey = jr.split(key)

            # num_tests = 100_000
            # self.states = jr.uniform(key=subkey, shape=(num_tests, 3,), minval=jnp.array([0.,-1,-1]), maxval=jnp.array([2*jnp.pi,1,1]))
            # self.states = jnp.stack([jnp.cos(self.states[:,0]), 
            #                          jnp.sin(self.states[:,0]), 
            #                          self.env.dynamics_params.max_speed*self.states[:,1],
            #                          self.env.dynamics_params.max_torque*self.states[:,2]], axis=-1)

            num_tests = 10_000
            self.states = jr.uniform(key=subkey, shape=(num_tests, 5,), minval=jnp.array([-1,0.,-1,-1,-1]), maxval=jnp.array([-1,2*jnp.pi,1,1,1]))
            self.states = jnp.stack([10*self.env.dynamics_params.max_lin_speed*self.states[:,0],
                                     jnp.cos(self.states[:,1]), 
                                     jnp.sin(self.states[:,1]), 
                                     self.env.dynamics_params.max_lin_speed*self.states[:,2],
                                     self.env.dynamics_params.max_ang_speed*self.states[:,3],
                                     self.env.dynamics_params.max_force*self.states[:,4]], axis=-1)
            agent_state = self.init(key)
        for episode_idx in range(num_episodes):
            print(f'Starting with Episode {episode_idx}')
            agent_state = self.do_episode(agent_state=agent_state,
                                          episode_idx=episode_idx)
            print(f'End of Episode {episode_idx}')
        return agent_state
