from abc import abstractmethod

import wandb
import os
import pickle
import copy
from typing import Tuple, List
import chex
import jax.random as jr
import jax.numpy as jnp
from brax.envs import Env as BraxEnv
from brax.training.replay_buffers import ReplayBufferState
from bsm.utils.normalization import Data
from mbpo.optimizers.base_optimizer import BaseOptimizer
from mbpo.optimizers.policy_optimizers.brax_optimizers import BraxOptimizer
from mbpo.utils.type_aliases import OptimizerState
from mbpo.systems.base_systems import System, Dynamics
from mbpo.systems.rewards.base_rewards import Reward


from mbrl.model_based_agent.base_model_based_agent import BaseModelBasedAgent
from mbrl.model_based_agent.optimizer_wrapper import Actor, PetsActor
from mbrl.model_based_agent.system_wrapper import WtsScMeanDynamics, WtcScMeanSystem
from mbrl.utils.brax_utils import EnvInteractor


@chex.dataclass
class ModelBasedAgentState:
    optimizer_state: OptimizerState
    env_steps: chex.Array
    key: chex.Array


class WtcBaseModelBasedAgent(BaseModelBasedAgent):

    def __init__(self,
                 dt: float,
                 min_time_between_switches: float,
                 max_time_between_switches: float,
                 episode_time: float,
                 *args,
                 running_reward_max_bound: float = 1e5,
                 running_reward_min_bound: float = -1e5,
                 eval_envs: List[BraxEnv],
                 reward_model_list: List[Reward], 
                 optimizer: BaseOptimizer, 
                 eval_frequency: int = 1,
                 **kwargs):

        self.running_reward_max_bound = running_reward_max_bound
        self.running_reward_min_bound = running_reward_min_bound
        self.dt = dt
        self.min_time_between_switches = min_time_between_switches
        self.max_time_between_switches = max_time_between_switches
        self.episode_time = episode_time
        super().__init__(*args, **kwargs, optimizer=optimizer)
        self.num_rewards = len(reward_model_list)
        assert self.num_rewards > 0, 'Need at least one reward function'
        assert len(eval_envs) == self.num_rewards, 'Need as many eval envs as reward functions'
        self.reward_model_list = reward_model_list
        self.eval_frequency = eval_frequency
        self.env_interactors = self.prepare_env_interactors(eval_envs)
        actors_key, self.key = jr.split(self.key, 2)
        self.actors_and_opt_states = self.prepare_actors_for_reward_models(optimizer=optimizer, key=actors_key)

    @abstractmethod
    def prepare_actor(self,
                      optimizer: BaseOptimizer,
                      ) -> Actor:
        pass

    def prepare_wtc_actor(self,
                          optimizer: BaseOptimizer,
                          dynamics,
                          system,
                          actor,
                          ) -> Actor:
        dynamics = dynamics(statistical_model=self.statistical_model,
                            running_reward_max_bound=self.running_reward_max_bound,
                            running_reward_min_bound=self.running_reward_min_bound,
                            x_dim=self.env.observation_size,
                            u_dim=self.env.action_size,
                            min_time_between_switches=self.min_time_between_switches,
                            max_time_between_switches=self.max_time_between_switches,
                            episode_time=self.episode_time,
                            dt=self.dt)
        system = system(dynamics=dynamics,
                        reward=self.reward_model, )
        actor = actor(env_observation_size=self.env.observation_size,
                      env_action_size=self.env.action_size,
                      optimizer=optimizer)
        actor.set_system(system=system)
        return actor

    @staticmethod
    def compute_time(pseudo_time: chex.Array,
                     dt: float,
                     t_min: float,
                     t_max: float,
                     env_time: chex.Array,
                     time_horizon: float
                     ) -> chex.Array:
        time_for_action = ((t_max - t_min) / 2 * pseudo_time + (t_max + t_min) / 2)
        return jnp.minimum((time_for_action // dt) * dt, time_horizon - env_time)

    def _collected_buffer_to_train_data(self,
                                        collected_buffer_state: ReplayBufferState):
        idx = jnp.arange(start=collected_buffer_state.sample_position, stop=collected_buffer_state.insert_position)
        all_data = jnp.take(collected_buffer_state.data, idx, axis=0, mode='wrap')
        all_transitions = self.collected_data_buffer._unflatten_fn(all_data)
        # obs = [env_state, env_times]
        obs = all_transitions.observation
        # action = [env_action, time_to_control]
        actions = all_transitions.action

        env_states, env_times = obs[..., :-1], obs[..., 1]
        env_actions, pseudo_times_for_action = actions[..., :-1], actions[..., 1]
        rewards = all_transitions.reward.reshape(-1, 1)  # This should be only integrated reward

        times_for_action = self.compute_time(pseudo_times_for_action,
                                             dt=self.dt,
                                             t_min=self.min_time_between_switches,
                                             t_max=self.max_time_between_switches,
                                             env_time=env_times,
                                             time_horizon=self.episode_time)

        inputs = jnp.concatenate([env_states, env_actions, times_for_action[..., None]], axis=-1)
        env_states_next = all_transitions.next_observation[..., :-1]  # We remove time_to_go
        if self.predict_difference:
            target = env_states_next - env_states
        else:
            target = env_states_next
        outputs = jnp.concatenate([target, rewards], axis=-1)  # append the integrated reward to the output
        return Data(inputs=inputs, outputs=outputs)

    def prepare_env_interactors(self, eval_envs: List[BraxEnv]):
        env_interactors = []
        for eval_env in eval_envs:
            subkey, self.key = jr.split(self.key, 2)
            env = copy.deepcopy(self.env)
            interactor = EnvInteractor(
                env=env,
                eval_env=eval_env,
                num_envs=self.num_envs,
                num_eval_envs=self.num_eval_envs,
                episode_length=self.episode_length,
                action_repeat=self.action_repeat,
                key=subkey,
                deterministic_policy_for_data_collection=True)
            env_interactors.append(interactor)
        return env_interactors

    def prepare_actors_for_reward_models(self, optimizer: BaseOptimizer, key: chex.Array) -> List[Tuple[Actor,
    OptimizerState]]:
        actors_and_opt_states = []
        dynamics_type, system_type, actor_type = WtsScMeanDynamics, WtcScMeanSystem, PetsActor
        for reward_model in self.reward_model_list:
            optimizer_new = copy.deepcopy(optimizer)
            if isinstance(optimizer_new, BraxOptimizer):
                optimizer_new.agent_kwargs['wandb_logging'] = False
            model = copy.deepcopy(self.statistical_model)
            actor = self.prepare_wtc_actor(optimizer=optimizer_new,
                                      dynamics=dynamics_type,
                                      system=system_type,
                                      actor=actor_type)
            key, key_data_buffers, key_optimizer = jr.split(key, 3)
            collected_data_buffer_state = self._init_data_buffer_states(key_data_buffers)
            init_optimizer_state = actor.init(key=key_optimizer,
                                              true_buffer_state=collected_data_buffer_state)
            actors_and_opt_states.append((actor, init_optimizer_state))
        return actors_and_opt_states

    @staticmethod
    def train_single_actor(actor, optimizer_state):
        # TODO: here we always start training the optimizer from scratch, we might want to just continue training after
        #  the data buffer surpasses certain margin
        training_output = actor.train(opt_state=optimizer_state)
        return training_output

    def train_reward_policies(self,
                              actors_for_reward_models: List[Tuple[Actor, OptimizerState]],
                              agent_state: ModelBasedAgentState,
                              episode_idx: int) -> List[Tuple[Actor, OptimizerState]]:
        actors_for_reward_models = self.update_model_state_for_reward_optimizers(actors_for_reward_models, agent_state)
        training_outputs = [self.train_single_actor(actor, optimizer_state)
                                    for actor, optimizer_state in actors_for_reward_models]
        for actor_idx, (actor, _) in enumerate(actors_for_reward_models):
            training_output = training_outputs[actor_idx]
            new_optimizer_state, summaries = training_output.optimizer_state, \
                training_output.summary
            actors_for_reward_models[actor_idx] = (actor, new_optimizer_state)
            # log the output of the policy training in hindsight
            if self.log_to_wandb and isinstance(actor.optimizer, BraxOptimizer):
                for summary in summaries:
                    summary = {k + '_task_' + str(actor_idx): v for k, v in summary.items()}
                    wandb.log(summary)
        return actors_for_reward_models

    @staticmethod
    def update_model_state_for_reward_optimizers(actors_for_reward_models: List[Tuple[Actor, OptimizerState]],
                                                 agent_state: ModelBasedAgentState,
                                                 ):
        statistical_model_state = agent_state.optimizer_state.system_params.dynamics_params.statistical_model_state
        buffer_state = agent_state.optimizer_state.true_buffer_state
        for actor_idx, (actor, actor_opt_state) in enumerate(actors_for_reward_models):
            new_dynamics_params = actor_opt_state.system_params.dynamics_params.replace(
                statistical_model_state=statistical_model_state)
            new_system_params = actor_opt_state.system_params.replace(
                dynamics_params=new_dynamics_params)
            new_optimizer_state = actor_opt_state.replace(system_params=new_system_params,
                                                          true_buffer_state=buffer_state)
            actors_for_reward_models[actor_idx] = (actor, new_optimizer_state)
        return actors_for_reward_models

    
    def do_episode(self,
                   agent_state: ModelBasedAgentState,
                   actors_for_reward_models: List[Tuple[Actor, OptimizerState]],
                   episode_idx: int,
                   ) -> Tuple[ModelBasedAgentState, List[Tuple[Actor, OptimizerState]]]:
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
        print(f'Start of data collection')
        agent_state, trajectory_transitions = self.simulate_on_true_env(agent_state=agent_state)
        if self.save_trajectory_transitions and self.log_to_wandb:
            directory = os.path.join(wandb.run.dir, 'results')
            if not os.path.exists(directory):
                os.makedirs(directory)
            model_path = os.path.join(directory, f'episode_{episode_idx}_trajectory.pkl')
            with open(model_path, 'wb') as handle:
                pickle.dump(trajectory_transitions, handle)
            wandb.save(model_path, wandb.run.dir)
        print(f'End of data collection')
        print(f'Start with evaluation of the policy')
        if episode_idx % self.eval_frequency == 0:
            print(f'Start training of evaluation policy')
            actors_for_reward_models = self.train_reward_policies(actors_for_reward_models=actors_for_reward_models,
                                                                  agent_state=agent_state,
                                                                  episode_idx=episode_idx,
                                                                  )
            for i in range(self.num_rewards):
                env_interactor = self.env_interactors[i]
                actor, opt_state = actors_for_reward_models[i]
                metrics = env_interactor.run_evaluation(actor=actor,
                                                        actor_state=opt_state)
                metrics = {k + '_task_' + str(i): v for k, v in metrics.items()}
                if self.log_to_wandb:
                    wandb.log(metrics)
                else:
                    print(metrics)
            print(f'End with evaluation of the policy')
        return agent_state, actors_for_reward_models


    def run_episodes(self,
                     num_episodes: int,
                     start_from_scratch: bool = True,
                     key: chex.PRNGKey = jr.PRNGKey(0),
                     agent_state: ModelBasedAgentState | None = None) -> \
            Tuple[ModelBasedAgentState, List[Tuple[Actor, OptimizerState]]]:
        if start_from_scratch:
            # If we start collecting the data and need to initialize the agent state
            agent_state = self.init(key)
            actors_for_reward_models = self.actors_and_opt_states
        for episode_idx in range(num_episodes):
            print(f'Starting with Episode {episode_idx}')
            agent_state, actors_for_reward_models = self.do_episode(agent_state=agent_state,
                                                                    actors_for_reward_models=actors_for_reward_models,
                                                                    episode_idx=episode_idx)
            print(f'End of Episode {episode_idx}')
        return agent_state, actors_for_reward_models
    