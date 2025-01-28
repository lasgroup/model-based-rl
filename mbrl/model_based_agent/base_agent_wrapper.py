from mbrl.model_based_agent.base_model_based_agent import BaseModelBasedAgent, ModelBasedAgentState
from mbrl.model_based_agent.continuous_base_model_based_agent import ContinuousBaseModelBasedAgent
from mbrl.model_based_agent.optimizer_wrapper import Actor, MeanActor
from mbrl.model_based_agent.system_wrapper import MeanSystem, MeanDynamics, ContinuousMeanSystem, ContinuousMeanDynamics
from mbpo.optimizers.base_optimizer import BaseOptimizer
from mbpo.optimizers.policy_optimizers.brax_optimizers import BraxOptimizer
from mbpo.systems.rewards.base_rewards import Reward
from mbpo.utils.type_aliases import OptimizerState
from typing import Type, List, Tuple
from datetime import datetime

import wandb
import copy
import chex
from brax.envs import Env as BraxEnv
import jax.random as jr

from brax.envs import Env as BraxEnv
from mbrl.utils.brax_utils import EnvInteractor


class MultiEnvEvaluatorWrapper:
    def __init__(self, 
                 agent: ContinuousBaseModelBasedAgent, 
                 eval_envs: List[BraxEnv],
                 reward_model_list: List[Reward], 
                 eval_frequency: int = 1):
        self.agent = agent

        self.num_rewards = len(reward_model_list)
        assert self.num_rewards > 0, 'Need at least one reward function'
        assert len(eval_envs) == self.num_rewards, 'Need as many eval envs as reward functions'
        self.reward_model_list = reward_model_list
        self.eval_frequency = eval_frequency
        self.eval_envs = eval_envs
        # self.evaluators = self.prepare_evaluators(eval_envs)
        self.env_interactors = self.prepare_env_interactors(eval_envs)
        # self.key, actors_key, self.agent.key = jr.split(self.agent.key, 3) TODO: Maybe also adjust agent key
        self.key, actors_key = jr.split(self.agent.key, 2)
        self.actors_and_opt_states = self.prepare_actors_for_reward_models(optimizer=self.agent.actor.optimizer, key=actors_key)
    
    def prepare_evaluators(self, eval_envs: List[BraxEnv]) -> List[Tuple[EnvInteractor, Actor]]:
        """
        Prepares evaluators for multiple evaluation environments.
        """
        evaluators = []
        dynamics_type, system_type, actor_type = ContinuousMeanDynamics, ContinuousMeanSystem, MeanActor

        for eval_env in eval_envs:
            # Clone the statistical model for each environment
            dynamics = dynamics_type(
                statistical_model=self.agent.statistical_model,
                x_dim=self.agent.env.observation_size,
                u_dim=self.agent.env.action_size,
                predict_difference=self.agent.predict_difference,
                dt=self.agent.dt,
            )
            system = system_type(dynamics=dynamics, reward=None) # TODO: self.agent.reward_model)

            actor = actor_type(
                env_observation_size=self.agent.env.observation_size,
                env_action_size=self.agent.env.action_size,
                optimizer=self.agent.actor.optimizer  # Use the same optimizer configuration
            )
            actor.set_system(system=system)

            # Prepare EnvInteractor for each evaluation environment
            subkey, self.agent.key = jr.split(self.agent.key, 2)
            env_interactor = EnvInteractor(
                env=self.agent.env,
                eval_env=eval_env,
                num_envs=self.agent.num_envs,
                num_eval_envs=self.agent.num_eval_envs,
                episode_length=self.agent.episode_length,
                action_repeat=self.agent.action_repeat,
                key=subkey,
                deterministic_policy_for_data_collection=True,
            )
            evaluators.append((env_interactor, actor))

        return evaluators

    def prepare_env_interactors(self, eval_envs: List[BraxEnv]) -> List[EnvInteractor]:
        env_interactors = []
        for eval_env in eval_envs:
            subkey, self.key = jr.split(self.key, 2)
            env = copy.deepcopy(self.agent.env)
            interactor = EnvInteractor(
                env=env,
                eval_env=eval_env,
                num_envs=self.agent.num_envs,
                num_eval_envs=self.agent.num_eval_envs,
                episode_length=self.agent.episode_length,
                action_repeat=self.agent.action_repeat,
                key=subkey,
                deterministic_policy_for_data_collection=True)
            env_interactors.append(interactor)
        return env_interactors

    def prepare_actors_for_reward_models(self, optimizer: BaseOptimizer, key: chex.Array) -> List[Tuple[Actor,
    OptimizerState]]:
        actors_and_opt_states = []
        dynamics_type, system_type, actor_type = ContinuousMeanDynamics, ContinuousMeanSystem, MeanActor
        for reward_model in self.reward_model_list:
            optimizer_new = copy.deepcopy(optimizer)
            if isinstance(optimizer_new, BraxOptimizer):
                optimizer_new.agent_kwargs['wandb_logging'] = False
            model = copy.deepcopy(self.agent.statistical_model)
            dynamics = dynamics_type(statistical_model=model,
                                     x_dim=self.agent.env.observation_size,
                                     u_dim=self.agent.env.action_size,
                                     predict_difference=self.agent.predict_difference)
            system = system_type(dynamics=dynamics,
                                 reward=reward_model, )
            actor = actor_type(env_observation_size=self.agent.env.observation_size,
                               env_action_size=self.agent.env.action_size,
                               optimizer=optimizer_new)
            actor.set_system(system=system)
            key, key_data_buffers, key_optimizer = jr.split(key, 3)
            collected_data_buffer_state = self.agent._init_data_buffer_states(key_data_buffers)
            init_optimizer_state = actor.init(key=key_optimizer,
                                              true_buffer_state=collected_data_buffer_state)
            actors_and_opt_states.append((actor, init_optimizer_state))
        return actors_and_opt_states

    @staticmethod
    def train_single_actor(actor: Actor, optimizer_state: OptimizerState):
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
            if self.agent.log_to_wandb and isinstance(actor.optimizer, BraxOptimizer):
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
                   episode_idx: int):
        """
        Extend the agent's do_episode method to include evaluation across multiple environments/tasks.
        """
        agent_state = self.agent.do_episode(agent_state=agent_state,
                                            episode_idx=episode_idx)

        #if episode_idx % self.eval_frequency == 0:
        #    # Evaluate using each evaluator
        #    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Evaluating on downstream tasks")
        #    for i, (env_interactor, actor) in enumerate(self.evaluators):
        #        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Evaluating on task {i}")
        #        metrics, _ = env_interactor.run_evaluation(actor=actor, actor_state=agent_state.optimizer_state)
        #        metrics = {f"{k}_env_{i}": v for k, v in metrics.items()}
        #        if self.agent.log_to_wandb:
        #            wandb.log(metrics | {"episode_idx": episode_idx})
        #        else:
        #            print(metrics)
        if episode_idx % self.eval_frequency == 0:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Evaluating on downstream tasks")
            actors_for_reward_models = self.train_reward_policies(actors_for_reward_models=actors_for_reward_models,
                                                                  agent_state=agent_state,
                                                                  episode_idx=episode_idx,
                                                                  )
            for i in range(self.num_rewards):
                env_interactor = self.env_interactors[i]
                actor, opt_state = actors_for_reward_models[i]
                metrics, data = env_interactor.run_evaluation(actor=actor,
                                                        actor_state=opt_state)
                metrics = {k + '_task_' + str(i): v for k, v in metrics.items()}
                if self.agent.log_to_wandb:
                    wandb.log(metrics | {'episode_idx': episode_idx})
                else:
                    print(metrics)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - End with evaluating on downstream tasks")

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


    def __getattr__(self, name):
        # Delegate attribute access to the wrapped agent
        return getattr(self.agent, name)


class BaseAgentWrapper(BaseModelBasedAgent):
    def __init__(self, agent_type: Type[BaseModelBasedAgent],
                 **kwargs):
        # Get the args and kwargs from the agent
        self.agent = agent_type(**kwargs)
        super().__init__(**kwargs)

    # Only change the prepare_actor method (since this is what varies between the different agents)
    def prepare_actor(self, optimizer: BaseOptimizer) -> Actor:
        return self.agent.prepare_actor(optimizer) 