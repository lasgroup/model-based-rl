from .base_model_based_agent import ModelBasedAgentState
from .continuous_base_model_based_agent import ContinuousBaseModelBasedAgent
from mbpo.systems.rewards.base_rewards import Reward
from mbpo.optimizers.base_optimizer import BaseOptimizer
from mbpo.optimizers.policy_optimizers.brax_optimizers import BraxOptimizer
from mbrl.model_based_agent.optimizer_wrapper import Actor, OptimisticActor, PetsActor, MeanActor
from mbrl.model_based_agent.system_wrapper import ContinuousOptimisticExplorationSystem, ContinuousOptimisticExplorationDynamics, \
    ContinuousExplorationReward, ContinuousPetsSystem, ContinuousPetsDynamics, ContinuousPetsExplorationDynamics, ContinuousPetsExplorationSystem, \
    ContinuousMeanExplorationDynamics, ContinuousMeanExplorationSystem
from mbpo.utils.type_aliases import OptimizerState
import chex
import jax.random as jr
import jax.numpy as jnp
from typing import List, Tuple
import wandb
from brax.envs import Env as BraxEnv
from mbrl.utils.brax_utils import EnvInteractor
import copy


class ContinuousPetsActiveExplorationModelBasedAgent(ContinuousBaseModelBasedAgent):
    def __init__(self,
                 env: BraxEnv,
                 eval_envs: List[BraxEnv],
                 reward_model_list: List[Reward], optimizer: BaseOptimizer, eval_frequency: int = 1,
                 *args,
                 **kwargs):
        super().__init__(
            reward_model=ContinuousExplorationReward(x_dim=env.observation_size, u_dim=env.action_size),
            optimizer=optimizer,
            env=env,
            eval_env=env,
            *args,
            **kwargs,
        )
        self.num_rewards = len(reward_model_list)
        assert self.num_rewards > 0, 'Need at least one reward function'
        assert len(eval_envs) == self.num_rewards, 'Need as many eval envs as reward functions'
        self.reward_model_list = reward_model_list
        self.eval_frequency = eval_frequency
        self.env_interactors = self.prepare_env_interactors(eval_envs)
        actors_key, self.key = jr.split(self.key, 2)
        self.actors_and_opt_states = self.prepare_actors_for_reward_models(optimizer=optimizer, key=actors_key)

    def prepare_actor(self,
                      optimizer: BaseOptimizer,
                      ) -> Actor:
        dynamics, system, actor = ContinuousPetsExplorationDynamics, ContinuousPetsExplorationSystem, PetsActor
        dynamics = dynamics(statistical_model=self.statistical_model,
                            x_dim=self.env.observation_size,
                            u_dim=self.env.action_size,
                            predict_difference=self.predict_difference)
        system = system(dynamics=dynamics,
                        reward=self.reward_model, )
        actor = actor(env_observation_size=self.env.observation_size,
                      env_action_size=self.env.action_size,
                      optimizer=optimizer)
        actor.set_system(system=system)
        return actor

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
        dynamics_type, system_type, actor_type = ContinuousPetsDynamics, ContinuousPetsSystem, PetsActor
        for reward_model in self.reward_model_list:
            optimizer_new = copy.deepcopy(optimizer)
            if isinstance(optimizer_new, BraxOptimizer):
                optimizer_new.agent_kwargs['wandb_logging'] = False
            model = copy.deepcopy(self.statistical_model)
            dynamics = dynamics_type(statistical_model=model,
                                     x_dim=self.env.observation_size,
                                     u_dim=self.env.action_size,
                                     predict_difference=self.predict_difference)
            system = system_type(dynamics=dynamics,
                                 reward=reward_model, )
            actor = actor_type(env_observation_size=self.env.observation_size,
                               env_action_size=self.env.action_size,
                               optimizer=optimizer_new)
            actor.set_system(system=system)
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
                   episode_idx: int) -> Tuple[ModelBasedAgentState, List[Tuple[Actor, OptimizerState]]]:
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
        agent_state, trajectory_transitions = self.simulate_on_true_env(agent_state=agent_state)
        print(f'End of data collection')
        print(f'Start with evaluation of the policy')
        if episode_idx % self.eval_frequency == 0:
            print(f'Start training of evaluation policy')
            actors_for_reward_models = self.train_reward_policies(actors_for_reward_models=actors_for_reward_models,
                                                                  agent_state=agent_state,
                                                                  episode_idx=episode_idx,
                                                                  )
            
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
            
            if self.log_to_wandb:
                wandb.log(ep_uncert_metrics | {'episode_idx': episode_idx})
            else:
                print(ep_uncert_metrics)
            ## Until here

            for i in range(self.num_rewards):
                env_interactor = self.env_interactors[i]
                actor, opt_state = actors_for_reward_models[i]
                metrics, data = env_interactor.run_evaluation(actor=actor,
                                                        actor_state=opt_state)
                metrics = {k + '_task_' + str(i): v for k, v in metrics.items()}
                if self.log_to_wandb:
                    wandb.log(metrics | {'episode_idx': episode_idx})
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
            key, subkey = jr.split(key)

            num_tests = 100_000
            self.states = jr.uniform(key=subkey, shape=(num_tests, 5,), minval=jnp.array([-1,0.,-1,-1,-1]), maxval=jnp.array([-1,2*jnp.pi,1,1,1]))
            self.states = jnp.stack([10*self.env.dynamics_params.max_lin_speed*self.states[:,0],
                                     jnp.cos(self.states[:,1]), 
                                     jnp.sin(self.states[:,1]), 
                                     self.env.dynamics_params.max_lin_speed*self.states[:,2],
                                     self.env.dynamics_params.max_ang_speed*self.states[:,3],
                                     self.env.dynamics_params.max_force*self.states[:,4]], axis=-1)
            agent_state = self.init(key)
            actors_for_reward_models = self.actors_and_opt_states
        for episode_idx in range(num_episodes):
            print(f'Starting with Episode {episode_idx}')
            agent_state, actors_for_reward_models = self.do_episode(agent_state=agent_state,
                                                                    actors_for_reward_models=actors_for_reward_models,
                                                                    episode_idx=episode_idx)
            print(f'End of Episode {episode_idx}')
        return agent_state, actors_for_reward_models


class ContinuousOptimisticActiveExplorationModelBasedAgent(ContinuousPetsActiveExplorationModelBasedAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_actor(self,
                      optimizer: BaseOptimizer,
                      ) -> Actor:
        dynamics, system, actor = ContinuousOptimisticExplorationDynamics, ContinuousOptimisticExplorationSystem, OptimisticActor
        dynamics = dynamics(statistical_model=self.statistical_model,
                            x_dim=self.env.observation_size,
                            u_dim=self.env.action_size,
                            predict_difference=self.predict_difference)
        system = system(dynamics=dynamics,
                        reward=self.reward_model, )
        actor = actor(env_observation_size=self.env.observation_size,
                      env_action_size=self.env.action_size,
                      optimizer=optimizer)
        actor.set_system(system=system)
        return actor


class ContinuousMeanActiveExplorationModelBasedAgent(ContinuousPetsActiveExplorationModelBasedAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_actor(self,
                      optimizer: BaseOptimizer,
                      ) -> Actor:
        dynamics, system, actor = ContinuousMeanExplorationDynamics, ContinuousMeanExplorationSystem, MeanActor
        dynamics = dynamics(statistical_model=self.statistical_model,
                            x_dim=self.env.observation_size,
                            u_dim=self.env.action_size,
                            predict_difference=self.predict_difference)
        system = system(dynamics=dynamics,
                        reward=self.reward_model, )
        actor = actor(env_observation_size=self.env.observation_size,
                      env_action_size=self.env.action_size,
                      optimizer=optimizer)
        actor.set_system(system=system)
        return actor


if __name__ == "__main__":
    from mbrl.envs.pendulum import PendulumEnv
    from bsm.statistical_model.bnn_statistical_model import BNNStatisticalModel
    from mbpo.optimizers import SACOptimizer
    from distrax import Normal
    from mbrl.utils.offline_data import PendulumOfflineData

    import chex
    import jax.numpy as jnp
    import jax.random as jr
    from brax.training.replay_buffers import UniformSamplingQueue
    from brax.training.types import Transition
    from jax.nn import swish
    from mbpo.optimizers.base_optimizer import BaseOptimizer
    from mbpo.systems.rewards.base_rewards import Reward, RewardParams

    from mbrl.model_based_agent.optimizer_wrapper import Actor

    log_wandb = False
    env = PendulumEnv(reward_source='dm-control')
    swing_down_params = env.reward_params.replace(target_angle=jnp.pi)
    swing_down_env = PendulumEnv(reward_source='dm-control')
    swing_down_env.reward_params = swing_down_params


    class PendulumRewardSwingUp(Reward):
        def __init__(self):
            super().__init__(x_dim=2, u_dim=1)

        def __call__(self,
                     x: chex.Array,
                     u: chex.Array,
                     reward_params: RewardParams,
                     x_next: chex.Array | None = None
                     ):
            assert x.shape == (3,) and u.shape == (1,)
            theta, omega = jnp.arctan2(x[1], x[0]), x[-1]
            target_angle = env.reward_params.target_angle
            diff_th = theta - target_angle
            diff_th = ((diff_th + jnp.pi) % (2 * jnp.pi)) - jnp.pi
            reward = env.tolerance_reward(jnp.sqrt(env.reward_params.angle_cost * diff_th ** 2 +
                                                   0.1 * omega ** 2)) - env.reward_params.control_cost * u ** 2
            reward = reward.squeeze()
            reward_dist = Normal(reward, jnp.zeros_like(reward))
            return reward_dist, reward_params

        def init_params(self, key: chex.PRNGKey) -> RewardParams:
            return {'dt': env.dt}


    class PendulumRewardSwingDown(Reward):
        def __init__(self):
            super().__init__(x_dim=2, u_dim=1)

        def __call__(self,
                     x: chex.Array,
                     u: chex.Array,
                     reward_params: RewardParams,
                     x_next: chex.Array | None = None
                     ):
            assert x.shape == (3,) and u.shape == (1,)
            theta, omega = jnp.arctan2(x[1], x[0]), x[-1]
            target_angle = swing_down_env.reward_params.target_angle + jnp.pi
            diff_th = theta - target_angle
            diff_th = ((diff_th + jnp.pi) % (2 * jnp.pi)) - jnp.pi
            reward = swing_down_env.tolerance_reward(jnp.sqrt(swing_down_env.reward_params.angle_cost * diff_th ** 2 +
                                                              0.1 * omega ** 2)) - \
                     swing_down_env.reward_params.control_cost * u ** 2
            reward = reward.squeeze()
            reward_dist = Normal(reward, jnp.zeros_like(reward))
            return reward_dist, reward_params

        def init_params(self, key: chex.PRNGKey) -> RewardParams:
            return {'dt': swing_down_env.dt}


    offline_data_gen = PendulumOfflineData()
    key = jr.PRNGKey(0)

    offline_data = offline_data_gen.sample_transitions(key=key,
                                                       num_samples=100)

    horizon = 200
    model = BNNStatisticalModel(
        input_dim=env.observation_size + env.action_size,
        output_dim=env.observation_size,
        num_training_steps=3_000,
        output_stds=1e-3 * jnp.ones(env.observation_size),
        features=(64, 64, 64),
        num_particles=5,
        logging_wandb=log_wandb,
        return_best_model=True,
        eval_batch_size=64,
        train_share=0.8,
        eval_frequency=1_000,
    )

    sac_kwargs = {
        'num_timesteps': 20_000,
        'episode_length': 64,
        'num_env_steps_between_updates': 10,
        'num_envs': 16,
        'num_eval_envs': 4,
        'lr_alpha': 3e-4,
        'lr_policy': 3e-4,
        'lr_q': 3e-4,
        'wd_alpha': 0.,
        'wd_policy': 0.,
        'wd_q': 0.,
        'max_grad_norm': 1e5,
        'discounting': 0.99,
        'batch_size': 32,
        'num_evals': 20,
        'normalize_observations': True,
        'reward_scaling': 1.,
        'tau': 0.005,
        'min_replay_size': 10 ** 4,
        'max_replay_size': 10 ** 5,
        'grad_updates_per_step': 10 * 16,  # should be num_envs * num_env_steps_between_updates
        'deterministic_eval': True,
        'init_log_alpha': 0.,
        'policy_hidden_layer_sizes': (64, 64),
        'policy_activation': swish,
        'critic_hidden_layer_sizes': (64, 64),
        'critic_activation': swish,
        'wandb_logging': log_wandb,
        'return_best_model': True,
    }
    max_replay_size_true_data_buffer = 10 ** 4
    dummy_sample = Transition(observation=jnp.ones(env.observation_size),
                              action=jnp.zeros(shape=(env.action_size,)),
                              reward=jnp.array(0.0),
                              discount=jnp.array(0.99),
                              next_observation=jnp.ones(env.observation_size))

    sac_buffer = UniformSamplingQueue(
        max_replay_size=max_replay_size_true_data_buffer,
        dummy_data_sample=dummy_sample,
        sample_batch_size=1)

    optimizer = SACOptimizer(system=None,
                             true_buffer=sac_buffer,
                             **sac_kwargs)

    agent = ContinuousPetsActiveExplorationModelBasedAgent(
        env=env,
        eval_envs=[env, swing_down_env],
        statistical_model=model,
        optimizer=optimizer,
        reward_model_list=[PendulumRewardSwingUp(), PendulumRewardSwingDown()],
        episode_length=horizon,
        offline_data=offline_data,
        num_envs=1,
        num_eval_envs=1,
        log_to_wandb=log_wandb,
    )

    agent_state = agent.run_episodes(num_episodes=20,
                                     start_from_scratch=True,
                                     key=jr.PRNGKey(0))
