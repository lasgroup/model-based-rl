import argparse
import os

import chex
import jax.numpy as jnp
import jax.random as jr
import optax
import wandb
from brax.training.replay_buffers import UniformSamplingQueue
from brax.training.types import Transition
from bsm.bayesian_regression import ProbabilisticEnsemble, ProbabilisticFSVGDEnsemble
from bsm.statistical_model.bnn_statistical_model import BNNStatisticalModel
from bsm.statistical_model.gp_statistical_model import GPStatisticalModel
from distrax import Normal
from jax.nn import swish
from mbpo.optimizers import SACOptimizer
from mbpo.systems.rewards.base_rewards import Reward, RewardParams
from wtc.utils import discrete_to_continuous_discounting
from wtc.wrappers.ih_switching_cost import IHSwitchCostWrapper, ConstantSwitchCost

from mbrl.model_based_agent import WtcPets, WtcMean, WtcOptimistic
from wtc.envs.reacher_dm_control import ReacherDMControl

log_wandb = True
ENTITY = 'trevenl'


def experiment(project_name: str = 'GPUSpeedTest',
               num_offline_samples: int = 100,
               sac_horizon: int = 100,
               deterministic_policy_for_data_collection: bool = False,
               seed: int = 42,
               num_episodes: int = 20,
               sac_steps: int = 1_000_000,
               min_bnn_steps: int = 1_000,
               max_bnn_steps: int = 50_000,
               linear_scheduler_steps: int = 20_000,
               first_episode_for_policy_training: int = -1,
               exploration: str = 'optimistic',  # Should be one of the ['optimistic', 'pets', 'mean'],
               reset_statistical_model: bool = True,
               regression_model: str = 'probabilistic_ensemble',
               max_time_factor: int = 30,
               beta_factor: float = 2.0,
               horizon: int = 100,
               transition_cost: float = 0.1,
               ):
    assert exploration in ['optimistic', 'pets',
                           'mean'], "Unrecognized exploration strategy, should be 'optimistic' or 'pets' or 'mean'"
    assert regression_model in ['probabilistic_ensemble', 'FSVGD', 'GP']

    num_training_points = optax.linear_schedule(init_value=min_bnn_steps, end_value=max_bnn_steps,
                                                transition_steps=linear_scheduler_steps)
    config = dict(num_offline_samples=num_offline_samples,
                  sac_horizon=sac_horizon,
                  deterministic_policy_for_data_collection=deterministic_policy_for_data_collection,
                  seed=seed,
                  num_episodes=num_episodes,
                  sac_steps=sac_steps,
                  min_bnn_steps=min_bnn_steps,
                  max_bnn_steps=max_bnn_steps,
                  linear_scheduler_steps=linear_scheduler_steps,
                  first_episode_for_policy_training=first_episode_for_policy_training,
                  exploration=exploration,
                  reset_statistical_model=reset_statistical_model,
                  regression_model=regression_model,
                  max_time_factor=max_time_factor,
                  beta_factor=beta_factor,
                  horizon=horizon,
                  transition_cost=transition_cost
                  )

    base_env = ReacherDMControl(backend='generalized')

    min_time_between_switches = 1 * base_env.dt
    max_time_between_switches = max_time_factor * base_env.dt

    running_reward_max_bound = 50.0 + 5  # We add some margin
    running_reward_min_bound = -2 - 1  # We add some margin

    env = IHSwitchCostWrapper(base_env,
                              num_integrator_steps=horizon,
                              min_time_between_switches=min_time_between_switches,
                              max_time_between_switches=max_time_between_switches,
                              switch_cost=ConstantSwitchCost(value=jnp.array(0.0)),
                              time_as_part_of_state=True)

    episode_time = base_env.dt * horizon

    class TransitionReward(Reward):
        def __init__(self):
            super().__init__(x_dim=7, u_dim=2)

        def __call__(self,
                     x: chex.Array,
                     u: chex.Array,
                     reward_params: RewardParams,
                     x_next: chex.Array | None = None
                     ):
            reward = jnp.array(-transition_cost)
            reward_dist = Normal(reward, jnp.zeros_like(reward))
            return reward_dist, reward_params

        def init_params(self, key: chex.PRNGKey) -> RewardParams:
            return {'dt': 0.02}

    key_offline_data, key_agent = jr.split(jr.PRNGKey(seed))

    if num_offline_samples == 0:
        offline_data = None
    else:
        raise NotImplementedError('Offline data not implemented yet.')

    if regression_model == 'probabilistic_ensemble':
        model = BNNStatisticalModel(
            input_dim=env.observation_size + env.action_size - 1,  # -1 since we don't input env_time
            output_dim=env.observation_size + 1 - 1,  # +1 for the reward -1 for env time
            num_training_steps=num_training_points,
            output_stds=1e-3 * jnp.ones(env.observation_size + 1 - 1),  # +1 for the reward -1 for env_time
            beta=beta_factor * jnp.ones(shape=(env.observation_size + 1 - 1,)),
            features=(64, 64, 64),
            bnn_type=ProbabilisticEnsemble,
            num_particles=10,
            logging_wandb=log_wandb,
            return_best_model=True,
            eval_batch_size=256,
            eval_frequency=500,
            weight_decay=0.0,
            logging_frequency=100,
        )
    elif regression_model == 'FSVGD':
        model = BNNStatisticalModel(
            input_dim=env.observation_size + env.action_size - 1,  # -1 since we don't input env_time
            output_dim=env.observation_size + 1 - 1,  # +1 for the reward -1 for env time
            num_training_steps=num_training_points,
            output_stds=1e-3 * jnp.ones(env.observation_size + 1 - 1),  # +1 for the reward -1 for env_time
            beta=beta_factor * jnp.ones(shape=(env.observation_size + 1 - 1,)),
            features=(64, 64, 64),
            bnn_type=ProbabilisticFSVGDEnsemble,
            num_particles=5,
            logging_wandb=log_wandb,
            return_best_model=True,
            eval_batch_size=256,
            eval_frequency=10,
            weight_decay=0.0,
            logging_frequency=100,

        )
    elif regression_model == 'GP':
        model = GPStatisticalModel(
            input_dim=env.observation_size + env.action_size - 1,  # -1 since we don't input env_time
            output_dim=env.observation_size + 1 - 1,  # +1 for the reward -1 for env time
            output_stds=1e-3 * jnp.ones(env.observation_size + 1 - 1),  # +1 for the reward -1 for env_time
            f_norm_bound=1.0,
            delta=0.1,
            num_training_steps=1000,
            logging_frequency=100,
        )

    discount_factor = 0.99
    continuous_discounting = discrete_to_continuous_discounting(discrete_discounting=discount_factor,
                                                                dt=env.dt)

    num_envs = 128
    num_env_steps_between_updates = 5
    sac_kwargs = {
        'num_timesteps': sac_steps,
        'episode_length': sac_horizon,
        'num_env_steps_between_updates': num_env_steps_between_updates,
        'num_envs': num_envs,
        'num_eval_envs': num_envs,
        'lr_alpha': 3e-4,
        'lr_policy': 3e-4,
        'lr_q': 3e-4,
        'wd_alpha': 0.,
        'wd_policy': 0.,
        'wd_q': 0.,
        'max_grad_norm': 1e5,
        'discounting': 0.99,
        'batch_size': 64,
        'num_evals': 20,
        'normalize_observations': True,
        'reward_scaling': 10.,
        'tau': 0.005,
        'min_replay_size': 10 ** 3,
        'max_replay_size': sac_steps,
        'grad_updates_per_step': num_envs * num_env_steps_between_updates,
        'deterministic_eval': True,
        'init_log_alpha': 0.,
        'policy_hidden_layer_sizes': (64, 64,),
        'policy_activation': swish,
        'critic_hidden_layer_sizes': (64, 64,),
        'critic_activation': swish,
        'wandb_logging': log_wandb,
        'return_best_model': True,
        'non_equidistant_time': True,
        'continuous_discounting': continuous_discounting,
        'min_time_between_switches': min_time_between_switches,
        'max_time_between_switches': max_time_between_switches,
        'env_dt': env.dt,
    }
    max_replay_size_true_data_buffer = num_episodes * horizon
    dummy_sample = Transition(observation=jnp.ones(env.observation_size),
                              action=jnp.zeros(shape=(env.action_size,)),
                              reward=jnp.array(0.0),
                              discount=jnp.array(discount_factor),
                              next_observation=jnp.ones(env.observation_size))

    sac_buffer = UniformSamplingQueue(
        max_replay_size=max_replay_size_true_data_buffer,
        dummy_data_sample=dummy_sample,
        sample_batch_size=1)

    optimizer = SACOptimizer(system=None,
                             true_buffer=sac_buffer,
                             **sac_kwargs)
    if log_wandb:
        wandb.init(project=project_name,
                   dir='/cluster/scratch/' + ENTITY,
                   config=config)

    agent_class = None
    if exploration == 'optimistic':
        agent_class = WtcOptimistic
    elif exploration == 'mean':
        agent_class = WtcMean
    elif exploration == 'pets':
        agent_class = WtcPets

    agent = agent_class(
        env=env,
        eval_env=env,
        statistical_model=model,
        optimizer=optimizer,
        reward_model=TransitionReward(),
        episode_length=horizon,
        offline_data=offline_data,
        num_envs=1,
        num_eval_envs=1,
        log_to_wandb=log_wandb,
        dt=env.dt,
        min_time_between_switches=min_time_between_switches,
        max_time_between_switches=max_time_between_switches,
        episode_time=episode_time,
        deterministic_policy_for_data_collection=deterministic_policy_for_data_collection,
        running_reward_max_bound=running_reward_max_bound,
        running_reward_min_bound=running_reward_min_bound,
        first_episode_for_policy_training=first_episode_for_policy_training,
        reset_statistical_model=reset_statistical_model,
        max_collected_data_in_buffer=max_replay_size_true_data_buffer,
        save_trajectory_transitions=True
    )

    agent_state = agent.run_episodes(num_episodes=num_episodes,
                                     start_from_scratch=True,
                                     key=key_agent)

    wandb.finish()


def main(args):
    experiment(project_name=args.project_name,
               num_offline_samples=args.num_offline_samples,
               sac_horizon=args.sac_horizon,
               deterministic_policy_for_data_collection=bool(args.deterministic_policy_for_data_collection),
               seed=args.seed,
               num_episodes=args.num_episodes,
               sac_steps=args.sac_steps,
               min_bnn_steps=args.min_bnn_steps,
               max_bnn_steps=args.max_bnn_steps,
               linear_scheduler_steps=args.linear_scheduler_steps,
               first_episode_for_policy_training=args.first_episode_for_policy_training,
               exploration=args.exploration,
               reset_statistical_model=bool(args.reset_statistical_model),
               regression_model=args.regression_model,
               max_time_factor=args.max_time_factor,
               beta_factor=args.beta_factor,
               horizon=args.horizon,
               transition_cost=args.transition_cost,
               )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='Model_based_pets')
    parser.add_argument('--num_offline_samples', type=int, default=0)
    parser.add_argument('--sac_horizon', type=int, default=100)
    parser.add_argument('--deterministic_policy_for_data_collection', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_episodes', type=int, default=50)
    parser.add_argument('--sac_steps', type=int, default=20_000)
    parser.add_argument('--min_bnn_steps', type=int, default=5_000)
    parser.add_argument('--max_bnn_steps', type=int, default=50_000)
    parser.add_argument('--linear_scheduler_steps', type=int, default=20_000)
    parser.add_argument('--first_episode_for_policy_training', type=int, default=0)
    parser.add_argument('--exploration', type=str, default='optimistic')
    parser.add_argument('--reset_statistical_model', type=int, default=0)
    parser.add_argument('--regression_model', type=str, default='FSVGD')
    parser.add_argument('--max_time_factor', type=int, default=10)
    parser.add_argument('--beta_factor', type=float, default=2.0)
    parser.add_argument('--horizon', type=int, default=200)
    parser.add_argument('--transition_cost', type=float, default=0.5)

    args = parser.parse_args()
    main(args)
