import argparse

import chex
import jax.numpy as jnp
import jax.random as jr
import wandb
from brax.training.replay_buffers import UniformSamplingQueue
from brax.training.types import Transition
from bsm.bayesian_regression import DeterministicEnsemble
from bsm.statistical_model.bnn_statistical_model import BNNStatisticalModel
from distrax import Normal
from jax.nn import swish
from mbpo.optimizers import SACOptimizer
from mbpo.systems.rewards.base_rewards import Reward, RewardParams
from wtc.utils import discrete_to_continuous_discounting
from wtc.wrappers.ih_switching_cost import IHSwitchCostWrapper, ConstantSwitchCost

from mbrl.envs.pendulum import PendulumEnv
from mbrl.model_based_agent import WtcPets, WtcMean, WtcOptimistic
from mbrl.utils.offline_data import WhenToControlWrapper

log_wandb = True
ENTITY = 'trevenl'


def experiment(project_name: str = 'GPUSpeedTest',
               num_offline_samples: int = 100,
               sac_horizon: int = 100,
               deterministic_policy_for_data_collection: bool = False,
               seed: int = 42,
               num_episodes: int = 20,
               sac_steps: int = 1_000_000,
               bnn_steps: int = 5_000,
               first_episode_for_policy_training: int = -1,
               exploration: str = 'optimistic',  # Should be one of the ['optimistic', 'pets', 'mean'],
               reset_statistical_model: bool = True
               ):
    assert exploration in ['optimistic', 'pets',
                           'mean'], "Unrecognized exploration strategy, should be 'optimistic' or 'pets' or 'mean'"
    config = dict(num_offline_samples=num_offline_samples,
                  sac_horizon=sac_horizon,
                  deterministic_policy_for_data_collection=deterministic_policy_for_data_collection,
                  seed=seed,
                  num_episodes=num_episodes,
                  sac_steps=sac_steps,
                  bnn_steps=bnn_steps,
                  first_episode_for_policy_training=first_episode_for_policy_training,
                  exploration=exploration,
                  reset_statistical_model=reset_statistical_model
                  )

    base_env = PendulumEnv(reward_source='dm-control')

    min_time_between_switches = 1 * base_env.dt
    max_time_between_switches = 30 * base_env.dt
    num_integrator_steps = 100
    switch_cost = 0.1

    running_reward_max_bound = 20.0
    running_reward_min_bound = -5

    env = IHSwitchCostWrapper(base_env,
                              num_integrator_steps=num_integrator_steps,
                              min_time_between_switches=min_time_between_switches,
                              max_time_between_switches=max_time_between_switches,
                              switch_cost=ConstantSwitchCost(value=jnp.array(0.0)),
                              time_as_part_of_state=True)

    episode_time = base_env.dt * num_integrator_steps

    class TransitionReward(Reward):
        def __init__(self):
            super().__init__(x_dim=2, u_dim=1)

        def __call__(self,
                     x: chex.Array,
                     u: chex.Array,
                     reward_params: RewardParams,
                     x_next: chex.Array | None = None
                     ):
            reward = jnp.array(-switch_cost)
            reward_dist = Normal(reward, jnp.zeros_like(reward))
            return reward_dist, reward_params

        def init_params(self, key: chex.PRNGKey) -> RewardParams:
            return {'dt': 0.05}

    offline_data_gen = WhenToControlWrapper(
        num_integrator_steps=num_integrator_steps,
        min_time_between_switches=min_time_between_switches,
        max_time_between_switches=max_time_between_switches
    )

    key_offline_data, key_agent = jr.split(jr.PRNGKey(seed))

    offline_data = offline_data_gen.sample_transitions(key=key_offline_data,
                                                       num_samples=num_offline_samples)

    horizon = 100
    model = BNNStatisticalModel(
        input_dim=env.observation_size + env.action_size - 1,  # -1 since we don't input env_time
        output_dim=env.observation_size + 1 - 1,  # +1 for the reward -1 for env time
        num_training_steps=bnn_steps,
        output_stds=1e-3 * jnp.ones(env.observation_size + 1 - 1),  # +1 for the reward -1 for env_time
        beta=2.0 * jnp.ones(shape=(env.observation_size + 1 - 1,)),
        features=(64,) * 3,
        bnn_type=DeterministicEnsemble,
        num_particles=5,
        logging_wandb=False,
        return_best_model=True,
        eval_batch_size=64,
        train_share=0.8,
        eval_frequency=500,
        weight_decay=0.0,
    )

    discount_factor = 0.99
    continuous_discounting = discrete_to_continuous_discounting(discrete_discounting=discount_factor,
                                                                dt=env.dt)
    sac_kwargs = {
        'num_timesteps': sac_steps,
        'episode_length': sac_horizon,
        'num_env_steps_between_updates': 10,
        'num_envs': 64,
        'num_eval_envs': 4,
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
        'reward_scaling': 1.,
        'tau': 0.005,
        'min_replay_size': 10 ** 3,
        'max_replay_size': 10 ** 5,
        'grad_updates_per_step': 10 * 64,  # should be num_envs * num_env_steps_between_updates
        'deterministic_eval': True,
        'init_log_alpha': 0.,
        'policy_hidden_layer_sizes': (32,) * 5,
        'policy_activation': swish,
        'critic_hidden_layer_sizes': (128,) * 3,
        'critic_activation': swish,
        'wandb_logging': log_wandb,
        'return_best_model': True,
        'non_equidistant_time': True,
        'continuous_discounting': continuous_discounting,
        'min_time_between_switches': min_time_between_switches,
        'max_time_between_switches': max_time_between_switches,
        'env_dt': env.dt,
    }
    max_replay_size_true_data_buffer = 10 ** 4
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
               bnn_steps=args.bnn_steps,
               first_episode_for_policy_training=args.first_episode_for_policy_training,
               exploration=args.exploration,
               reset_statistical_model=bool(args.reset_statistical_model),
               )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='Model_based_pets')
    parser.add_argument('--num_offline_samples', type=int, default=200)
    parser.add_argument('--sac_horizon', type=int, default=100)
    parser.add_argument('--deterministic_policy_for_data_collection', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_episodes', type=int, default=5)
    parser.add_argument('--sac_steps', type=int, default=20_000)
    parser.add_argument('--bnn_steps', type=int, default=5_000)
    parser.add_argument('--first_episode_for_policy_training', type=int, default=2)
    parser.add_argument('--exploration', type=str, default='mean')
    parser.add_argument('--reset_statistical_model', type=int, default=0)

    args = parser.parse_args()
    main(args)
