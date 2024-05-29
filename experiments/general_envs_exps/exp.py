import argparse

import chex
import jax.numpy as jnp
import jax.random as jr
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
from flax import linen as nn

from mbrl.envs.pendulum import PendulumEnv
from mbrl.model_based_agent import PETSModelBasedAgent, OptimisticModelBasedAgent

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
               reset_statistical_model: bool = True,
               regression_model: str = 'probabilistic_ensemble',
               env_name: str = 'Pendulum'
               ):
    assert exploration in ['optimistic',
                           'pets'], "Unrecognized exploration strategy, should be 'optimistic' or 'pets' or 'mean'"
    assert regression_model in ['probabilistic_ensemble', 'FSVGD', 'GP']
    assert env_name in ['Pendulum', 'RCCar', 'Greenhouse', 'Reacher']

    config = dict(num_offline_samples=num_offline_samples,
                  sac_horizon=sac_horizon,
                  deterministic_policy_for_data_collection=deterministic_policy_for_data_collection,
                  seed=seed,
                  num_episodes=num_episodes,
                  sac_steps=sac_steps,
                  bnn_steps=bnn_steps,
                  first_episode_for_policy_training=first_episode_for_policy_training,
                  exploration=exploration,
                  reset_statistical_model=reset_statistical_model,
                  regression_model=regression_model,
                  env_name=env_name
                  )

    if env_name == 'Pendulum':
        env = PendulumEnv(reward_source='dm-control')

        class PendulumReward(Reward):
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

        reward_model = PendulumReward()
        horizon = 200

    elif env_name == 'RCCar':
        from wtc.envs.rccar import RCCar, RCCarEnvReward
        env = RCCar(margin_factor=20, dt=0.04)
        horizon = 100

        class RCCarReward(Reward):
            def __init__(self):
                super().__init__(x_dim=7, u_dim=2)
                self.reward_model = RCCarEnvReward(goal=jnp.array([0.0, 0.0, 0.0]),
                                                   ctrl_cost_weight=0.005,
                                                   encode_angle=True,
                                                   margin_factor=20)

            def __call__(self,
                         x: chex.Array,
                         u: chex.Array,
                         reward_params: RewardParams,
                         x_next: chex.Array | None = None
                         ):
                reward = self.reward_model.forward(obs=x,
                                                   action=u,
                                                   next_obs=x_next)
                reward_dist = Normal(reward, jnp.zeros_like(reward))
                return reward_dist, reward_params

            def init_params(self, key: chex.PRNGKey) -> RewardParams:
                return {}

        reward_model = RCCarReward()

    elif env_name == 'Greenhouse':
        from wtc.envs.greenhouse import GreenHouseEnv
        horizon = 200

        env = GreenHouseEnv()

        class GreenHouseReward(Reward):
            def __init__(self):
                super().__init__(x_dim=7, u_dim=2)

            def __call__(self,
                         x: chex.Array,
                         u: chex.Array,
                         reward_params: RewardParams,
                         x_next: chex.Array | None = None
                         ):
                reward = env.reward(x, u, )
                reward_dist = Normal(reward, jnp.zeros_like(reward))
                return reward_dist, reward_params

            def init_params(self, key: chex.PRNGKey) -> RewardParams:
                return {}

        reward_model = GreenHouseReward()

    elif env_name == 'Reacher':
        from wtc.envs.reacher_dm_control import ReacherDMControl
        env = ReacherDMControl(backend='generalized')
        horizon = 200

        class ReacherReward(Reward):
            def __init__(self):
                super().__init__(x_dim=7, u_dim=2)

            def __call__(self,
                         x: chex.Array,
                         u: chex.Array,
                         reward_params: RewardParams,
                         x_next: chex.Array | None = None
                         ):
                reward = env.reward(x, u)
                reward_dist = Normal(reward, jnp.zeros_like(reward))
                return reward_dist, reward_params

            def init_params(self, key: chex.PRNGKey) -> RewardParams:
                return {}

        reward_model = ReacherReward()

    key_offline_data, key_agent = jr.split(jr.PRNGKey(seed))

    offline_data = None

    if regression_model == 'probabilistic_ensemble':
        model = BNNStatisticalModel(
            input_dim=env.observation_size + env.action_size,
            output_dim=env.observation_size,
            num_training_steps=bnn_steps,
            output_stds=1e-3 * jnp.ones(env.observation_size),
            beta=2.0 * jnp.ones(shape=(env.observation_size,)),
            features=(64, 64, 64),
            bnn_type=ProbabilisticEnsemble,
            num_particles=10,
            logging_wandb=False,
            return_best_model=True,
            eval_batch_size=64,
            train_share=0.8,
            eval_frequency=500,
            weight_decay=1e-3,
            activation=nn.leaky_relu
        )
    elif regression_model == 'FSVGD':
        model = BNNStatisticalModel(
            input_dim=env.observation_size + env.action_size,
            output_dim=env.observation_size,
            num_training_steps=bnn_steps,
            output_stds=1e-3 * jnp.ones(env.observation_size),
            beta=2.0 * jnp.ones(shape=(env.observation_size,)),
            features=(64, 64, 64),
            bnn_type=ProbabilisticFSVGDEnsemble,
            num_particles=10,
            logging_wandb=False,
            return_best_model=True,
            eval_batch_size=64,
            train_share=0.8,
            eval_frequency=500,
            weight_decay=1e-3,
            activation=nn.leaky_relu
        )
    elif regression_model == 'GP':
        model = GPStatisticalModel(
            input_dim=env.observation_size + env.action_size,
            output_dim=env.observation_size,
            output_stds=1e-3 * jnp.ones(env.observation_size),
            f_norm_bound=1.0,
            delta=0.1,
            num_training_steps=1000,
        )

    discount_factor = 0.99

    num_envs = 128
    num_env_steps_between_updates = 16
    sac_kwargs = {
        'num_timesteps': sac_steps,
        'episode_length': sac_horizon,
        'num_env_steps_between_updates': num_env_steps_between_updates,
        'num_envs': num_envs,
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
        'reward_scaling': 10.,
        'tau': 0.005,
        'min_replay_size': 10 ** 3,
        'max_replay_size': sac_steps,
        'grad_updates_per_step': num_env_steps_between_updates * num_envs,
        'deterministic_eval': True,
        'init_log_alpha': 0.,
        'policy_hidden_layer_sizes': (64,) * 2,
        'policy_activation': swish,
        'critic_hidden_layer_sizes': (64,) * 2,
        'critic_activation': swish,
        'wandb_logging': True,
        'return_best_model': True,
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
        agent_class = OptimisticModelBasedAgent
    elif exploration == 'pets':
        agent_class = PETSModelBasedAgent

    agent = agent_class(
        env=env,
        eval_env=env,
        statistical_model=model,
        optimizer=optimizer,
        episode_length=horizon,
        reward_model=reward_model,
        offline_data=offline_data,
        num_envs=1,
        num_eval_envs=1,
        log_to_wandb=log_wandb,
        deterministic_policy_for_data_collection=deterministic_policy_for_data_collection,
        first_episode_for_policy_training=first_episode_for_policy_training,
        reset_statistical_model=reset_statistical_model,
        max_collected_data_in_buffer=max_replay_size_true_data_buffer
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
               regression_model=args.regression_model,
               env_name=args.env_name
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
    parser.add_argument('--first_episode_for_policy_training', type=int, default=-1)
    parser.add_argument('--exploration', type=str, default='pets')
    parser.add_argument('--reset_statistical_model', type=int, default=0)
    parser.add_argument('--regression_model', type=str, default='FSVGD')
    parser.add_argument('--env_name', type=str, default='Reacher')

    args = parser.parse_args()
    main(args)
