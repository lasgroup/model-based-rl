import argparse

import chex
import jax.numpy as jnp
import jax.random as jr
import wandb
from bsm.bayesian_regression import ProbabilisticEnsemble
from bsm.statistical_model.bnn_statistical_model import BNNStatisticalModel
from distrax import Normal
from mbpo.optimizers import iCemParams, iCEMOptimizer
from mbpo.systems.rewards.base_rewards import Reward, RewardParams

from mbrl.envs.pendulum import PendulumEnv
from mbrl.model_based_agent.pets_model_based_agent import PETSModelBasedAgent
from mbrl.utils.offline_data import PendulumOfflineData

ENTITY = 'sukhijab'


def experiment(project_name: str = 'GPUSpeedTest',
               num_offline_samples: int = 100,
               deterministic_policy_for_data_collection: bool = False,
               seed: int = 0,
               icem_horizon: int = 20,
               bnn_train_steps: int = 15_000,
               reset_statistical_model: bool = False,
               ):
    config = dict(num_offline_samples=num_offline_samples,
                  deterministic_policy_for_data_collection=deterministic_policy_for_data_collection,
                  icem_horizon=icem_horizon,
                  bnn_train_steps=bnn_train_steps,
                  reset_statistical_model=reset_statistical_model,
                  seed=seed,
                  )

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

    offline_data_gen = PendulumOfflineData()
    key = jr.PRNGKey(seed)
    key, offline_data_key = jr.split(key, 2)

    if num_offline_samples > 0:
        offline_data = offline_data_gen.sample_transitions(key=offline_data_key,
                                                           num_samples=num_offline_samples)
    else:
        offline_data = None

    horizon = 200
    model = BNNStatisticalModel(
        input_dim=env.observation_size + env.action_size,
        output_dim=env.observation_size,
        num_training_steps=bnn_train_steps,
        output_stds=1e-3 * jnp.ones(env.observation_size),
        features=(64, 64, 64),
        num_particles=5,
        bnn_type=ProbabilisticEnsemble,
        logging_wandb=True,
        return_best_model=True,
        eval_batch_size=64,
        train_share=0.8,
        eval_frequency=5_000,
    )

    opt_params = iCemParams(exponent=1.0, )
    optimizer = iCEMOptimizer(horizon=icem_horizon,
                              key=jr.PRNGKey(seed),
                              opt_params=opt_params,
                              )

    wandb.init(project=project_name,
               dir='/cluster/scratch/' + ENTITY,
               config=config
               )

    agent = PETSModelBasedAgent(
        env=env,
        eval_env=env,
        statistical_model=model,
        optimizer=optimizer,
        reward_model=PendulumReward(),
        episode_length=horizon,
        offline_data=offline_data,
        num_envs=1,
        num_eval_envs=1,
        log_to_wandb=True,
        deterministic_policy_for_data_collection=deterministic_policy_for_data_collection,
        reset_statistical_model=reset_statistical_model,
    )

    agent_state = agent.run_episodes(num_episodes=20,
                                     start_from_scratch=True,
                                     key=key)

    wandb.finish()


def main(args):
    experiment(project_name=args.project_name,
               num_offline_samples=args.num_offline_samples,
               deterministic_policy_for_data_collection=bool(args.deterministic_policy_for_data_collection),
               seed=args.seed,
               icem_horizon=args.icem_horizon,
               bnn_train_steps=args.bnn_train_steps,
               reset_statistical_model=bool(args.reset_statistical_model),
               )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='Model_based_pets')
    parser.add_argument('--num_offline_samples', type=int, default=100)
    parser.add_argument('--deterministic_policy_for_data_collection', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--icem_horizon', type=int, default=20)
    parser.add_argument('--bnn_train_steps', type=int, default=15_000)
    parser.add_argument('--reset_statistical_model', type=int, default=0)

    args = parser.parse_args()
    main(args)
