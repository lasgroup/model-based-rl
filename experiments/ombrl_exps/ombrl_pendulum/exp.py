import argparse

import chex
import jax.numpy as jnp
import jax.random as jr
import optax
import wandb
from distrax import Normal
from mbpo.systems.rewards.base_rewards import Reward, RewardParams
from mbpo.optimizers import iCEMOptimizer, iCemParams
from bsm.bayesian_regression import ProbabilisticEnsemble, ProbabilisticFSVGDEnsemble
from bsm.statistical_model.bnn_statistical_model import BNNStatisticalModel
from bsm.statistical_model.gp_statistical_model import GPStatisticalModel
from mbrl.envs.pendulum import PendulumEnv

from mbrl.model_based_agent import OptimisticModelBasedAgent, PETSModelBasedAgent


def experiment(project_name: str = 'GPUSpeedTest',
               seed: int = 42,
               num_offline_samples: int = 0,
               icem_horizon: int = 20,
               num_particles: int = 10,
               num_samples: int = 500,
               num_elites: int = 50,
               init_std: float = 1.0,
               num_steps: int = 5,
               exponent: float = 1.0,
               num_episodes: int = 20,
               min_bnn_steps: int = 1_000,
               max_bnn_steps: int = 50_000,
               linear_scheduler_steps: int = 20_000,
               exploration: str = 'optimistic',  # Should be one of the ['optimistic', 'pets', 'mean'],
               reset_statistical_model: bool = True,
               regression_model: str = 'probabilistic_ensemble',
               exploration_factor: float = 1.0,
               horizon: int = 100,
               log_wandb: bool = False,
               entity: str = 'trevenl'
               ):
    assert exploration in ['optimistic', 'pets',
                           'mean'], "Unrecognized exploration strategy, should be 'optimistic' or 'pets' or 'mean'"
    assert regression_model in ['probabilistic_ensemble', 'FSVGD', 'GP']

    num_training_points = optax.linear_schedule(init_value=min_bnn_steps, end_value=max_bnn_steps,
                                                transition_steps=linear_scheduler_steps)
    config = dict(num_offline_samples=num_offline_samples,
                  icem_horizon=icem_horizon,
                  num_samples=num_samples,
                  seed=seed,
                  num_episodes=num_episodes,
                  num_steps=num_steps,
                  min_bnn_steps=min_bnn_steps,
                  max_bnn_steps=max_bnn_steps,
                  linear_scheduler_steps=linear_scheduler_steps,
                  num_particles=num_particles,
                  exploration=exploration,
                  reset_statistical_model=reset_statistical_model,
                  regression_model=regression_model,
                  num_elites=num_elites,
                  exploration_factor=exploration_factor,
                  horizon=horizon,
                  init_std=init_std,
                  exponent=exponent,
                  )

    env = PendulumEnv(reward_source='dm-control')

    class PendulumReward(Reward):
        def __init__(self):
            super().__init__(x_dim=3, u_dim=1)

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

    key_offline_data, key_agent = jr.split(jr.PRNGKey(seed))

    if num_offline_samples == 0:
        offline_data = None
    else:
        raise NotImplementedError('Offline data not implemented yet.')

    if regression_model == 'probabilistic_ensemble':
        model = BNNStatisticalModel(
            input_dim=env.observation_size + env.action_size,
            output_dim=env.observation_size,
            num_training_steps=num_training_points,
            output_stds=1e-3 * jnp.ones(env.observation_size),
            beta=exploration_factor * jnp.ones(shape=(env.observation_size,)),
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
            input_dim=env.observation_size + env.action_size,
            output_dim=env.observation_size,
            num_training_steps=num_training_points,
            output_stds=1e-3 * jnp.ones(env.observation_size),
            beta=exploration_factor * jnp.ones(shape=(env.observation_size,)),
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
            input_dim=env.observation_size + env.action_size,
            output_dim=env.observation_size,
            output_stds=1e-3 * jnp.ones(env.observation_size),
            f_norm_bound=1.0,
            delta=0.1,
            num_training_steps=1000,
            logging_frequency=100,
            beta=exploration_factor * jnp.ones(shape=(env.observation_size,)),
        )
    else:
        raise NotImplementedError

    opt_params = iCemParams(
        num_particles=num_particles,
        num_samples=num_samples,
        num_elites=num_samples,
        init_std=init_std,
        num_steps=num_steps,
        exponent=exponent,
    )
    key_agent, key_icem = jr.split(key_agent, 2)

    optimizer = iCEMOptimizer(horizon=icem_horizon,
                              key=key_icem,
                              opt_params=opt_params,
                              )

    if log_wandb:
        wandb.init(project=project_name,
                   dir='/cluster/scratch/' + entity,
                   config=config)

    if exploration == 'optimistic':
        agent_class = OptimisticModelBasedAgent
        additional_agent_kwarg = {'use_hallucinated_controls': False,
                                  'int_reward_weight': 1.0}
    elif exploration == 'hucrl':
        agent_class = OptimisticModelBasedAgent
        additional_agent_kwarg = {'use_hallucinated_controls': True}
    elif exploration == 'pets':
        agent_class = PETSModelBasedAgent
        additional_agent_kwarg = {}
    else:
        raise NotImplementedError

    agent = agent_class(
        env=env,
        eval_env=env,
        statistical_model=model,
        optimizer=optimizer,
        reward_model=PendulumReward(),
        episode_length=horizon,
        offline_data=offline_data,
        num_envs=1,
        num_eval_envs=1,
        log_to_wandb=log_wandb,
        reset_statistical_model=reset_statistical_model,
        **additional_agent_kwarg
    )

    agent_state = agent.run_episodes(num_episodes=num_episodes,
                                     start_from_scratch=True,
                                     key=key_agent)

    wandb.finish()


def main(args):
    experiment(project_name=args.project_name,
               num_offline_samples=args.num_offline_samples,
               icem_horizon=args.icem_horizon,
               num_particles=args.num_particles,
               num_samples=args.num_samples,
               num_elites=args.num_elites,
               init_std=args.init_std,
               exponent=args.exponent,
               num_steps=args.num_steps,
               seed=args.seed,
               num_episodes=args.num_episodes,
               min_bnn_steps=args.min_bnn_steps,
               max_bnn_steps=args.max_bnn_steps,
               linear_scheduler_steps=args.linear_scheduler_steps,
               exploration=args.exploration,
               reset_statistical_model=bool(args.reset_statistical_model),
               regression_model=args.regression_model,
               exploration_factor=args.exploration_factor,
               horizon=args.horizon,
               log_wandb=bool(args.log_wandb),
               entity=args.entity,
               )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='Model_based_pets')
    parser.add_argument('--num_offline_samples', type=int, default=0)
    parser.add_argument('--icem_horizon', type=int, default=20)
    parser.add_argument('--num_particles', type=int, default=10)
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--num_elites', type=int, default=50)
    parser.add_argument('--init_std', type=float, default=1.0)
    parser.add_argument('--num_steps', type=int, default=5)
    parser.add_argument('--exponent', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_episodes', type=int, default=50)

    parser.add_argument('--min_bnn_steps', type=int, default=5_000)
    parser.add_argument('--max_bnn_steps', type=int, default=50_000)
    parser.add_argument('--linear_scheduler_steps', type=int, default=20_000)
    parser.add_argument('--exploration', type=str, default='optimistic')
    parser.add_argument('--reset_statistical_model', type=int, default=0)
    parser.add_argument('--regression_model', type=str, default='FSVGD')
    parser.add_argument('--exploration_factor', type=float, default=2.0)
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--log_wandb', type=int, default=0)
    parser.add_argument('--entity', type=str, default='trevenl')

    args = parser.parse_args()
    main(args)
