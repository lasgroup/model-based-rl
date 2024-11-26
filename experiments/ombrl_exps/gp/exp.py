import argparse

import chex
import jax.numpy as jnp
import jax.random as jr
import wandb
from bsm.statistical_model.gp_statistical_model import GPStatisticalModel
from bsm.utils import Data
from distrax import Normal
from mbpo.optimizers import iCEMOptimizer, iCemParams
from mbpo.systems.rewards.base_rewards import Reward, RewardParams

from mbrl.envs.pendulum import PendulumEnv
from mbrl.model_based_agent import OptimisticModelBasedAgent, PETSModelBasedAgent
from mbrl.utils.kernels import ARD


def experiment(project_name: str = 'GPUSpeedTest',
               seed: int = 42,
               icem_horizon: int = 20,
               num_particles: int = 10,
               num_samples: int = 500,
               num_elites: int = 50,
               init_std: float = 1.0,
               num_steps: int = 5,
               exponent: float = 1.0,
               num_episodes: int = 20,
               exploration: str = 'optimistic',  # Should be one of the ['optimistic', 'pets', 'mean'],
               reset_statistical_model: bool = True,
               exploration_factor: float = 1.0,
               horizon: int = 100,
               log_wandb: bool = False,
               entity: str = 'trevenl',
               calibration: bool = False,
               int_reward_weight: float = 1.0,
               reward_source: str = 'gym',
               sample_with_eps_std: bool = False,
               action_repeat: int = 2,
               ):
    assert exploration in ['optimistic', 'pets',
                           'hucrl'], "Unrecognized exploration strategy, should be 'optimistic' or 'pets' or 'mean'"

    import jax
    jax.config.update("jax_enable_x64", True)
    config = dict(icem_horizon=icem_horizon,
                  num_samples=num_samples,
                  seed=seed,
                  num_episodes=num_episodes,
                  num_steps=num_steps,
                  num_particles=num_particles,
                  exploration=exploration,
                  reset_statistical_model=reset_statistical_model,
                  regression_model='GP',
                  num_elites=num_elites,
                  exploration_factor=exploration_factor,
                  horizon=horizon,
                  init_std=init_std,
                  exponent=exponent,
                  int_reward_weight=int_reward_weight,
                  calibration=calibration,
                  sample_with_eps_std=sample_with_eps_std,
                  reward_source=reward_source,
                  env_name='pendulum',
                  action_repeat=action_repeat
                  )

    env = PendulumEnv(reward_source=reward_source, margin_factor=100.0)

    class PendulumReward(Reward):
        def __init__(self, reward_source: str = 'gym'):
            super().__init__(x_dim=3, u_dim=1)
            self.reward_source = reward_source

        def __call__(self,
                     x: chex.Array,
                     u: chex.Array,
                     reward_params: RewardParams,
                     x_next: chex.Array | None = None
                     ):
            assert x.shape == (3,) and u.shape == (1,)
            if self.reward_source == 'gym':
                reward = env.reward(x, u)
            else:
                reward = env.dm_reward(x, u)
            reward = reward.squeeze()
            reward_dist = Normal(reward, jnp.zeros_like(reward))
            return reward_dist, reward_params

        def init_params(self, key: chex.PRNGKey) -> RewardParams:
            return {'dt': env.dt}

    key_offline_data, key_agent = jr.split(jr.PRNGKey(seed))
    offline_data = None

    model = GPStatisticalModel(
        kernel=ARD(input_dim=env.observation_size + env.action_size),
        input_dim=env.observation_size + env.action_size,
        output_dim=env.observation_size,
        output_stds=1e-3 * jnp.ones(shape=(env.observation_size,)),
        logging_wandb=log_wandb,
        beta=jnp.ones(3) * exploration_factor,
        num_training_steps=2000,
        lr_rate=1e-2,
        weight_decay=0.0,
    )

    opt_params = iCemParams(
        num_particles=num_particles,
        num_samples=num_samples,
        num_elites=num_samples,
        init_std=init_std,
        num_steps=num_steps,
        exponent=exponent,
        alpha=0.2,
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
                                  'int_reward_weight': int_reward_weight,
                                  'sample_with_eps_std': sample_with_eps_std,
                                  }
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
        reward_model=PendulumReward(reward_source=reward_source),
        episode_length=horizon,
        offline_data=offline_data,
        num_envs=1,
        num_eval_envs=1,
        log_to_wandb=log_wandb,
        reset_statistical_model=reset_statistical_model,
        action_repeat=action_repeat,
        **additional_agent_kwarg
    )

    key_agent, key = jr.split(key_agent)
    agent_state = agent.init(key)

    consistent_data = Data(inputs=jnp.array([[-1.0, 0., 0., 0., ]]),
                           outputs=jnp.array([[0., 0., 0.]]))

    agent_state.optimizer_state.system_params.dynamics_params.statistical_model_state.model_state.history = consistent_data

    agent_state = agent.run_episodes(num_episodes=num_episodes,
                                     start_from_scratch=False,
                                     agent_state=agent_state,
                                     key=key_agent)

    wandb.finish()


def main(args):
    experiment(project_name=args.project_name,
               icem_horizon=args.icem_horizon,
               num_particles=args.num_particles,
               num_samples=args.num_samples,
               num_elites=args.num_elites,
               init_std=args.init_std,
               exponent=args.exponent,
               num_steps=args.num_steps,
               seed=args.seed,
               num_episodes=args.num_episodes,
               exploration=args.exploration,
               reset_statistical_model=bool(args.reset_statistical_model),
               exploration_factor=args.exploration_factor,
               horizon=args.horizon,
               log_wandb=bool(args.log_wandb),
               entity=args.entity,
               calibration=bool(args.calibration),
               reward_source=args.reward_source,
               sample_with_eps_std=bool(args.sample_with_eps_std),
               int_reward_weight=args.int_reward_weight,
               action_repeat=args.action_repeat
               )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='Model_based_pets')
    parser.add_argument('--num_offline_samples', type=int, default=0)
    parser.add_argument('--icem_horizon', type=int, default=5)
    parser.add_argument('--num_particles', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--num_elites', type=int, default=50)
    parser.add_argument('--init_std', type=float, default=0.5)
    parser.add_argument('--num_steps', type=int, default=5)
    parser.add_argument('--exponent', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default=20)

    parser.add_argument('--exploration', type=str, default='pets')
    parser.add_argument('--reset_statistical_model', type=int, default=1)
    parser.add_argument('--exploration_factor', type=float, default=1.0)
    parser.add_argument('--horizon', type=int, default=200)
    parser.add_argument('--log_wandb', type=int, default=1)
    parser.add_argument('--entity', type=str, default='trevenl')
    parser.add_argument('--int_reward_weight', type=float, default=2.0)
    parser.add_argument('--calibration', type=int, default=1)
    parser.add_argument('--reward_source', type=str, default='dm-control')
    parser.add_argument('--sample_with_eps_std', type=int, default=0)
    parser.add_argument('--action_repeat', type=int, default=5)

    args = parser.parse_args()
    main(args)
