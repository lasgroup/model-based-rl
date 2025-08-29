import argparse
import os
import sys

import numpy as np

from ombrl.utils.experiment_utils import Logger, hash_dict
from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv


def experiment(
        project_name: str = 'Ombrl-gp',
        entity_name: str = 'trevenl',
        exp_hash: str = '42',
        seed: int = 0,
        num_particles: int = 10,
        num_samples: int = 500,
        alpha: float = 0.2,
        num_steps: int = 5,
        exponent: int = 2,
        icem_horizon: int = 20,
        episode_length: int = 50,
        action_repeat: int = 2,
        num_training_steps: int = 1_000,
        log_wandb: bool = True,
        logs_dir: str = 'runs',
        num_gpus: int = 0,
        function_norm: float = 1.0,
        num_elites: int = 50,
        beta: float = 3.0,
        exploration_type: str = 'greedy',
        lambda_param: float = 1.0,
        intrinsic_reward_computation_type: str = 'norm',
        scale_with_aleatoric_std: bool = False,
        num_episodes: int = 10,
):
    if num_gpus == 0:
        import os
        os.environ['JAX_PLATFORMS'] = 'cpu'

    import jax.numpy as jnp
    import jax.random as jr
    import chex
    import wandb
    from ombrl.agent.actsafe import SafeHUCRL, Task
    from distrax import Distribution, Normal
    from typing import Tuple
    from optax import constant_schedule
    from mbpo.systems.rewards.base_rewards import Reward, RewardParams
    from ombrl.optimizer.icem import iCemParams
    from ombrl.envs.mountain_car import MountainCarEnv
    from bsm.statistical_model import GPStatisticalModel
    from ombrl.dynamics_models.gps import ARD
    from jaxtyping import Array, Float

    env = MountainCarEnv()
    key = jr.PRNGKey(seed)

    configs = dict(
        seed=seed,
        num_particles=num_particles,
        num_samples=num_samples,
        alpha=alpha,
        num_steps=num_steps,
        exponent=exponent,
        icem_horizon=icem_horizon,
        episode_length=episode_length,
        action_repeat=action_repeat,
        num_training_steps=num_training_steps,
        num_gpus=num_gpus,
        function_norm=function_norm,
        num_elites=num_elites,
        beta=beta,
        exploration_type=exploration_type,
        lambda_param=lambda_param,
        intrinsic_reward_computation_type=intrinsic_reward_computation_type,
        scale_with_aleatoric_std=scale_with_aleatoric_std,
        num_episodes=num_episodes,
    )

    model = GPStatisticalModel(
        kernel=ARD(input_dim=env.observation_size + env.action_size, length_scale=0.1),
        input_dim=env.observation_size + env.action_size,
        output_dim=env.observation_size,
        output_stds=1e-3 * jnp.ones(shape=(env.observation_size,)),
        logging_wandb=log_wandb,
        beta=jnp.ones(env.observation_size) * beta,
        num_training_steps=constant_schedule(num_training_steps),
        lr_rate=1e-2,
        weight_decay=1e-3,
    )

    class MCReward(Reward):
        def __init__(self):
            super().__init__(x_dim=2, u_dim=1)
            self.env = Continuous_MountainCarEnv()

        def reward(self,
                   obs: Float[Array, '2'],
                   action: Float[Array, '1'],
                   next_obs: Float[Array, '2'], ) -> Float[Array, '1']:
            pos = next_obs[..., 0]
            velocity = next_obs[..., 1]
            terminate = jnp.logical_and(pos >= self.env.goal_position, velocity >= self.env.goal_velocity)
            reward = - (action[..., 0] ** 2) * 0.1 + 100 * terminate
            return reward.reshape(-1).squeeze()

        def __call__(self,
                     x: chex.Array,
                     u: chex.Array,
                     reward_params: Tuple,
                     x_next: chex.Array | None = None) -> Tuple[Distribution, RewardParams]:
            chex.assert_shape(x, (self.x_dim,))
            chex.assert_shape(u, (self.u_dim,))
            chex.assert_shape(x_next, (self.x_dim,))
            reward = self.reward(x, u, x_next)
            return Normal(loc=reward, scale=jnp.zeros_like(reward)), reward_params

        def init_params(self, key: chex.PRNGKey) -> Tuple:
            return ()

    icem_params = iCemParams(
        num_particles=num_particles,
        num_samples=num_samples,
        num_elites=num_elites,
        alpha=alpha,
        num_steps=num_steps,
        exponent=exponent,
    )

    agent = SafeHUCRL(
        env=MountainCarEnv(),
        model=model,
        episode_length=episode_length,
        action_repeat=action_repeat,
        cost_fn=None,
        test_tasks=[
            Task(reward=MCReward(), name='Swing up', env=env),
        ],
        predict_difference=True,
        num_training_steps=constant_schedule(num_training_steps),
        icem_horizon=icem_horizon,
        icem_params=icem_params,
        log_to_wandb=log_wandb,
        exploration_type=exploration_type,
        lambda_param=lambda_param,
        intrinsic_reward_computation_type=intrinsic_reward_computation_type,
        scale_with_aleatoric_std=scale_with_aleatoric_std,
    )

    if log_wandb:
        wandb.init(project=project_name,
                   config=configs,
                   entity=entity_name,
                   dir=logs_dir,
                   )

    model_state = model.init(jr.PRNGKey(seed))
    agent.run_episodes(num_episodes=num_episodes,
                       key=key,
                       model_state=model_state,
                       folder_name=f'{logs_dir}/{exp_hash}/',
                       )
    wandb.finish()


def main(args):
    """"""
    from pprint import pprint
    print(args)
    """ generate experiment hash and set up redirect of output streams """
    exp_hash = hash_dict(args.__dict__)
    if args.exp_result_folder is not None:
        os.makedirs(args.exp_result_folder, exist_ok=True)
        log_file_path = os.path.join(args.exp_result_folder, '%s.log ' % exp_hash)
        logger = Logger(log_file_path)
        sys.stdout = logger
        sys.stderr = logger

    pprint(args.__dict__)
    print('\n ------------------------------------ \n')

    """ Experiment core """
    np.random.seed(args.seed)

    experiment(
        project_name=args.project_name,
        entity_name=args.entity_name,
        action_repeat=args.action_repeat,
        num_particles=args.num_particles,
        num_samples=args.num_samples,
        alpha=args.alpha,
        num_steps=args.num_steps,
        exponent=args.exponent,
        icem_horizon=args.icem_horizon,
        episode_length=args.episode_length,
        num_training_steps=args.num_training_steps,
        log_wandb=bool(args.log_wandb),
        seed=args.seed,
        logs_dir=args.logs_dir,
        num_gpus=args.num_gpus,
        exp_hash=exp_hash,
        function_norm=args.function_norm,
        num_elites=args.num_elites,
        beta=args.beta,
        exploration_type=args.exploration_type,
        lambda_param=args.lambda_param,
        intrinsic_reward_computation_type=args.intrinsic_reward_computation_type,
        scale_with_aleatoric_std=bool(args.scale_with_aleatoric_std),
        num_episodes=args.num_episodes,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MTTest')

    # general experiment args
    parser.add_argument('--logs_dir', type=str, default='logs')
    parser.add_argument('--project_name', type=str, default='ActSafeTest')
    parser.add_argument('--entity_name', type=str, default='trevenl')
    parser.add_argument('--num_particles', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--num_steps', type=int, default=5)
    parser.add_argument('--exponent', type=float, default=1.0)
    parser.add_argument('--icem_horizon', type=int, default=25)
    parser.add_argument('--episode_length', type=int, default=50)
    parser.add_argument('--action_repeat', type=int, default=4)
    parser.add_argument('--num_training_steps', type=int, default=1_000)
    parser.add_argument('--log_wandb', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=0)
    parser.add_argument('--function_norm', type=float, default=1.0)
    parser.add_argument('--num_elites', type=int, default=100)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--lambda_param', type=float, default=10.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_result_folder', type=str, default=None)
    parser.add_argument('--exploration_type', type=str, default='ombrl')
    parser.add_argument('--intrinsic_reward_computation_type', type=str, default='norm')
    parser.add_argument('--scale_with_aleatoric_std', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default=10)

    args = parser.parse_args()
    main(args)
