import argparse

log_wandb = True
ENTITY = 'kiten'


def experiment(project_name: str = 'GPUSpeedTest',
               num_offline_samples: int = 0,
               optimizer_horizon: int = 100,
               num_online_samples: int = 200,
               deterministic_policy_for_data_collection: bool = False,
               noise_level: list = [0.1, 0.1],
               icem_num_steps: int = 10,
               icem_colored_noise_exponent: float = 3.0,
               reward_source = 'dm-control',
               seed: int = 42,
               num_episodes: int = 20,
               bnn_steps: int = 5_000,
               first_episode_for_policy_training: int = -1,
               exploration: str = 'optimistic',  # Should be one of the ['optimistic', 'pets', 'mean'],
               reset_statistical_model: bool = True,
               regression_model: str = 'probabilistic_ensemble',
               beta: float = 2.0,
               weight_decay: float = 0.0
               ):
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
    from mbpo.optimizers import iCEMOptimizer, iCemParams
    from mbpo.systems.rewards.base_rewards import Reward, RewardParams

    from mbrl.envs.pendulum import PendulumEnv
    from mbrl.model_based_agent import PETSModelBasedAgent, OptimisticModelBasedAgent
    
    assert exploration in ['optimistic',
                           'pets'], "Unrecognized exploration strategy, should be 'optimistic' or 'pets' or 'mean'"
    assert regression_model in ['probabilistic_ensemble', 'deterministic_ensemble', 'FSVGD', 'GP']
    assert reward_source in ['dm-control', 'gym']

    config = dict(num_offline_samples=num_offline_samples,
                  optimizer_horizon=optimizer_horizon,
                  num_online_samples=num_online_samples,
                  deterministic_policy_for_data_collection=deterministic_policy_for_data_collection,
                  noise_level=noise_level,
                  seed=seed,
                  num_episodes=num_episodes,
                  icem_num_steps=icem_num_steps,
                  icem_colored_noise_exponent=icem_colored_noise_exponent,
                  reward_source=reward_source,
                  bnn_steps=bnn_steps,
                  first_episode_for_policy_training=first_episode_for_policy_training,
                  exploration=exploration,
                  reset_statistical_model=reset_statistical_model,
                  regression_model=regression_model,
                  beta=beta,
                  weight_decay=weight_decay
                  )

    swing_up_env = PendulumEnv(reward_source=reward_source)
    balance_env = PendulumEnv(reward_source=reward_source, initial_angle=0.)

    env = balance_env
    eval_env = swing_up_env

    key_offline_data, key_agent = jr.split(jr.PRNGKey(seed))

    if num_offline_samples > 0:
        raise NotImplementedError
    else:
        offline_data = None

    if regression_model == 'probabilistic_ensemble':
        model = BNNStatisticalModel(
            input_dim=env.observation_size + env.action_size,
            output_dim=env.observation_size,
            num_training_steps=bnn_steps,
            output_stds=1e-3 * jnp.ones(env.observation_size),
            beta=beta * jnp.ones(shape=(env.observation_size,)),
            features=(256,) * 2,
            bnn_type=ProbabilisticEnsemble,
            num_particles=10,
            logging_wandb=log_wandb,
            return_best_model=True,
            eval_batch_size=64,
            train_share=0.8,
            eval_frequency=5_000,
            weight_decay=weight_decay,
        )
    elif regression_model == 'deterministic_ensemble':
        model = BNNStatisticalModel(
            input_dim=env.observation_size + env.action_size,
            output_dim=env.observation_size,
            num_training_steps=bnn_steps,
            output_stds=1e-3 * jnp.ones(env.observation_size),
            features=(256,) * 2,
            num_particles=10,
            logging_wandb=log_wandb,
            return_best_model=True,
            eval_batch_size=64,
            train_share=0.8,
            eval_frequency=5_000,
            weight_decay=weight_decay,
        )
    elif regression_model == 'FSVGD':
        model = BNNStatisticalModel(
            input_dim=env.observation_size + env.action_size,
            output_dim=env.observation_size,
            num_training_steps=bnn_steps,
            output_stds=1e-3 * jnp.ones(env.observation_size),
            beta=beta * jnp.ones(shape=(env.observation_size,)),
            features=(64, 64, 64),
            bnn_type=ProbabilisticFSVGDEnsemble,
            num_particles=10,
            logging_wandb=log_wandb,
            return_best_model=True,
            eval_batch_size=64,
            train_share=0.8,
            eval_frequency=5_000,
            weight_decay=weight_decay,
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

    opt_params = iCemParams(
        num_steps=icem_num_steps,
        exponent=icem_colored_noise_exponent,
        )
    
    optimizer = iCEMOptimizer(horizon=optimizer_horizon,
                              key = jr.PRNGKey(seed),
                              opt_params=opt_params,
                              )
    
    if log_wandb:
        wandb.init(project=project_name,
                   dir='/cluster/scratch/' + ENTITY,
                   config=config)
        
    # debugging. TODO: REMOVE!
    print("Max speed is: ", env.max_speed)

    agent_class = None
    if exploration == 'optimistic':
        agent_class = OptimisticModelBasedAgent
    elif exploration == 'pets':
        agent_class = PETSModelBasedAgent

    class PendulumReward(Reward):
        def __init__(self, reward_env: PendulumEnv, reward_source: str):
            super().__init__(x_dim=3, u_dim=1)
            self.env = reward_env
            self.reward_source = reward_source

        def __call__(self,
                     x: chex.Array,
                     u: chex.Array,
                     reward_params: RewardParams,
                     x_next: chex.Array | None = None
                     ):
            assert x.shape == (3,) and u.shape == (1,)
            theta, omega = jnp.arctan2(x[1], x[0]), x[-1]
            target_angle = self.env.reward_params.target_angle
            diff_th = theta - target_angle
            diff_th = ((diff_th + jnp.pi) % (2 * jnp.pi)) - jnp.pi
            if self.reward_source == 'gym':
                reward = -(self.env.reward_params.angle_cost * diff_th ** 2 +
                           0.1 * omega ** 2) - self.env.reward_params.control_cost * u ** 2
            else:
                reward = self.env.tolerance_reward(jnp.sqrt(self.env.reward_params.angle_cost * diff_th ** 2 +
                                                            0.1 * omega ** 2)) - self.env.reward_params.control_cost * u ** 2
            reward = reward.squeeze()
            reward_dist = Normal(reward, jnp.zeros_like(reward))
            return reward_dist, reward_params

        def init_params(self, key: chex.PRNGKey) -> RewardParams:
            return {'dt': self.env.dt}

    agent = agent_class(
        env=env,
        eval_env=eval_env,
        statistical_model=model,
        optimizer=optimizer,
        episode_length=num_online_samples,
        reward_model=PendulumReward(env, reward_source),
        offline_data=offline_data,
        num_envs=1,
        num_eval_envs=1,
        log_to_wandb=log_wandb,
        deterministic_policy_for_data_collection=deterministic_policy_for_data_collection,
        first_episode_for_policy_training=first_episode_for_policy_training,
        reset_statistical_model=reset_statistical_model,
        dt=env.dt,
    )

    agent_state = agent.run_episodes(num_episodes=num_episodes,
                                     start_from_scratch=True,
                                     key=key_agent)

    print("Finishing wandb")
    wandb.finish()
    print("Ended wandb")

def main(args):
    experiment(project_name=args.project_name,
        num_offline_samples=args.num_offline_samples,
        optimizer_horizon=args.optimizer_horizon,
        num_online_samples=args.num_online_samples,
        deterministic_policy_for_data_collection=bool(args.deterministic_policy_for_data_collection),
        noise_level=args.noise_level,
        icem_num_steps=args.icem_num_steps,
        icem_colored_noise_exponent=args.icem_colored_noise_exponent,
        reward_source=args.reward_source,
        seed=args.seed,
        num_episodes=args.num_episodes,
        bnn_steps=args.bnn_steps,
        first_episode_for_policy_training=args.first_episode_for_policy_training,
        exploration=args.exploration,
        reset_statistical_model=bool(args.reset_statistical_model),
        regression_model=args.regression_model,
        beta=args.beta,
        weight_decay=args.weight_decay
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='Model_based_pets')
    parser.add_argument('--num_offline_samples', type=int, default=0)
    parser.add_argument('--optimizer_horizon', type=int, default=100)
    parser.add_argument('--num_online_samples', type=int, default=200)
    parser.add_argument('--deterministic_policy_for_data_collection', type=int, default=0)
    parser.add_argument('--noise_level', nargs=2, type=float, default=[0.1, 0.1])
    parser.add_argument('--icem_num_steps', type=int, default=10)
    parser.add_argument('--icem_colored_noise_exponent', type=float, default=3.0)
    parser.add_argument('--reward_source', type=str, default='dm-control')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_episodes', type=int, default=5)
    parser.add_argument('--bnn_steps', type=int, default=5_000)
    parser.add_argument('--first_episode_for_policy_training', type=int, default=2)
    parser.add_argument('--exploration', type=str, default='pets')
    parser.add_argument('--reset_statistical_model', type=int, default=0)
    parser.add_argument('--regression_model', type=str, default='FSVGD')
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    args = parser.parse_args()
    main(args)
