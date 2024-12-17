import argparse

def experiment(
        seed: int = 0,
        project_name: str = 'CT_Pendulum',
        entity: str = None,
        num_offline_samples: int = 0,
        num_online_samples: int = 100,
        deterministic_policy_for_data_collection: bool = False,
        noise_level: list = [0.1, 0.1], # TODO: This is from Chris
        reward_source: str = 'dm-control',
        control_cost: float = 0.02,
        num_episodes: int = 5,
        bnn_steps: int = 5_000,
        first_episode_for_policy_training: int = -1,
        exploration: str = 'optimistic',
        reset_statistical_model: bool = False,
        regression_model: str = 'probabilistic_ensemble',
        beta: float = 2.0,
        weight_decay: float = 0.0,
        env_name: str = 'swing-up',
        eval_env_name: str = 'swing-up',
        optimizer: str = 'icem',
        train_steps_sac: int = 500,
        optimizer_horizon: int = 100,
        icem_num_steps: int = 10,
        icem_colored_noise_exponent: float = 3.0
        ):
    
    import chex
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import wandb
    from brax.training.replay_buffers import UniformSamplingQueue
    from brax.training.types import Transition
    from bsm.bayesian_regression import ProbabilisticEnsemble, DeterministicFSVGDEnsemble, ProbabilisticFSVGDEnsemble
    from bsm.statistical_model.bnn_statistical_model import BNNStatisticalModel
    from bsm.statistical_model.gp_statistical_model import GPStatisticalModel
    from distrax import Normal
    from jax.nn import swish
    from mbpo.optimizers import SACOptimizer, iCemParams, iCEMOptimizer
    from mbpo.systems.rewards.base_rewards import Reward, RewardParams
    from mbrl.envs.pendulum_ct import ContinuousPendulumEnv
    from mbrl.model_based_agent import ContinuousPETSModelBasedAgent, ContinuousOptimisticModelBasedAgent, ContinuousMeanModelBasedAgent

    log_wandb = True
    # jax.config.update('jax_log_compiles', True)
    jax.config.update('jax_enable_x64', True)

    assert exploration in ['optimistic', 'mean',
                           'pets'], "Unrecognized exploration strategy, should be 'optimistic' or 'pets' or 'mean'"
    assert regression_model in ['probabilistic_ensemble', 'deterministic_ensemble', 'deterministic_FSVGD', 'probabilistic_FSVGD', 'GP']
    assert reward_source in ['dm-control', 'gym']
    assert env_name in ['swing-up', 'balance']
    assert eval_env_name in ['swing-up', 'balance']


    general_config = dict(num_offline_samples=num_offline_samples,
                  num_online_samples=num_online_samples,
                  deterministic_policy_for_data_collection=deterministic_policy_for_data_collection,
                  noise_level=noise_level,
                  seed=seed,
                  num_episodes=num_episodes,
                  reward_source=reward_source,
                  control_cost=control_cost,
                  bnn_steps=bnn_steps,
                  first_episode_for_policy_training=first_episode_for_policy_training,
                  exploration=exploration,
                  reset_statistical_model=reset_statistical_model,
                  regression_model=regression_model,
                  beta=beta,
                  weight_decay=weight_decay,
                  env=env_name,
                  eval_env=eval_env_name
                  )
    
    if optimizer == 'sac':
        config = general_config | dict(
            optimizer='sac',
            train_steps_sac=train_steps_sac
        )
    elif optimizer=='icem':
        config = general_config | dict(
            optimizer='icem',
            optimizer_horizon=optimizer_horizon,
            icem_num_steps=icem_num_steps,
            icem_colored_noise_exponent=icem_colored_noise_exponent
        )

    swing_up_env = ContinuousPendulumEnv(reward_source=reward_source)
    swing_down_params = swing_up_env.reward_params.replace(target_angle=jnp.pi)
    swing_down_env = ContinuousPendulumEnv(reward_source=reward_source)
    swing_down_env.reward_params = swing_down_params
    balance_env = ContinuousPendulumEnv(reward_source=reward_source, initial_angle=0.)

    env_mapping = {
        'swing-up': swing_up_env,
        'swing-down': swing_down_env,
        'balance': balance_env
    }

    env = env_mapping.get(env_name, None)
    eval_env = env_mapping.get(eval_env_name, None)

    # adjust control cost
    control_cost_params = env.reward_params.replace(control_cost=jnp.array(control_cost))
    env.reward_params = control_cost_params
    eval_env.reward_params = control_cost_params

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
            beta=beta * jnp.ones(shape=(env.observation_size,)),
            features=(256,) * 2,
            num_particles=10,
            logging_wandb=log_wandb,
            return_best_model=True,
            eval_batch_size=64,
            train_share=0.8,
            eval_frequency=5_000,
            weight_decay=weight_decay,
        )
    elif regression_model == 'deterministic_FSVGD':
        # For optimistic case: Tune beta and stuff
        model = BNNStatisticalModel(
            input_dim=env.observation_size + env.action_size,
            output_dim=env.observation_size,
            num_training_steps=bnn_steps,
            output_stds=1e-3 * jnp.ones(env.observation_size),
            beta=beta * jnp.ones(shape=(env.observation_size,)),
            features=(64, 64, 64),
            bnn_type=DeterministicFSVGDEnsemble,
            num_particles=10,
            logging_wandb=log_wandb,
            return_best_model=True,
            eval_batch_size=64,
            train_share=0.8,
            eval_frequency=5_000,
            weight_decay=weight_decay,
        )
    elif regression_model == 'probabilistic_FSVGD':
        # For optimistic case: Tune beta and stuff
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
    else:
        raise ValueError(f"Invalid regression model: {regression_model}. Expected 'probabilistic_ensemble', 'deterministic_ensemble' or 'GP'.")


    discount_factor = 0.99

    max_replay_size_true_data_buffer = 10 ** 4

    extra_fields = ('derivative', 't', 'dt')
    extra_fields_shape = (swing_up_env.observation_size, 1, 1)
    # extra_fields_shape = (env.observation_size,) * 1 + (1,) * 2
    state_extras: dict = {x: jnp.zeros(shape=(y,)) for x,y in zip(extra_fields, extra_fields_shape)}

    dummy_sample = Transition(observation=jnp.ones(swing_up_env.observation_size),
                              action=jnp.zeros(shape=(swing_up_env.action_size,)),
                              reward=jnp.array(0.0),
                              discount=jnp.array(0.99),
                              next_observation=jnp.ones(swing_up_env.observation_size),
                              extras={'state_extras': state_extras})

    if optimizer == 'sac':
        sac_kwargs = {
            'num_timesteps': train_steps_sac,
            'episode_length': num_online_samples,
            'num_env_steps_between_updates': 20,
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
            'batch_size': 32,
            'num_evals': 20,
            'normalize_observations': True,
            'reward_scaling': 1.,
            'tau': 0.005,
            'min_replay_size': 10 ** 4,
            'max_replay_size': train_steps_sac,
            'grad_updates_per_step': 20 * 32,  # should be num_envs * num_env_steps_between_updates
            'deterministic_eval': True,
            'init_log_alpha': 0.,
            'policy_hidden_layer_sizes': (32,) * 5,
            'policy_activation': swish,
            'critic_hidden_layer_sizes': (128,) * 4,
            'critic_activation': swish,
            'wandb_logging': log_wandb,
            'return_best_model': True,
            }
        max_replay_size_true_data_buffer = 10 ** 4

        sac_buffer = UniformSamplingQueue(
            max_replay_size=max_replay_size_true_data_buffer,
            dummy_data_sample=dummy_sample,
            sample_batch_size=1)

        optimizer = SACOptimizer(system=None,
                                 true_buffer=sac_buffer,
                                 **sac_kwargs)
        
    elif optimizer == 'icem':
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
                   dir='/cluster/scratch/' + entity,
                   config=config)

    
    # debugging. TODO: REMOVE!
    print("Max speed is: ", env.max_speed)
    
    agent_class = None
    if exploration == 'optimistic':
        agent_class = ContinuousOptimisticModelBasedAgent
    elif exploration == 'pets':
        agent_class = ContinuousPETSModelBasedAgent
    elif exploration == 'mean':
        agent_class = ContinuousMeanModelBasedAgent
    else:
        raise ValueError(f"Invalid agent class: {agent_class}. Check exploration method, got: {exploration}")

    class PendulumReward(Reward):
        def __init__(self, reward_env: ContinuousPendulumEnv, reward_source: str):
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
            elif self.reward_source == 'dm-control':
                reward = self.env.tolerance_reward(jnp.sqrt(self.env.reward_params.angle_cost * diff_th ** 2 +
                                                            0.1 * omega ** 2)) - self.env.reward_params.control_cost * u ** 2
            else:
                raise ValueError(f"Invalid reward source: {self.reward_source}. Expected 'gym' or 'dm-control'.")            
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
        predict_difference=False,
        reset_statistical_model=reset_statistical_model,
        dt=env.dt,
        state_extras_ref=state_extras,
    )

    agent_state = agent.run_episodes(num_episodes=num_episodes,
                                     start_from_scratch=True,
                                     key=key_agent)

    print("Finishing wandb")
    wandb.finish()
    print("Ended wandb")

def main(args):
    experiment(seed=args.seed,
               project_name=args.project_name,
               entity=args.entity,
               num_offline_samples=args.num_offline_samples,
               num_online_samples=args.num_online_samples,
               deterministic_policy_for_data_collection=bool(args.deterministic_policy_for_data_collection),
               noise_level=args.noise_level,
               reward_source=args.reward_source,
               control_cost=args.control_cost,
               num_episodes=args.num_episodes,
               bnn_steps=args.bnn_steps,
               first_episode_for_policy_training=args.first_episode_for_policy_training,
               exploration=args.exploration,
               reset_statistical_model=bool(args.reset_statistical_model),
               regression_model=args.regression_model,
               beta=args.beta,
               weight_decay=args.weight_decay,
               env_name=args.env,
               eval_env_name=args.eval_env,
               optimizer=args.optimizer,
               train_steps_sac=args.train_steps_sac,
               optimizer_horizon=args.optimizer_horizon,
               icem_num_steps=args.icem_num_steps,
               icem_colored_noise_exponent=args.icem_colored_noise_exponent
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--project_name', type=str, default='ICEM_CT_Pendulum')
    parser.add_argument('--entity', type=str, default='kiten')
    parser.add_argument('--num_offline_samples', type=int, default=0)
    parser.add_argument('--num_online_samples', type=int, default=200)
    parser.add_argument('--deterministic_policy_for_data_collection', type=int, default=0)
    parser.add_argument('--noise_level', type=float, nargs=2, default=[0.1, 0.1])
    parser.add_argument('--reward_source', type=str, default='dm-control')
    parser.add_argument('--control_cost', type=float, default=0.02)
    parser.add_argument('--num_episodes', type=int, default=5)
    parser.add_argument('--bnn_steps', type=int, default=5_000)
    parser.add_argument('--first_episode_for_policy_training', type=int, default=-1)
    parser.add_argument('--exploration', type=str, choices=['optimistic', 'pets', 'mean'], default='optimistic')
    parser.add_argument('--reset_statistical_model', type=int, default=0)
    parser.add_argument('--regression_model', type=str, default='probabilistic_ensemble')
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--env', type=str, default='swing-up')
    parser.add_argument('--eval_env', type=str, default='swing-up')
    parser.add_argument('--optimizer', type=str, choices=['sac','icem'], default='icem')
    parser.add_argument('--train_steps_sac', type=int, default=50_000)
    parser.add_argument('--optimizer_horizon', type=int, default=100)
    parser.add_argument('--icem_num_steps', type=int, default=10)
    parser.add_argument('--icem_colored_noise_exponent', type=float, default=3.0)

    args = parser.parse_args()
    main(args)
