import argparse
ENTITY = 'cbiel01'

# Logging mode:
# 0 - No logging
# 1 - Logging learning metrics
# 2 - Additional Logging of performance plots
# 3 - Additional Logging of Dyn. Model Learning metrics
# 4 - Additional Logging of offline data

def experiment(project_name: str = 'ICEM_Pendulum',
               environment: str = 'pendulum',
               num_offline_samples: int = 0,
               optimizer_horizon: int = 20,
               num_online_samples: int = 200,
               deterministic_policy_for_data_collection: bool = False,
               noise_level: float | None = None,
               icem_num_particles: int = 1,
               icem_num_steps: int = 10,
               icem_num_samples: int = 1000,
               icem_num_elites: int = 100,
               icem_colored_noise_exponent: float = 3.0,
               reward_source: str = 'gym',
               seed: int = 42,
               num_episodes: int = 20,
               bnn_steps: int = 50_000,
               bnn_use_schedule: bool = True,
               bnn_features: tuple = (256,) * 2,
               bnn_train_share: float = 0.8,
               bnn_weight_decay: float = 1e-4,
               first_episode_for_policy_training: int = -1,
               exploration: str = 'pets',
               reset_statistical_model: bool = True,
               regression_model: str = 'probabilistic_ensemble',
               beta: float = 2.0,
               smoother_steps: int = 16_000,
               smoother_features: tuple = (64, 64),
               smoother_train_share: float = 1.0,
               smoother_weight_decay: float = 1e-4,
               state_data_source: str = 'smoother',
               measurement_dt_ratio: int = 1,
               load_offline_data: str = None,
               log_mode: int = 2,
               ):
    
    import chex
    import jax
    import jax.numpy as jnp
    import numpy as np
    import jax.random as jr
    import optax
    import wandb

    from bsm.bayesian_regression.bayesian_neural_networks.deterministic_ensembles import DeterministicEnsemble
    from bsm.bayesian_regression import ProbabilisticEnsemble, DeterministicFSVGDEnsemble, ProbabilisticFSVGDEnsemble
    from bsm.statistical_model.bnn_statistical_model import BNNStatisticalModel
    from distrax import Normal

    from mbpo.optimizers.trajectory_optimizers.icem_brax_wrapper import iCEMOptimizer
    from mbpo.optimizers.trajectory_optimizers.icem_optimizer import iCemParams
    from mbpo.systems.rewards.base_rewards import Reward, RewardParams

    from diff_smoothers.BNN_Differentiator import BNNSmootherDifferentiator
    
    jax.config.update('jax_enable_x64', True)

    assert exploration in ['optimistic', 'mean',
                           'pets'], "Unrecognized exploration strategy, should be 'optimistic' or 'pets' or 'mean'"
    assert regression_model in ['probabilistic_ensemble', 'deterministic_ensemble', 'deterministic_FSVGD', 'probabilistic_FSVGD']
    assert reward_source in ['dm-control', 'gym']
    assert environment in ['pendulum', 'cartpole', 'bicycle', 'rccar']
    assert state_data_source in ['discrete', 'smoother', 'true']

    # ------------------------------------------------------------------
    # ------------------ Environment and Reward Setup ------------------    

    if environment == 'pendulum':
        from mbrl.envs.pendulum_ct import ContinuousPendulumEnv
        if noise_level is not None:
            noise_level = jnp.array([0.05, 0.1]) * noise_level
            env = ContinuousPendulumEnv(reward_source=reward_source,
                                        noise_level=jnp.array(noise_level),
                                        init_noise_key=jr.PRNGKey(seed=seed*2))
        else:
            env = ContinuousPendulumEnv(reward_source=reward_source)
        eval_env = ContinuousPendulumEnv(reward_source=reward_source)

        init_state_range = jnp.array([[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])

        class DMPendulumReward(Reward):
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
            
        class GymPendulumReward(Reward):
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
                reward = -(env.reward_params.angle_cost * diff_th ** 2 +
                            0.1 * omega ** 2) - env.reward_params.control_cost * u ** 2
                reward = reward.squeeze()
                reward_dist = Normal(reward, jnp.zeros_like(reward))
                return reward_dist, reward_params

            def init_params(self, key: chex.PRNGKey) -> RewardParams:
                return {'dt': env.dt}

        if reward_source == 'dm-control':
            reward_model = DMPendulumReward()
        elif reward_source == 'gym':
            reward_model = GymPendulumReward()
        else:   
            raise NotImplementedError(f'Unknown reward source {reward_source}')

    elif environment == 'cartpole':
        from mbrl.envs.cartpole import ContinuousCartpoleEnv
        if noise_level is not None:
            noise_level = jnp.array([0.2, 0.05, 0.1, 0.1]) * noise_level
            env = ContinuousCartpoleEnv(reward_source=reward_source,
                                        noise_level=jnp.array(noise_level),
                                        init_noise_key=jr.PRNGKey(seed=seed*2))
        else:
            env = ContinuousCartpoleEnv(reward_source=reward_source)
        eval_env = ContinuousCartpoleEnv(reward_source=reward_source)

        init_state_range = jnp.array([[0.0, -1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0, 0.0]])
            
        class GymCartpoleReward(Reward):
            def __init__(self):
                super().__init__(x_dim=3, u_dim=1)

            def __call__(self,
                        x: chex.Array,
                        u: chex.Array,
                        reward_params: RewardParams,
                        x_next: chex.Array | None = None
                        ):
                assert x.shape == (5,) and u.shape == (1,)
                theta, omega = jnp.arctan2(x[2], x[1]), x[-1]
                target_angle = env.reward_params.target_angle
                pos, vel = x[0], x[3]
                diff_th = theta - target_angle
                diff_th = ((diff_th + jnp.pi) % (2 * jnp.pi)) - jnp.pi
                reward = -(env.reward_params.angle_cost * diff_th ** 2 +
                           env.reward_params.pos_cost * pos ** 2 +
                           0.1 * omega ** 2+ 
                           0.1 * vel ** 2) - env.reward_params.control_cost * u ** 2
                reward = reward.squeeze()
                reward_dist = Normal(reward, jnp.zeros_like(reward))
                return reward_dist, reward_params

            def init_params(self, key: chex.PRNGKey) -> RewardParams:
                return {'dt': env.dt}

        if reward_source == 'dm-control':
            raise NotImplementedError('DM-Control reward not implemented for Cartpole')
        elif reward_source == 'gym':
            reward_model = GymCartpoleReward()
        else:   
            raise NotImplementedError(f'Unknown reward source {reward_source}')

    elif environment == 'bicycle':
        from mbrl.envs.bicyclecar import BicycleEnv
        if noise_level is not None:
            use_obs_noise = True
        else:
            use_obs_noise = False
        env = BicycleEnv(init_noise_key=jr.PRNGKey(seed=seed*2),
                         use_obs_noise=use_obs_noise)
        eval_env = BicycleEnv(reward_source=reward_source)

        init_state_range = jnp.concatenate([env.reset().pipeline_state.reshape(1, -1),
                                            env.reset().pipeline_state.reshape(1, -1)], axis=0)
        
        from mbrl.utils.bicyclecar_utils import BicycleCarReward
        class BicycleCarEnvReward(BicycleCarReward):
            def __init__(self, action_cost: float = 0.0):
                super().__init__(goal = env.goal,
                                action_cost = action_cost,)
            
            def __call__(self,
                         x: chex.Array,
                         u: chex.Array,
                         reward_params: RewardParams,
                         x_next: chex.Array | None = None):
                """ Computes the reward for the given transition """
                reward = self.predict(x, u)
                reward_dist = Normal(reward, jnp.zeros_like(reward))
                return reward_dist, reward_params
        
            def init_params(self, key: chex.PRNGKey) -> RewardParams:
                return {'dt': env.dt}
        
        reward_model = BicycleCarEnvReward()

    elif environment == 'rccar':
        from mbrl.envs.rccar import RCCarSimEnv
        if noise_level is not None:
            use_obs_noise = True
        else:
            use_obs_noise = False

        # Set environment specific parameters
        margin_factor = 20
        num_online_samples = 200

        env = RCCarSimEnv(seed=seed*2,
                          use_obs_noise=use_obs_noise,
                          encode_angle=True,
                          use_tire_model=True,
                          margin_factor=margin_factor)
        eval_env = RCCarSimEnv(use_obs_noise=False,
                               encode_angle=True,
                               use_tire_model=True,
                               margin_factor=margin_factor)
        
        init_state_range = jnp.concatenate([env.reset().pipeline_state.reshape(1, -1),
                                            env.reset().pipeline_state.reshape(1, -1)], axis=0)
        
        
        from mbrl.utils.rccar_utils import RCCarEnvReward
        class RCCarReward(RCCarEnvReward):
            def __init__(self,
                        ctrl_cost_weight: float,
                        encode_angle: bool,
                        margin_factor: float):
                super().__init__(goal=env._goal,
                                ctrl_cost_weight=ctrl_cost_weight,
                                encode_angle=encode_angle,
                                margin_factor=margin_factor)
                
            def __call__(self,
                        x: chex.Array,
                        u: chex.Array,
                        reward_params: RewardParams,
                        x_next: chex.Array | None = None):
                """ Computes the reward for the given transition """
                reward = self.forward(x, u, x_next)
                reward_dist = Normal(reward, jnp.zeros_like(reward))
                return reward_dist, reward_params
            
            def init_params(self, key: chex.PRNGKey) -> RewardParams:
                return {'dt': env.dt}
            
        reward_model = RCCarReward(ctrl_cost_weight=env._reward_model.ctrl_cost_weight,
                                   encode_angle=env._reward_model.encode_angle,
                                   margin_factor=margin_factor)

    else:
        raise NotImplementedError(f'Unknown environment {environment}')
    
    # ------------------------------------------------------------------
    # ------------------   Statistical  Model  Setup  ------------------

    if bnn_use_schedule:
        bnn_schedule = optax.linear_schedule(
            init_value=bnn_steps/4,
            end_value=bnn_steps,
            transition_steps=2000,
        )
    else:
        bnn_schedule = optax.constant_schedule(bnn_steps)

    if regression_model == 'probabilistic_ensemble':
        model = BNNStatisticalModel(
            input_dim=env.observation_size + env.action_size,
            output_dim=env.observation_size,
            num_training_steps=bnn_schedule,
            output_stds=1e-3 * jnp.ones(env.observation_size),
            beta=beta * jnp.ones(shape=(env.observation_size,)),
            features=bnn_features,
            bnn_type=ProbabilisticEnsemble,
            num_particles=10,
            logging_wandb=log_mode > 2,
            return_best_model=True,
            eval_batch_size=64,
            train_share=bnn_train_share,
            eval_frequency=5_000,
            weight_decay=bnn_weight_decay,
        )
    elif regression_model == 'deterministic_ensemble':
        model = BNNStatisticalModel(
            input_dim=env.observation_size + env.action_size,
            output_dim=env.observation_size,
            num_training_steps=bnn_schedule,
            output_stds=1e-3 * jnp.ones(env.observation_size),
            beta=beta * jnp.ones(shape=(env.observation_size,)),
            features=bnn_features,
            num_particles=10,
            logging_wandb=log_mode > 2,
            return_best_model=True,
            eval_batch_size=64,
            train_share=bnn_train_share,
            eval_frequency=5_000,
            weight_decay=bnn_weight_decay,
        )
    elif regression_model == 'deterministic_FSVGD':
        # For optimistic case: Tune beta and stuff
        model = BNNStatisticalModel(
            input_dim=env.observation_size + env.action_size,
            output_dim=env.observation_size,
            num_training_steps=bnn_schedule,
            output_stds=1e-3 * jnp.ones(env.observation_size),
            beta=beta * jnp.ones(shape=(env.observation_size,)),
            features=bnn_features,
            bnn_type=DeterministicFSVGDEnsemble,
            num_particles=10,
            logging_wandb=log_mode > 2,
            return_best_model=True,
            eval_batch_size=64,
            train_share=bnn_train_share,
            eval_frequency=5_000,
            weight_decay=bnn_weight_decay,
        )
    elif regression_model == 'probabilistic_FSVGD':
        # For optimistic case: Tune beta and stuff
        model = BNNStatisticalModel(
            input_dim=env.observation_size + env.action_size,
            output_dim=env.observation_size,
            num_training_steps=bnn_schedule,
            output_stds=1e-3 * jnp.ones(env.observation_size),
            beta=beta * jnp.ones(shape=(env.observation_size,)),
            features=bnn_features,
            bnn_type=ProbabilisticFSVGDEnsemble,
            num_particles=10,
            logging_wandb=log_mode > 2,
            return_best_model=True,
            eval_batch_size=64,
            train_share=bnn_train_share,
            eval_frequency=5_000,
            weight_decay=bnn_weight_decay,
        )
    else:
        raise NotImplementedError(f'Unknown regression model {regression_model}')

    discount_factor = 0.99

    BNN_Differentiator = BNNSmootherDifferentiator(state_dim=env.observation_size,
                                               output_stds=jnp.ones(shape=(env.observation_size,)) * 0.1,
                                               logging_wandb=False,
                                               beta=jnp.ones(shape=(env.observation_size,))*2,
                                               num_particles=5,
                                               features=smoother_features,
                                               bnn_type=DeterministicEnsemble,
                                               train_share=smoother_train_share,
                                               num_training_steps=smoother_steps,
                                               weight_decay=smoother_weight_decay,
                                               return_best_model=True,
                                               eval_frequency=1_000,)


    # ------------------------------------------------------------------
    # ------------------   Offline  Data  Generation  ------------------

    key_offline_data, key_agent = jr.split(jr.PRNGKey(seed))
    if num_offline_samples > 0:
        from mbrl.utils.offline_data import DifferentiatorOfflineData, load_transitions
        offline_data_gen = DifferentiatorOfflineData(differentiator=BNN_Differentiator,
                                                     env=env,
                                                     init_state_range=init_state_range,)
        if load_offline_data is not None:
            offline_data = load_transitions(filename=load_offline_data)
        else:
            offline_data = offline_data_gen.sample_transitions(key=key_offline_data,
                                                               num_samples=num_offline_samples,
                                                               trajectory_length=num_online_samples,
                                                               plot_results=log_mode > 3,
                                                               measurement_dt_ratio=measurement_dt_ratio,
                                                               state_data_source=state_data_source)
    else:
        offline_data = None


    # ------------------------------------------------------------------
    # -------------------------  Agent  Setup  -------------------------

    max_replay_size_true_data_buffer = 10 ** 4
    # Define extra fields for the state
    if state_data_source == 'discrete':
        extra_fields = ('derivative', 't', 'dt')
        extra_fields_shape = (env.observation_size,) + (1,) * 2
        state_extras: dict = {x: jnp.zeros(shape=(y,)) for x, y in zip(extra_fields, extra_fields_shape)}
    elif state_data_source == 'smoother' or 'true':
        extra_fields = ('derivative', 'true_derivative', 't', 'dt')
        extra_fields_shape = (env.observation_size,) * 2 + (1,) * 2
        state_extras: dict = {x: jnp.zeros(shape=(y,)) for x, y in zip(extra_fields, extra_fields_shape)}

    # Change the optimizer parameters based on the environments
    if environment == 'rccar':
        icem_num_steps = 30
        opt_params = iCemParams(
            num_particles=icem_num_particles,
            num_steps=icem_num_steps,
            num_samples=icem_num_samples,
            num_elites=icem_num_elites,
            exponent=icem_colored_noise_exponent)
        optimizer_horizon = 55

    elif environment == 'pendulum':
        icem_num_steps = 10
        opt_params = iCemParams(
            num_particles=icem_num_particles,
            num_steps=icem_num_steps,
            num_samples=icem_num_samples,
            num_elites=icem_num_elites,
            exponent=icem_colored_noise_exponent)
        optimizer_horizon = 20
    
    elif environment == 'cartpole':
        icem_num_steps = 10
        opt_params = iCemParams(
            num_particles=icem_num_particles,
            num_steps=icem_num_steps,
            num_samples=icem_num_samples,
            num_elites=icem_num_elites,
            exponent=icem_colored_noise_exponent)
        optimizer_horizon = 20

    else:
        raise NotImplementedError(f'Unknown environment {environment}')
        
    optimizer = iCEMOptimizer(horizon=optimizer_horizon,
                              key = jr.PRNGKey(seed),
                              opt_params=opt_params,
                              )

    agent_class = None
    # Define agent class based on the state data source and exploration strategy
    if state_data_source == 'discrete':
        if exploration == 'optimistic':
            from mbrl.model_based_agent import OptimisticModelBasedAgent
            agent_class = OptimisticModelBasedAgent
        elif exploration == 'pets':
            from mbrl.model_based_agent import PETSModelBasedAgent
            agent_class = PETSModelBasedAgent
        elif exploration == 'mean':
            from mbrl.model_based_agent import MeanModelBasedAgent
            agent_class = MeanModelBasedAgent
        else:
            raise NotImplementedError(f'Unknown exploration strategy {exploration}')
    elif state_data_source == 'smoother' or 'true':
        if exploration == 'optimistic':
            from mbrl.model_based_agent import ContinuousOptimisticModelBasedAgent
            agent_class = ContinuousOptimisticModelBasedAgent
        elif exploration == 'pets':
            from mbrl.model_based_agent import ContinuousPETSModelBasedAgent
            agent_class = ContinuousPETSModelBasedAgent
        elif exploration == 'mean':
            from mbrl.model_based_agent import ContinuousMeanModelBasedAgent
            agent_class = ContinuousMeanModelBasedAgent
        else:
            raise NotImplementedError(f'Unknown exploration strategy {exploration}')

    agent_kwargs = {
        'env': env,
        'eval_env': eval_env,
        'statistical_model': model,
        'optimizer': optimizer,
        'episode_length': num_online_samples,
        'reward_model': reward_model,
        'offline_data': offline_data,
        'num_envs': 1,
        'num_eval_envs': 1,
        'log_mode': log_mode,
        'deterministic_policy_for_data_collection': deterministic_policy_for_data_collection,
        'first_episode_for_policy_training': first_episode_for_policy_training,
        'predict_difference': True if state_data_source == 'discrete' else False,
        'reset_statistical_model': reset_statistical_model,
        'dt': env.dt,
        'dynamics_dt': env.dt*measurement_dt_ratio if state_data_source == 'discrete' else env.dt,
        'state_extras_ref': state_extras,
        'measurement_dt_ratio': measurement_dt_ratio,
    }

    # ------------------------------------------------------------------
    # -----------------------   Logging  Setup   -----------------------

    config = dict(environment=environment,
                  num_offline_samples=num_offline_samples,
                  sample_horizon=num_online_samples,
                  optimizer_horizon=optimizer_horizon,
                  deterministic_policy_for_data_collection=deterministic_policy_for_data_collection,
                  noise_level=noise_level,
                  icem_num_particles=icem_num_particles,
                  icem_num_steps=icem_num_steps,
                  icem_num_samples=icem_num_samples,
                  icem_num_elites=icem_num_elites,
                  icem_colored_noise_exponent=icem_colored_noise_exponent,
                  reward_source=reward_source,
                  seed=seed,
                  num_episodes=num_episodes,
                  bnn_steps=bnn_steps,
                  bnn_use_schedule=bnn_use_schedule,
                  bnn_features=bnn_features,
                  bnn_train_share=bnn_train_share,
                  bnn_weight_decay=bnn_weight_decay,
                  exploration=exploration,
                  reset_statistical_model=reset_statistical_model,
                  regression_model=regression_model,
                  beta=beta,
                  smoother_steps=smoother_steps,
                  smoother_features=smoother_features,
                  smoother_train_share=smoother_train_share,
                  smoother_weight_decay=smoother_weight_decay,
                  state_data_source=state_data_source,
                  measurement_dt_ratio=measurement_dt_ratio,
                  load_offline_data=load_offline_data,
                  agent_kwargs=agent_kwargs,
                  )
    
    if log_mode > 0:
        wandb.init(project=project_name,
                   dir='/cluster/scratch/' + ENTITY,
                   config=config)


    # ------------------------------------------------------------------
    # -----------------------   Run Experiment   -----------------------

    # Initialize the agent
    if state_data_source == 'discrete':
        base_agent = agent_class(**agent_kwargs)
    elif state_data_source == 'smoother' or 'true':
        from mbrl.model_based_agent.differentiating_agent import DifferentiatingAgent
        base_agent = DifferentiatingAgent(agent_type=agent_class,
                                          differentiator=BNN_Differentiator,
                                          state_data_source=state_data_source,
                                          **agent_kwargs)
    else:
        raise NotImplementedError(f'Unknown state data source {state_data_source}')
    
    agent_state = base_agent.run_episodes(num_episodes=num_episodes,
                                          start_from_scratch=True,
                                          key=key_agent)
    wandb.finish()


def main(args):
    experiment(project_name=args.project_name,
               environment=args.environment,
               num_offline_samples=args.num_offline_samples,
               deterministic_policy_for_data_collection=bool(args.deterministic_policy_for_data_collection),
               noise_level=args.noise_level,
               icem_colored_noise_exponent=args.icem_colored_noise_exponent,
               reward_source=args.reward_source,
               seed=args.seed,
               num_episodes=args.num_episodes,
               bnn_steps=args.bnn_steps,
               bnn_features=args.bnn_features,
               bnn_train_share=args.bnn_train_share,
               bnn_weight_decay=args.bnn_weight_decay,
               exploration=args.exploration,
               reset_statistical_model=bool(args.reset_statistical_model),
               regression_model=args.regression_model,
               beta=args.beta,
               smoother_steps=args.smoother_steps,
               smoother_features=args.smoother_features,
               smoother_train_share=args.smoother_train_share,
               smoother_weight_decay=args.smoother_weight_decay,
               state_data_source=args.state_data_source,
               measurement_dt_ratio=args.measurement_dt_ratio,
               log_mode=args.log_mode,
               )


if __name__ == '__main__':

    def underscore_to_tuple(value: str):
        return tuple(map(int, value.split('_')))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='ICEM_Diff_General')
    parser.add_argument('--environment', type=str, default='pendulum')
    parser.add_argument('--num_offline_samples', type=int, default=0)
    parser.add_argument('--deterministic_policy_for_data_collection', type=int, default=1)
    parser.add_argument('--noise_level', type=float, default=1.0)
    parser.add_argument('--icem_colored_noise_exponent', type=float, default=2.0)
    parser.add_argument('--reward_source', type=str, default='gym')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_episodes', type=int, default=30)
    parser.add_argument('--bnn_steps', type=int, default=48_000)
    parser.add_argument('--bnn_features', type=underscore_to_tuple, default='64_64')
    parser.add_argument('--bnn_train_share', type=float, default=0.8)
    parser.add_argument('--bnn_weight_decay', type=float, default=1e-4)
    parser.add_argument('--exploration', type=str, default='pets')
    parser.add_argument('--reset_statistical_model', type=int, default=0)
    parser.add_argument('--regression_model', type=str, default='probabilistic_ensemble')
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--smoother_steps', type=int, default=64_000)
    parser.add_argument('--smoother_features', type=underscore_to_tuple, default='64_64_64')
    parser.add_argument('--smoother_train_share', type=float, default=1.0)
    parser.add_argument('--smoother_weight_decay', type=float, default=1e-4)
    parser.add_argument('--state_data_source', type=str, default='discrete')
    parser.add_argument('--measurement_dt_ratio', type=int, default=1)
    parser.add_argument('--log_mode', type=int, default=2)

    args = parser.parse_args()
    main(args)
