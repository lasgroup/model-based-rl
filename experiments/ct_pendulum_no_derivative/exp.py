import argparse
#import os
#os.environ['JAX_PLATFORMS'] = 'cpu'

ENTITY = 'cbiel01'

# Logging mode:
# 0 - No logging
# 1 - Logging learning metrics
# 2 - Additional Logging of performance plots
# 3 - Additional Logging of Dyn. Model Learning metrics
# 4 - Additional Logging of offline data

def experiment(project_name: str = 'CT_Pendulum',
               num_offline_samples: int = 0,
               sac_horizon: int = 100,
               num_online_samples: int = 100,
               deterministic_policy_for_data_collection: bool = False,
               seed: int = 42,
               num_episodes: int = 20,
               sac_steps: int = 500_000,
               bnn_steps: int = 50_000,
               bnn_use_schedule: bool = True,
               bnn_features: tuple = (256,) * 2,
               bnn_train_share: float = 0.8,
               bnn_weight_decay: float = 1e-4,
               first_episode_for_policy_training: int = -1,
               exploration: str = 'pets',  # Should be one of the ['optimistic', 'pets', 'mean'],
               reset_statistical_model: bool = True,
               regression_model: str = 'probabilistic_ensemble',
               beta: float = 2.0,
               smoother_steps: int = 16_000,
               smoother_features: tuple = (64, 64),
               smoother_train_share: float = 1.0,
               smoother_weight_decay: float = 1e-4,
               log_mode: int = 2,
               ):
    
    import chex
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import optax
    import wandb
    from brax.training.replay_buffers import UniformSamplingQueue
    from brax.training.types import Transition
    from bsm.bayesian_regression.bayesian_neural_networks.deterministic_ensembles import DeterministicEnsemble
    from bsm.bayesian_regression import ProbabilisticEnsemble, DeterministicFSVGDEnsemble, ProbabilisticFSVGDEnsemble
    from bsm.statistical_model.bnn_statistical_model import BNNStatisticalModel
    from bsm.statistical_model.gp_statistical_model import GPStatisticalModel
    from distrax import Normal
    from jax.nn import swish
    from mbpo.optimizers import SACOptimizer
    from mbpo.systems.rewards.base_rewards import Reward, RewardParams

    from mbrl.envs.pendulum_ct import ContinuousPendulumEnv
    from mbrl.model_based_agent import ContinuousPETSModelBasedAgent, ContinuousOptimisticModelBasedAgent
    from mbrl.model_based_agent.Smoother_Wrapper import SmootherWrapper
    from mbrl.utils.offline_data import SmootherPendulumOfflineData
    from diff_smoothers.smoother_net import SmootherNet
    
    jax.config.update('jax_enable_x64', True)

    assert exploration in ['optimistic',
                           'pets'], "Unrecognized exploration strategy, should be 'optimistic' or 'pets' or 'mean'"
    assert regression_model in ['probabilistic_ensemble', 'deterministic_ensemble', 'deterministic_FSVGD', 'probabilistic_FSVGD', 'GP']

    env = ContinuousPendulumEnv(reward_source='dm-control')

    # Create the BNN num_training_steps schedule
    if bnn_use_schedule:
        bnn_steps = optax.piecewise_constant_schedule(
            init_value=bnn_steps/8,
            boundaries_and_scales={500: 2, 1_000: 2, 4_000: 2},
        )

    else:
        bnn_steps = optax.constant_schedule(bnn_steps)

    if regression_model == 'probabilistic_ensemble':
        model = BNNStatisticalModel(
            input_dim=env.observation_size + env.action_size,
            output_dim=env.observation_size,
            num_training_steps=bnn_steps,
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
            num_training_steps=bnn_steps,
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
            num_training_steps=bnn_steps,
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
            num_training_steps=bnn_steps,
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

    smoother_model = SmootherNet(input_dim=1,
                            output_dim=env.observation_size,
                            output_stds=jnp.ones(shape=(env.observation_size,)) * 0.001,
                            logging_wandb=False,
                            beta=jnp.ones(shape=(env.observation_size,))*3,
                            num_particles=5,
                            features=smoother_features,
                            bnn_type=DeterministicEnsemble,
                            train_share=smoother_train_share,
                            num_training_steps=smoother_steps,
                            weight_decay=smoother_weight_decay,
                            return_best_model=True,
                            eval_frequency=1_000,
                            )

    offline_data_gen = SmootherPendulumOfflineData(smoother_net=smoother_model)
    key_offline_data, key_agent = jr.split(jr.PRNGKey(seed))
    if num_offline_samples > 0:
        offline_data = offline_data_gen.sample_transitions(key=key_offline_data,
                                                           num_samples=num_offline_samples,
                                                           trajectory_length=num_online_samples,
                                                           plot_results=log_mode > 3)
    else:
        offline_data = None

    num_envs = 64
    num_env_steps_between_updates = 20
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
        'batch_size': 32,
        'num_evals': 20,
        'normalize_observations': True,
        'reward_scaling': 1.,
        'tau': 0.005,
        'min_replay_size': 10 ** 4,
        'max_replay_size': 10 ** 5,
        'grad_updates_per_step': num_envs * num_env_steps_between_updates,
        'deterministic_eval': True,
        'init_log_alpha': 0.,
        'policy_hidden_layer_sizes': (32,) * 5,
        'policy_activation': swish,
        'critic_hidden_layer_sizes': (128,) * 4,
        'critic_activation': swish,
        'wandb_logging': log_mode > 1,
        'return_best_model': True,
    }

    max_replay_size_true_data_buffer = 10 ** 4

    extra_fields = ('derivative', 'true_derivative', 't', 'dt')
    extra_fields_shape = (env.observation_size,) * 2 + (1,) * 2
    state_extras: dict = {x: jnp.zeros(shape=(y,)) for x, y in zip(extra_fields, extra_fields_shape)}

    dummy_sample = Transition(observation=jnp.ones(env.observation_size),
                              action=jnp.zeros(shape=(env.action_size,)),
                              reward=jnp.array(0.0),
                              discount=jnp.array(discount_factor),
                              next_observation=jnp.ones(env.observation_size),
                              extras={'state_extras': state_extras})

    sac_buffer = UniformSamplingQueue(
        max_replay_size=max_replay_size_true_data_buffer,
        dummy_data_sample=dummy_sample,
        sample_batch_size=1)

    optimizer = SACOptimizer(system=None,
                             true_buffer=sac_buffer,
                             **sac_kwargs)

    agent_class = None
    if exploration == 'optimistic':
        agent_class = ContinuousOptimisticModelBasedAgent
    elif exploration == 'pets':
        agent_class = ContinuousPETSModelBasedAgent

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

    sac_learning_schedule = {
        first_episode_for_policy_training: 20_000,
        first_episode_for_policy_training + 2: 50_000,
        first_episode_for_policy_training + 4: 100_000,
        first_episode_for_policy_training + 8: 200_000,
        first_episode_for_policy_training + 12: sac_steps,
    }
    for key in sac_learning_schedule:
        if sac_learning_schedule[key] < 2*sac_kwargs['min_replay_size']:
        # '2 times' since due to the environment number it has to sometimes do more steps than min_replay_size
        # -> to make sure it works
            raise ValueError(f"min_replay_size should be at least 2*{sac_learning_schedule[key]}")
        
    agent_kwargs = {
        'env': env,
        'eval_env': env,
        'statistical_model': model,
        'optimizer': optimizer,
        'episode_length': num_online_samples,
        'reward_model': PendulumReward(),
        'offline_data': offline_data,
        'num_envs': 1,
        'num_eval_envs': 1,
        'log_mode': log_mode,
        'deterministic_policy_for_data_collection': deterministic_policy_for_data_collection,
        'first_episode_for_policy_training': first_episode_for_policy_training,
        'predict_difference': False,
        'reset_statistical_model': reset_statistical_model,
        'dt': env.dt,
        'state_extras_ref': state_extras,
        'actor_learning_schedule': sac_learning_schedule,
    }

    config = dict(num_offline_samples=num_offline_samples,
                  sample_horizon=num_online_samples,
                  sac_horizon=sac_horizon,
                  deterministic_policy_for_data_collection=deterministic_policy_for_data_collection,
                  seed=seed,
                  num_episodes=num_episodes,
                  sac_steps=sac_steps,
                  bnn_steps=bnn_steps,
                  bnn_features=bnn_features,
                  bnn_train_share=bnn_train_share,
                  bnn_weight_decay=bnn_weight_decay,
                  first_episode_for_policy_training=first_episode_for_policy_training,
                  exploration=exploration,
                  reset_statistical_model=reset_statistical_model,
                  regression_model=regression_model,
                  beta=beta,
                  smoother_steps=smoother_steps,
                  smoother_features=smoother_features,
                  smoother_train_share=smoother_train_share,
                  smoother_weight_decay=smoother_weight_decay,
                  sac_kwargs=sac_kwargs,
                  agent_kwargs=agent_kwargs,
                  )
    
    if log_mode > 0:
        wandb.init(project=project_name,
                   dir='/cluster/scratch/' + ENTITY,
                   config=config)

    base_agent = SmootherWrapper(agent_type=agent_class,
                                 smoother_net=smoother_model,
                                 state_data_source='smoother',
                                 **agent_kwargs)

    agent_state = base_agent.run_episodes(num_episodes=num_episodes,
                                          start_from_scratch=True,
                                          key=key_agent)
    wandb.finish()


def main(args):
    experiment(project_name=args.project_name,
               num_offline_samples=args.num_offline_samples,
               sac_horizon=args.sac_horizon,
               num_online_samples=args.num_online_samples,
               deterministic_policy_for_data_collection=bool(args.deterministic_policy_for_data_collection),
               seed=args.seed,
               num_episodes=args.num_episodes,
               sac_steps=args.sac_steps,
               bnn_steps=args.bnn_steps,
               bnn_features=args.bnn_features,
               bnn_train_share=args.bnn_train_share,
               bnn_weight_decay=args.bnn_weight_decay,
               first_episode_for_policy_training=args.first_episode_for_policy_training,
               exploration=args.exploration,
               reset_statistical_model=bool(args.reset_statistical_model),
               regression_model=args.regression_model,
               beta=args.beta,
               smoother_steps=args.smoother_steps,
               smoother_features=args.smoother_features,
               smoother_train_share=args.smoother_train_share,
               smoother_weight_decay=args.smoother_weight_decay,
               log_mode=args.log_mode,
               )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='CT_Pendulum_Debug')
    parser.add_argument('--num_offline_samples', type=int, default=0) # has to be multiple of num_online_samples
    parser.add_argument('--sac_horizon', type=int, default=100)
    parser.add_argument('--num_online_samples', type=int, default=200)
    parser.add_argument('--deterministic_policy_for_data_collection', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_episodes', type=int, default=30)
    parser.add_argument('--sac_steps', type=int, default=400_000)
    parser.add_argument('--bnn_steps', type=int, default=64_000)
    parser.add_argument('--bnn_features', type=tuple, default=(128, 128, 64))
    parser.add_argument('--bnn_train_share', type=float, default=0.8)
    parser.add_argument('--bnn_weight_decay', type=float, default=0.0)
    parser.add_argument('--first_episode_for_policy_training', type=int, default=1)
    parser.add_argument('--exploration', type=str, default='optimistic')
    parser.add_argument('--reset_statistical_model', type=int, default=1)
    parser.add_argument('--regression_model', type=str, default='probabilistic_ensemble')
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--smoother_steps', type=int, default=72_000)
    parser.add_argument('--smoother_features', type=tuple, default=(64, 64, 64))
    parser.add_argument('--smoother_train_share', type=float, default=1.0)
    parser.add_argument('--smoother_weight_decay', type=float, default=0.0)
    parser.add_argument('--log_mode', type=int, default=3)

    args = parser.parse_args()
    main(args)
