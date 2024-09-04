import argparse
ENTITY = 'cbiel01'

# Logging mode:
# 0 - No logging
# 1 - Logging learning metrics
# 2 - Additional Logging of performance plots
# 3 - Additional Logging of Dyn. Model Learning metrics
# 4 - Additional Logging of offline data

def experiment(project_name: str = 'ICEM_CT_RCCar',
               num_offline_samples: int = 0,
               optimizer_horizon: int = 20,
               num_online_samples: int = 200,
               deterministic_policy_for_data_collection: bool = False,
               icem_num_steps: int = 10,
               icem_colored_noise_exponent: float = 3.0,
               seed: int = 42,
               num_episodes: int = 20,
               bnn_steps: int = 50_000,
               bnn_use_schedule: bool = True,
               bnn_features: tuple = (256,) * 2,
               bnn_train_share: float = 0.8,
               bnn_weight_decay: float = 1e-4,
               first_episode_for_policy_training: int = -1,
               exploration: str = 'pets',  # Should be one of the ['optimistic', 'pets', 'mean'],
               reset_statistical_model: bool = True,
               regression_model: str = 'probabilistic_ensemble',
               bnn_dt: float = 0.05,
               beta: float = 2.0,
               smoother_steps: int = 16_000,
               smoother_features: tuple = (64, 64),
               smoother_train_share: float = 1.0,
               smoother_weight_decay: float = 1e-4,
               state_data_source: str = 'smoother',
               log_mode: int = 2,
               ):
    
    import chex
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import optax
    import wandb
    from typing import Tuple
    from brax.training.replay_buffers import UniformSamplingQueue
    from brax.training.types import Transition
    from bsm.bayesian_regression.bayesian_neural_networks.deterministic_ensembles import DeterministicEnsemble
    from bsm.bayesian_regression import ProbabilisticEnsemble, DeterministicFSVGDEnsemble, ProbabilisticFSVGDEnsemble
    from bsm.statistical_model.bnn_statistical_model import BNNStatisticalModel
    from bsm.statistical_model.gp_statistical_model import GPStatisticalModel
    from distrax import Normal
    from jax.nn import swish
    from mbpo.optimizers.trajectory_optimizers.icem_brax_wrapper import iCEMOptimizer
    from mbpo.optimizers.trajectory_optimizers.icem_optimizer import iCemParams
    from mbpo.systems.rewards.base_rewards import Reward, RewardParams

    from mbrl.envs.rccar import RCCarSimEnv
    from mbrl.envs.rccar_utils import decode_angles
    from mbrl.utils.tolerance_reward import ToleranceReward
    from mbrl.model_based_agent import ContinuousPETSModelBasedAgent, ContinuousOptimisticModelBasedAgent
    from mbrl.model_based_agent.differentiating_agent import DifferentiatingAgent
    from diff_smoothers.BNN_Differentiator import BNNSmootherDifferentiator
    
    jax.config.update('jax_enable_x64', True)

    assert exploration in ['optimistic',
                           'pets'], "Unrecognized exploration strategy, should be 'optimistic' or 'pets' or 'mean'"
    assert regression_model in ['probabilistic_ensemble', 'deterministic_ensemble', 'deterministic_FSVGD', 'probabilistic_FSVGD', 'GP']

    env = RCCarSimEnv(encode_angle=True,
                      use_tire_model=True)
    
    eval_env = RCCarSimEnv(encode_angle=True,
                           use_tire_model=True,
                           use_obs_noise=False)

    # Create the BNN num_training_steps schedule
    if bnn_use_schedule:
        # bnn_steps = optax.piecewise_constant_schedule(
        #     init_value=int(bnn_steps/8),
        #     boundaries_and_scales={500: 2, 1_000: 2, 2_000: 2},
        # )
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

    if num_offline_samples > 0:
        raise NotImplementedError('Offline data generation not implemented for cartpole')
    offline_data = None

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

    opt_params = iCemParams(
        num_steps=icem_num_steps,
        num_elites=1_000,
        exponent=icem_colored_noise_exponent,
        )
    
    optimizer = iCEMOptimizer(horizon=optimizer_horizon,
                              key = jr.PRNGKey(seed),
                              opt_params=opt_params,
                              )

    agent_class = None
    if exploration == 'optimistic':
        agent_class = ContinuousOptimisticModelBasedAgent
    elif exploration == 'pets':
        agent_class = ContinuousPETSModelBasedAgent
        
    class RCCarEnvReward(Reward):
        _angle_idx: int = 2
        dim_action: Tuple[int] = (2,)

        def __init__(self, goal: jnp.array, encode_angle: bool = False, ctrl_cost_weight: float = 0.005,
                    bound: float = 0.1, margin_factor: float = 10.0):
            super().__init__(x_dim=env.dim_state[0], u_dim=2)
            self.goal = goal
            self.ctrl_cost_weight = ctrl_cost_weight
            self.encode_angle = encode_angle
            # Margin 20 seems to work even better (maybe try at some point)
            self.tolerance_reward = ToleranceReward(bounds=(0.0, bound), margin=margin_factor * bound,
                                                    value_at_margin=0.1, sigmoid='long_tail')

        def __call__(self,
                     x: chex.Array,
                     u: chex.Array,
                     reward_params: RewardParams,
                     x_next: chex.Array | None = None):
            """ Computes the reward for the given transition """
            reward_ctrl = self.action_reward(u)
            reward_state = self.state_reward(x, x_next)
            reward = reward_state + self.ctrl_cost_weight * reward_ctrl
            reward_dist = Normal(reward, jnp.zeros_like(reward))
            return reward_dist, reward_params

        @staticmethod
        def action_reward(action: jnp.array) -> jnp.array:
            """ Computes the reward/penalty for the given action """
            return - (action ** 2).sum(-1)

        def state_reward(self, obs: jnp.array, next_obs: jnp.array) -> jnp.array:
            """ Computes the reward for the given observations """
            if self.encode_angle:
                next_obs = decode_angles(next_obs, angle_idx=self._angle_idx)
            pos_diff = next_obs[..., :2] - self.goal[:2]
            theta_diff = next_obs[..., 2] - self.goal[2]
            pos_dist = jnp.sqrt(jnp.sum(jnp.square(pos_diff), axis=-1))
            theta_dist = jnp.abs(((theta_diff + jnp.pi) % (2 * jnp.pi)) - jnp.pi)
            total_dist = jnp.sqrt(pos_dist ** 2 + theta_dist ** 2)
            reward = self.tolerance_reward(total_dist)
            return reward
        
        def init_params(self, key: chex.PRNGKey) -> RewardParams:
            return {'dt': env.dt}

    reward_model = RCCarEnvReward(goal=jnp.array([0.0, 0.0, 0.0]),
                                  encode_angle=True)
    
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
        'predict_difference': False,
        'reset_statistical_model': reset_statistical_model,
        'dt': env.dt,
        'dynamics_dt': bnn_dt,
        'state_extras_ref': state_extras,
    }

    config = dict(num_offline_samples=num_offline_samples,
                  sample_horizon=num_online_samples,
                  optimizer_horizon=optimizer_horizon,
                  deterministic_policy_for_data_collection=deterministic_policy_for_data_collection,
                  icem_num_steps=icem_num_steps,
                  icem_colored_noise_exponent=icem_colored_noise_exponent,
                  seed=seed,
                  num_episodes=num_episodes,
                  bnn_steps=bnn_steps,
                  bnn_features=bnn_features,
                  bnn_train_share=bnn_train_share,
                  bnn_weight_decay=bnn_weight_decay,
                  first_episode_for_policy_training=first_episode_for_policy_training,
                  exploration=exploration,
                  reset_statistical_model=reset_statistical_model,
                  regression_model=regression_model,
                  beta=beta,
                  bnn_dt = bnn_dt,
                  smoother_steps=smoother_steps,
                  smoother_features=smoother_features,
                  smoother_train_share=smoother_train_share,
                  smoother_weight_decay=smoother_weight_decay,
                  state_data_source=state_data_source,
                  agent_kwargs=agent_kwargs,
                  )
    
    if log_mode > 0:
        wandb.init(project=project_name,
                   dir='/cluster/scratch/' + ENTITY,
                   config=config)

    base_agent = DifferentiatingAgent(agent_type=agent_class,
                                      differentiator=BNN_Differentiator,
                                      state_data_source=state_data_source,
                                      **agent_kwargs)

    key_agent = jr.PRNGKey(seed)
    agent_state = base_agent.run_episodes(num_episodes=num_episodes,
                                          start_from_scratch=True,
                                          key=key_agent)
    wandb.finish()


def main(args):
    experiment(project_name=args.project_name,
               num_offline_samples=args.num_offline_samples,
               optimizer_horizon=args.optimizer_horizon,
               num_online_samples=args.num_online_samples,
               deterministic_policy_for_data_collection=bool(args.deterministic_policy_for_data_collection),
               icem_num_steps=args.icem_num_steps,
               icem_colored_noise_exponent=args.icem_colored_noise_exponent,
               seed=args.seed,
               num_episodes=args.num_episodes,
               bnn_steps=args.bnn_steps,
               bnn_features=args.bnn_features,
               bnn_train_share=args.bnn_train_share,
               bnn_weight_decay=args.bnn_weight_decay,
               first_episode_for_policy_training=args.first_episode_for_policy_training,
               exploration=args.exploration,
               reset_statistical_model=bool(args.reset_statistical_model),
               regression_model=args.regression_model,
               beta=args.beta,
               bnn_dt=args.bnn_dt,
               smoother_steps=args.smoother_steps,
               smoother_features=args.smoother_features,
               smoother_train_share=args.smoother_train_share,
               smoother_weight_decay=args.smoother_weight_decay,
               state_data_source=args.state_data_source,
               log_mode=args.log_mode,
               )


if __name__ == '__main__':

    def underscore_to_tuple(value: str):
        return tuple(map(int, value.split('_')))
    
    def underscore_to_list(value: str):
        return list(map(float, value.split('_')))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='ICEM_Cartpole_Debug')
    parser.add_argument('--num_offline_samples', type=int, default=0) # has to be multiple of num_online_samples
    parser.add_argument('--optimizer_horizon', type=int, default=50)
    parser.add_argument('--num_online_samples', type=int, default=100)
    parser.add_argument('--deterministic_policy_for_data_collection', type=int, default=1)
    parser.add_argument('--icem_num_steps', type=int, default=10)
    parser.add_argument('--icem_colored_noise_exponent', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_episodes', type=int, default=20)
    parser.add_argument('--bnn_steps', type=int, default=48_000)
    parser.add_argument('--bnn_features', type=underscore_to_tuple, default='64_64')
    parser.add_argument('--bnn_train_share', type=float, default=0.8)
    parser.add_argument('--bnn_weight_decay', type=float, default=0.0)
    parser.add_argument('--first_episode_for_policy_training', type=int, default=1)
    parser.add_argument('--exploration', type=str, default='pets')
    parser.add_argument('--reset_statistical_model', type=int, default=1)
    parser.add_argument('--regression_model', type=str, default='probabilistic_ensemble')
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--bnn_dt', type=float, default=0.05)
    parser.add_argument('--smoother_steps', type=int, default=48_000)
    parser.add_argument('--smoother_features', type=underscore_to_tuple, default='64_64_64')
    parser.add_argument('--smoother_train_share', type=float, default=1.0)
    parser.add_argument('--smoother_weight_decay', type=float, default=1e-4)
    parser.add_argument('--state_data_source', type=str, default='smoother')
    parser.add_argument('--log_mode', type=int, default=0)

    args = parser.parse_args()
    main(args)
