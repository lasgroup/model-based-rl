import argparse

ENTITY = 'kiten'


def experiment(
        seed: int = 0,
        project_name: str = 'GPUSpeedTest',
        num_offline_samples: int = 100,
        sac_horizon: int = 100,
        deterministic_policy_for_data_collection: bool = False,
        predict_difference: bool = False
):
    import chex
    import jax.numpy as jnp
    import jax.random as jr
    import wandb
    from brax.training.replay_buffers import UniformSamplingQueue
    from brax.training.types import Transition
    from bsm.statistical_model.bnn_statistical_model import BNNStatisticalModel
    from distrax import Normal
    from jax.nn import swish
    from mbpo.optimizers import SACOptimizer
    from mbpo.systems.rewards.base_rewards import Reward, RewardParams

    from mbrl.envs.pendulum import PendulumEnv
    from mbrl.model_based_agent.active_exploration_model_based_agents import OptimisticActiveExplorationModelBasedAgent
    from mbrl.utils.offline_data import PendulumOfflineData

    config = dict(num_offline_samples=num_offline_samples,
                  sac_horizon=sac_horizon,
                  deterministic_policy_for_data_collection=deterministic_policy_for_data_collection,
                  regression_model='ensemble',
                  exploration='optimistic',
                  reward_model='dm-control',
                  predict_difference=predict_difference,
                  )

    swing_up_env = PendulumEnv(reward_source='dm-control')
    swing_down_params = swing_up_env.reward_params.replace(target_angle=jnp.pi)
    swing_down_env = PendulumEnv(reward_source='dm-control')
    swing_down_env.reward_params = swing_down_params

    class PendulumReward(Reward):
        def __init__(self, env: PendulumEnv):
            super().__init__(x_dim=2, u_dim=1)
            self.env = env

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
            reward = self.env.tolerance_reward(jnp.sqrt(self.env.reward_params.angle_cost * diff_th ** 2 +
                                                        0.1 * omega ** 2)) - self.env.reward_params.control_cost * u ** 2
            reward = reward.squeeze()
            reward_dist = Normal(reward, jnp.zeros_like(reward))
            return reward_dist, reward_params

        def init_params(self, key: chex.PRNGKey) -> RewardParams:
            return {'dt': self.env.dt}

    reward_model_swing_up = PendulumReward(env=swing_up_env)
    reward_model_swing_down = PendulumReward(env=swing_down_env)

    offline_data_gen = PendulumOfflineData()
    key = jr.PRNGKey(seed)
    key_offline_data, key = jr.split(key, 2)
    if num_offline_samples > 0:
        offline_data = offline_data_gen.sample_transitions(key=key_offline_data,
                                                           num_samples=num_offline_samples)
    else:
        offline_data = None

    horizon = 200
    model = BNNStatisticalModel(
        input_dim=swing_up_env.observation_size + swing_up_env.action_size,
        output_dim=swing_up_env.observation_size,
        num_training_steps=15_000,
        output_stds=1e-3 * jnp.ones(swing_up_env.observation_size),
        features=(64, 64, 64),
        num_particles=5,
        logging_wandb=True,
        return_best_model=True,
        eval_batch_size=64,
        train_share=0.8,
        eval_frequency=5_000,
    )

    sac_kwargs = {
        'num_timesteps': 100_000,
        'episode_length': sac_horizon,
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
        'max_replay_size': 10 ** 5,
        'grad_updates_per_step': 20 * 32,  # should be num_envs * num_env_steps_between_updates
        'deterministic_eval': True,
        'init_log_alpha': 0.,
        'policy_hidden_layer_sizes': (32,) * 5,
        'policy_activation': swish,
        'critic_hidden_layer_sizes': (128,) * 4,
        'critic_activation': swish,
        'wandb_logging': True,
        'return_best_model': True,
    }
    max_replay_size_true_data_buffer = 10 ** 4
    dummy_sample = Transition(observation=jnp.ones(swing_up_env.observation_size),
                              action=jnp.zeros(shape=(swing_up_env.action_size,)),
                              reward=jnp.array(0.0),
                              discount=jnp.array(0.99),
                              next_observation=jnp.ones(swing_up_env.observation_size))

    sac_buffer = UniformSamplingQueue(
        max_replay_size=max_replay_size_true_data_buffer,
        dummy_data_sample=dummy_sample,
        sample_batch_size=1)

    optimizer = SACOptimizer(system=None,
                             true_buffer=sac_buffer,
                             **sac_kwargs)

    wandb.init(project=project_name,
               dir='/cluster/scratch/' + ENTITY,
               config=config
               )

    agent = OptimisticActiveExplorationModelBasedAgent(
        env=swing_up_env,
        eval_envs=[swing_up_env, swing_down_env],
        reward_model_list=[reward_model_swing_up, reward_model_swing_down],
        statistical_model=model,
        optimizer=optimizer,
        episode_length=horizon,
        offline_data=offline_data,
        num_envs=1,
        num_eval_envs=1,
        log_to_wandb=True,
        deterministic_policy_for_data_collection=deterministic_policy_for_data_collection,
        predict_difference=predict_difference,
    )

    agent_state, actors_for_reward_models = agent.run_episodes(num_episodes=20,
                                                               start_from_scratch=True,
                                                               key=key)

    wandb.finish()


def main(args):
    experiment(project_name=args.project_name,
               num_offline_samples=args.num_offline_samples,
               sac_horizon=args.sac_horizon,
               deterministic_policy_for_data_collection=bool(args.deterministic_policy_for_data_collection),
               predict_difference=bool(args.predict_difference))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--project_name', type=str, default='Model_based_pets')
    parser.add_argument('--num_offline_samples', type=int, default=10)
    parser.add_argument('--sac_horizon', type=int, default=10)
    parser.add_argument('--deterministic_policy_for_data_collection', type=int, default=0)
    parser.add_argument('--train_steps_sac', type=int, default=2_000)
    parser.add_argument('--train_steps_bnn', type=int, default=300)
    parser.add_argument('--predict_difference', type=int, default=1)

    args = parser.parse_args()
    main(args)
