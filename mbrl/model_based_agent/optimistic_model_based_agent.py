from .base_model_based_agent import BaseModelBasedAgent
from .continuous_base_model_based_agent import ContinuousBaseModelBasedAgent
from mbpo.optimizers.base_optimizer import BaseOptimizer
from mbrl.model_based_agent.optimizer_wrapper import Actor, OptimisticActor
from mbrl.model_based_agent.system_wrapper import OptimisticSystem, OptimisticDynamics, ContinuousOptimisticSystem, ContinuousOptimisticDynamics


class OptimisticModelBasedAgent(BaseModelBasedAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_actor(self,
                      optimizer: BaseOptimizer,
                      ) -> Actor:
        dynamics, system, actor = OptimisticDynamics, OptimisticSystem, OptimisticActor
        dynamics = dynamics(statistical_model=self.statistical_model,
                            x_dim=self.env.observation_size,
                            u_dim=self.env.action_size)
        system = system(dynamics=dynamics,
                        reward=self.reward_model, )
        actor = actor(env_observation_size=self.env.observation_size,
                      env_action_size=self.env.action_size,
                      optimizer=optimizer)
        actor.set_system(system=system)
        return actor
    

class ContinuousOptimisticModelBasedAgent(ContinuousBaseModelBasedAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_actor(self,
                      optimizer: BaseOptimizer,
                      ) -> Actor:
        dynamics, system, actor = ContinuousOptimisticDynamics, ContinuousOptimisticSystem, OptimisticActor
        dynamics = dynamics(statistical_model=self.statistical_model,
                            x_dim=self.env.observation_size,
                            u_dim=self.env.action_size)
        system = system(dynamics=dynamics,
                        reward=self.reward_model, )
        actor = actor(env_observation_size=self.env.observation_size,
                      env_action_size=self.env.action_size,
                      optimizer=optimizer)
        actor.set_system(system=system)
        return actor

if __name__ == "__main__":
    from mbrl.envs.pendulum import PendulumEnv
    from bsm.statistical_model.bnn_statistical_model import BNNStatisticalModel
    from mbpo.optimizers import SACOptimizer
    from distrax import Normal
    from mbrl.utils.offline_data import PendulumOfflineData

    import chex
    import jax.numpy as jnp
    import jax.random as jr
    from brax.training.replay_buffers import UniformSamplingQueue
    from brax.training.types import Transition
    from jax.nn import swish
    from mbpo.optimizers.base_optimizer import BaseOptimizer
    from mbpo.systems.rewards.base_rewards import Reward, RewardParams

    from mbrl.model_based_agent.optimizer_wrapper import Actor

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
    key = jr.PRNGKey(0)

    offline_data = offline_data_gen.sample_transitions(key=key,
                                                       num_samples=100)

    horizon = 200
    model = BNNStatisticalModel(
        input_dim=env.observation_size + env.action_size,
        output_dim=env.observation_size,
        num_training_steps=3_000,
        output_stds=1e-3 * jnp.ones(env.observation_size),
        features=(64, 64, 64),
        num_particles=5,
        logging_wandb=True,
        return_best_model=True,
        eval_batch_size=64,
        train_share=0.8,
        eval_frequency=1_000,
    )

    sac_kwargs = {
        'num_timesteps': 20_000,
        'episode_length': 64,
        'num_env_steps_between_updates': 10,
        'num_envs': 16,
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
        'grad_updates_per_step': 10 * 16,  # should be num_envs * num_env_steps_between_updates
        'deterministic_eval': True,
        'init_log_alpha': 0.,
        'policy_hidden_layer_sizes': (64, 64),
        'policy_activation': swish,
        'critic_hidden_layer_sizes': (64, 64),
        'critic_activation': swish,
        'wandb_logging': True,
        'return_best_model': True,
    }
    max_replay_size_true_data_buffer = 10 ** 4
    dummy_sample = Transition(observation=jnp.ones(env.observation_size),
                              action=jnp.zeros(shape=(env.action_size,)),
                              reward=jnp.array(0.0),
                              discount=jnp.array(0.99),
                              next_observation=jnp.ones(env.observation_size))

    sac_buffer = UniformSamplingQueue(
        max_replay_size=max_replay_size_true_data_buffer,
        dummy_data_sample=dummy_sample,
        sample_batch_size=1)

    optimizer = SACOptimizer(system=None,
                             true_buffer=sac_buffer,
                             **sac_kwargs)

    agent = OptimisticModelBasedAgent(
        env=env,
        eval_env=env,
        statistical_model=model,
        optimizer=optimizer,
        learning_style='Optimistic',
        reward_model=PendulumReward(),
        episode_length=horizon,
        offline_data=offline_data,
        num_envs=1,
        num_eval_envs=1,
        log_to_wandb=True,
    )

    agent_state = agent.run_episodes(num_episodes=20,
                                     start_from_scratch=True,
                                     key=jr.PRNGKey(0))
