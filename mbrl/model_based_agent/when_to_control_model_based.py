import chex
import jax.numpy as jnp
import jax.random as jr
import wandb
from brax.training.replay_buffers import UniformSamplingQueue, ReplayBufferState
from brax.training.types import Transition
from bsm.utils.normalization import Data
from jax.nn import swish
from mbpo.optimizers.base_optimizer import BaseOptimizer
from mbpo.systems.rewards.base_rewards import Reward, RewardParams
from mbpo.utils.type_aliases import OptimizerState

from mbrl.model_based_agent.optimizer_wrapper import Actor, PetsActor
from mbrl.model_based_agent.system_wrapper import TransitionCostDynamics, TransitionCostPetsSystem
from mbrl.model_based_agent.base_model_based_agent import BaseModelBasedAgent


@chex.dataclass
class ModelBasedAgentState:
    optimizer_state: OptimizerState
    env_steps: chex.Array
    key: chex.Array


class WhenToControlModelBasedAgent(BaseModelBasedAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_actor(self,
                      optimizer: BaseOptimizer,
                      ) -> Actor:
        dynamics, system, actor = TransitionCostDynamics, TransitionCostPetsSystem, PetsActor
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

    def _collected_buffer_to_train_data(self,
                                        collected_buffer_state: ReplayBufferState):
        idx = jnp.arange(start=collected_buffer_state.sample_position, stop=collected_buffer_state.insert_position)
        all_data = jnp.take(collected_buffer_state.data, idx, axis=0, mode='wrap')
        all_transitions = self.collected_data_buffer._unflatten_fn(all_data)
        obs = all_transitions.observation[..., :-1]  # We remove the time-to-go component
        actions = all_transitions.action
        rewards = all_transitions.reward.reshape(-1, 1)  # This should be only integrated reward
        inputs = jnp.concatenate([obs, actions], axis=-1)
        next_obs = all_transitions.next_observation[..., :-1]  # We remove the time-to-go component
        if self.predict_difference:
            delta_obs = next_obs - obs
        else:
            delta_obs = next_obs
        outputs = jnp.concatenate([delta_obs, rewards], axis=-1)
        return Data(inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    from mbrl.envs.pendulum import PendulumEnv
    from bsm.statistical_model.bnn_statistical_model import BNNStatisticalModel
    from mbpo.optimizers import SACOptimizer
    from distrax import Normal
    from mbrl.utils.offline_data import PendulumOfflineData, WhenToControlWrapper
    from wtc.wrappers.ih_switching_cost import IHSwitchCostWrapper, ConstantSwitchCost
    from wtc.utils import discrete_to_continuous_discounting

    ENTITY = 'trevenl'

    base_env = PendulumEnv(reward_source='dm-control')

    env = IHSwitchCostWrapper(base_env,
                              num_integrator_steps=100,
                              min_time_between_switches=1 * base_env.dt,
                              max_time_between_switches=30 * base_env.dt,
                              switch_cost=ConstantSwitchCost(value=jnp.array(1.0)),
                              time_as_part_of_state=True)


    class TransitionReward(Reward):
        def __init__(self):
            super().__init__(x_dim=2, u_dim=1)

        def __call__(self,
                     x: chex.Array,
                     u: chex.Array,
                     reward_params: RewardParams,
                     x_next: chex.Array | None = None
                     ):
            assert x.shape == (4,) and u.shape == (2,)
            reward = jnp.array(-1.0)
            reward_dist = Normal(reward, jnp.zeros_like(reward))
            return reward_dist, reward_params

        def init_params(self, key: chex.PRNGKey) -> RewardParams:
            return {'dt': 0.05}


    offline_data_gen = WhenToControlWrapper()
    key = jr.PRNGKey(0)

    offline_data = offline_data_gen.sample_transitions(key=key,
                                                       num_samples=10_000)

    offline_data = None
    horizon = 100
    model = BNNStatisticalModel(
        input_dim=env.observation_size + env.action_size - 1,
        output_dim=env.observation_size - 1 + 1,
        num_training_steps=30_000,
        output_stds=1e-3 * jnp.ones(env.observation_size - 1 + 1),
        features=(64, 64, 64),
        num_particles=5,
        logging_wandb=True,
        return_best_model=True,
        eval_batch_size=64,
        train_share=0.8,
        eval_frequency=5_000,
    )
    discount_factor = 0.99
    continuous_discounting = discrete_to_continuous_discounting(discrete_discounting=discount_factor,
                                                                dt=env.dt)
    sac_kwargs = {
        'num_timesteps': 20_000,
        'episode_length': 100,
        'num_env_steps_between_updates': 10,
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
        'batch_size': 64,
        'num_evals': 20,
        'normalize_observations': True,
        'reward_scaling': 1.,
        'tau': 0.005,
        'min_replay_size': 10 ** 3,
        'max_replay_size': 10 ** 5,
        'grad_updates_per_step': 10 * 64,  # should be num_envs * num_env_steps_between_updates
        'deterministic_eval': True,
        'init_log_alpha': 0.,
        'policy_hidden_layer_sizes': (32,) * 5,
        'policy_activation': swish,
        'critic_hidden_layer_sizes': (128,) * 3,
        'critic_activation': swish,
        'wandb_logging': True,
        'return_best_model': True,
        # 'non_equidistant_time': True,
        # 'continuous_discounting': continuous_discounting,
        # 'min_time_between_switches': 1 * base_env.dt,
        # 'max_time_between_switches': 30 * base_env.dt,
        # 'env_dt': env.dt,
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

    wandb.init(project="Model-based Agent",
               dir='/cluster/scratch/' + ENTITY,
               )

    agent = WhenToControlModelBasedAgent(
        env=env,
        eval_env=env,
        statistical_model=model,
        optimizer=optimizer,
        reward_model=TransitionReward(),
        episode_length=horizon,
        offline_data=offline_data,
        num_envs=1,
        num_eval_envs=1,
        log_to_wandb=True,
    )

    agent_state = agent.run_episodes(num_episodes=20,
                                     start_from_scratch=True,
                                     key=jr.PRNGKey(0))

    wandb.finish()
