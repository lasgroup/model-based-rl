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

    def __init__(self,
                 dt: float,
                 min_time_between_switches: float,
                 max_time_between_switches: float,
                 episode_time: float,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.dt = dt
        self.min_time_between_switches = min_time_between_switches
        self.max_time_between_switches = max_time_between_switches
        self.episode_time = episode_time

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

    @staticmethod
    def compute_time(pseudo_time: chex.Array,
                     dt: float,
                     t_min: float,
                     t_max: float,
                     env_time: chex.Array,
                     time_horizon: float
                     ) -> chex.Array:
        time_for_action = ((t_max - t_min) / 2 * pseudo_time + (t_max + t_min) / 2)
        return jnp.minimum((time_for_action // dt) * dt, time_horizon - env_time)

    def _collected_buffer_to_train_data(self,
                                        collected_buffer_state: ReplayBufferState):
        idx = jnp.arange(start=collected_buffer_state.sample_position, stop=collected_buffer_state.insert_position)
        all_data = jnp.take(collected_buffer_state.data, idx, axis=0, mode='wrap')
        all_transitions = self.collected_data_buffer._unflatten_fn(all_data)
        # obs = [env_state, env_times]
        obs = all_transitions.observation
        # action = [env_action, time_to_control]
        actions = all_transitions.action

        env_states, env_times = obs[..., :-1], obs[..., 1]
        env_actions, pseudo_times_for_action = actions[..., :-1], actions[..., 1]
        rewards = all_transitions.reward.reshape(-1, 1)  # This should be only integrated reward

        times_for_action = self.compute_time(pseudo_times_for_action,
                                             dt=self.dt,
                                             t_min=self.min_time_between_switches,
                                             t_max=self.max_time_between_switches,
                                             env_time=env_times,
                                             time_horizon=self.episode_time)

        inputs = jnp.concatenate([env_states, env_actions, times_for_action[..., None]], axis=-1)
        env_states_next = all_transitions.next_observation[..., :-1]  # We remove time_to_go
        if self.predict_difference:
            target = env_states_next - env_states
        else:
            target = env_states_next
        outputs = jnp.concatenate([target, rewards], axis=-1)  # append the integrated reward to the output
        return Data(inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    from mbrl.envs.pendulum import PendulumEnv
    from bsm.statistical_model.bnn_statistical_model import BNNStatisticalModel
    from mbpo.optimizers import SACOptimizer
    from distrax import Normal
    from mbrl.utils.offline_data import PendulumOfflineData, WhenToControlWrapper
    from wtc.wrappers.ih_switching_cost import IHSwitchCostWrapper, ConstantSwitchCost
    from wtc.utils import discrete_to_continuous_discounting
    from bsm.bayesian_regression import DeterministicEnsemble

    log_wandb = True
    ENTITY = 'trevenl'

    base_env = PendulumEnv(reward_source='dm-control')

    min_time_between_switches = 1 * base_env.dt
    max_time_between_switches = 30 * base_env.dt
    num_integrator_steps = 100
    switch_cost = 0.1

    env = IHSwitchCostWrapper(base_env,
                              num_integrator_steps=num_integrator_steps,
                              min_time_between_switches=min_time_between_switches,
                              max_time_between_switches=max_time_between_switches,
                              switch_cost=ConstantSwitchCost(value=jnp.array(0.0)),
                              time_as_part_of_state=True)

    episode_time = base_env.dt * num_integrator_steps


    class TransitionReward(Reward):
        def __init__(self):
            super().__init__(x_dim=2, u_dim=1)

        def __call__(self,
                     x: chex.Array,
                     u: chex.Array,
                     reward_params: RewardParams,
                     x_next: chex.Array | None = None
                     ):
            reward = jnp.array(-switch_cost)
            reward_dist = Normal(reward, jnp.zeros_like(reward))
            return reward_dist, reward_params

        def init_params(self, key: chex.PRNGKey) -> RewardParams:
            return {'dt': 0.05}


    offline_data_gen = WhenToControlWrapper(
        num_integrator_steps=num_integrator_steps,
        min_time_between_switches=min_time_between_switches,
        max_time_between_switches=max_time_between_switches
    )
    key = jr.PRNGKey(0)

    offline_data = offline_data_gen.sample_transitions(key=key,
                                                       num_samples=100)

    horizon = 100
    model = BNNStatisticalModel(
        input_dim=env.observation_size + env.action_size - 1,  # -1 since we don't input env_time
        output_dim=env.observation_size + 1 - 1,  # +1 for the reward -1 for env time
        num_training_steps=30_000,
        output_stds=1e-3 * jnp.ones(env.observation_size + 1 - 1),  # +1 for the reward -1 for env_time
        beta=2.0 * jnp.ones(shape=(env.observation_size + 1 - 1,)),
        features=(64,) * 3,
        bnn_type=DeterministicEnsemble,
        num_particles=5,
        logging_wandb=log_wandb,
        return_best_model=True,
        eval_batch_size=64,
        train_share=0.8,
        eval_frequency=5_000,
        weight_decay=0.0,
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
        'wandb_logging': log_wandb,
        'return_best_model': True,
        'non_equidistant_time': True,
        'continuous_discounting': continuous_discounting,
        'min_time_between_switches': min_time_between_switches,
        'max_time_between_switches': max_time_between_switches,
        'env_dt': env.dt,
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
    if log_wandb:
        wandb.init(project="ModelBasedTest",
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
        log_to_wandb=log_wandb,
        dt=env.dt,
        min_time_between_switches=min_time_between_switches,
        max_time_between_switches=max_time_between_switches,
        episode_time=episode_time,
    )

    agent_state = agent.run_episodes(num_episodes=20,
                                     start_from_scratch=True,
                                     key=jr.PRNGKey(0))

    wandb.finish()
