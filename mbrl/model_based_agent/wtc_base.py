from abc import abstractmethod

import chex
import jax.numpy as jnp
from brax.training.replay_buffers import ReplayBufferState
from bsm.utils.normalization import Data
from mbpo.optimizers.base_optimizer import BaseOptimizer
from mbpo.utils.type_aliases import OptimizerState
from mbpo.systems.base_systems import System, Dynamics

from mbrl.model_based_agent.base_model_based_agent import BaseModelBasedAgent
from mbrl.model_based_agent.optimizer_wrapper import Actor


@chex.dataclass
class ModelBasedAgentState:
    optimizer_state: OptimizerState
    env_steps: chex.Array
    key: chex.Array


class WtcBaseModelBasedAgent(BaseModelBasedAgent):

    def __init__(self,
                 dt: float,
                 min_time_between_switches: float,
                 max_time_between_switches: float,
                 episode_time: float,
                 *args,
                 **kwargs):
        self.dt = dt
        self.min_time_between_switches = min_time_between_switches
        self.max_time_between_switches = max_time_between_switches
        self.episode_time = episode_time
        super().__init__(*args, **kwargs)

    @abstractmethod
    def prepare_actor(self,
                      optimizer: BaseOptimizer,
                      ) -> Actor:
        pass

    def prepare_wtc_actor(self,
                          optimizer: BaseOptimizer,
                          dynamics,
                          system,
                          actor,
                          ) -> Actor:
        dynamics = dynamics(statistical_model=self.statistical_model,
                            x_dim=self.env.observation_size,
                            u_dim=self.env.action_size,
                            min_time_between_switches=self.min_time_between_switches,
                            max_time_between_switches=self.max_time_between_switches,
                            episode_time=self.episode_time,
                            dt=self.dt)
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
