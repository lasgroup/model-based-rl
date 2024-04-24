from abc import ABC, abstractmethod

import chex
import jax.numpy as jnp
import jax.random as jr
from brax.envs import Env, State
from jaxtyping import Float, Array
from brax.training.types import Transition
from jax import vmap

from mbrl.envs.pendulum import PendulumEnv
from wtc.wrappers.ih_switching_cost import IHSwitchCostWrapper, ConstantSwitchCost, AugmentedPipelineState


class OfflineData(ABC):

    def __init__(self,
                 env: Env):
        self.env = env

    def sample_states(self,
                      key: chex.PRNGKey,
                      num_samples: int) -> Float[Array, 'dim_batch dim_state']:
        xs = self._sample_states(key=key, num_samples=num_samples)
        assert xs.ndim == 2, f"Expected 2d array, got {xs.shape}"
        assert xs.shape[1] == self.env.observation_size, f"Expected observation size, got {xs.shape[1]}"
        return xs

    def sample_actions(self,
                       key: chex.PRNGKey,
                       num_samples: int) -> Float[Array, 'dim_batch dim_action']:
        us = self._sample_actions(key=key, num_samples=num_samples)
        assert us.ndim == 2, f"Expected 2d array, got {us.shape}"
        assert us.shape[1] == self.env.action_size, f"Expected action size, got {us.shape[1]}"
        return us

    @abstractmethod
    def _sample_actions(self,
                        key: chex.PRNGKey,
                        num_samples: int) -> Float[Array, 'dim_batch dim_action']:
        pass

    @abstractmethod
    def _sample_states(self,
                       key: chex.PRNGKey,
                       num_samples: int) -> Float[Array, 'dim_batch dim_state']:
        pass

    def sample_transitions(self,
                           key: chex.PRNGKey,
                           num_samples: int) -> Transition:
        states = self.sample_states(key=key, num_samples=num_samples)
        actions = self.sample_actions(key=key, num_samples=num_samples)

        brax_state = State(pipeline_state=jnp.zeros(shape=(num_samples,)),
                           obs=states,
                           reward=jnp.zeros(shape=(num_samples,)),
                           done=jnp.zeros(shape=(num_samples,)))

        next_states = vmap(self.env.step)(brax_state, actions)
        transitions = Transition(observation=brax_state.obs,
                                 action=actions,
                                 reward=next_states.reward,
                                 discount=jnp.zeros(shape=(num_samples,)),
                                 next_observation=next_states.obs
                                 )
        return transitions


class PendulumOfflineData(OfflineData):

    def __init__(self):
        super().__init__(env=PendulumEnv(reward_source='dm-control'))

    def _sample_states(self,
                       key: chex.PRNGKey,
                       num_samples: int) -> Float[Array, 'dim_batch dim_state']:
        key_angle, key_angular_velocity = jr.split(key)
        angles = jr.uniform(key_angle, shape=(num_samples,), minval=-jnp.pi, maxval=jnp.pi)
        cos, sin = jnp.cos(angles), jnp.sin(angles)
        angular_velocity = jr.uniform(key_angular_velocity, shape=(num_samples,), minval=-10, maxval=10)
        return jnp.stack([cos, sin, angular_velocity], axis=-1)

    def _sample_actions(self,
                        key: chex.PRNGKey,
                        num_samples: int) -> Float[Array, 'dim_batch dim_action']:
        actions = jr.uniform(key, shape=(num_samples, 1), minval=-1, maxval=1)
        return actions


class WhenToControlWrapper(OfflineData):

    def __init__(self):
        base_env = PendulumEnv(reward_source='dm-control')
        env = IHSwitchCostWrapper(base_env,
                                  num_integrator_steps=100,
                                  min_time_between_switches=1 * base_env.dt,
                                  max_time_between_switches=30 * base_env.dt,
                                  switch_cost=ConstantSwitchCost(value=jnp.array(0.0)),
                                  time_as_part_of_state=True,
                                  discounting=1.0
                                  )
        super().__init__(env=env)

    def _sample_states(self,
                       key: chex.PRNGKey,
                       num_samples: int) -> Float[Array, 'dim_batch dim_state']:
        key_angle, key_angular_velocity = jr.split(key)
        angles = jr.uniform(key_angle, shape=(num_samples,), minval=-jnp.pi, maxval=jnp.pi)
        cos, sin = jnp.cos(angles), jnp.sin(angles)
        angular_velocity = jr.uniform(key_angular_velocity, shape=(num_samples,), minval=-5, maxval=5)
        current_times = jr.uniform(key=key, shape=(num_samples,),
                                   minval=0, maxval=5.0)  # TODO: works only for Pendulum
        return jnp.stack([cos, sin, angular_velocity, current_times], axis=-1)

    def _sample_actions(self,
                        key: chex.PRNGKey,
                        num_samples: int) -> Float[Array, 'dim_batch dim_action']:
        actions = jr.uniform(key=key, shape=(num_samples, self.env.action_size), minval=-1, maxval=1)
        return actions

    def sample_transitions(self,
                           key: chex.PRNGKey,
                           num_samples: int) -> Transition:
        key_states, key_actions = jr.split(key, 2)
        states = self.sample_states(key=key_states, num_samples=num_samples)
        actions = self.sample_actions(key=key_actions, num_samples=num_samples)
        brax_state = State(pipeline_state=jnp.zeros(shape=(num_samples,)),
                           obs=states,
                           reward=jnp.zeros(shape=(num_samples,)),
                           done=jnp.zeros(shape=(num_samples,)))

        next_states = vmap(self.env.step)(brax_state, actions)
        transitions = Transition(observation=brax_state.obs,
                                 action=actions,
                                 reward=next_states.reward,
                                 discount=jnp.zeros(shape=(num_samples,)),
                                 next_observation=next_states.obs
                                 )
        return transitions


if __name__ == '__main__':
    offline_data_gen = WhenToControlWrapper()
    key = jr.PRNGKey(0)

    data = offline_data_gen.sample_transitions(key=key, num_samples=5)
