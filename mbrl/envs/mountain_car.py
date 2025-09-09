from functools import partial

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from brax.envs.base import State, Env
from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
from jaxtyping import Float, Array


class MountainCarEnv(Env):
    def __init__(self,
                 ):
        self.env = Continuous_MountainCarEnv()

    def reset(self,
              rng: jax.Array) -> State:
        obs = jnp.array([jr.uniform(key=rng, minval=-0.6, maxval=-0.4, shape=()), 0.0])
        state = State(
            pipeline_state=None,
            obs=obs.astype(jnp.float32),
            reward=jnp.array(0.0, jnp.float32),
            done=jnp.array(0.0, jnp.float32),
            metrics={},
            info={}
        )
        return state

    def reward(self,
               obs: Float[Array, '2'],
               action: Float[Array, '1'],
               next_obs: Float[Array, '2'], ) -> Float[Array, '1']:
        pos = next_obs[..., 0]
        velocity = next_obs[..., 1]
        terminate = jnp.logical_and(pos >= self.env.goal_position, velocity >= self.env.goal_velocity)
        reward = 100.0 * terminate.astype(jnp.float32) # action handled by IHSwitchingCost
        return reward.reshape(-1).squeeze()

    def next_step(self,
                  obs: Float[Array, '2'],
                  action: Float[Array, '1'], ) -> Float[Array, '2']:
        obs = jnp.atleast_2d(obs).reshape(-1, 2)
        action = jnp.atleast_2d(action).reshape(-1, 1)
        pos = obs[..., 0]
        velocity = obs[..., 1]
        force = jnp.clip(action, a_min=-1, a_max=1) * self.env.max_action
        next_velocity = velocity + force * self.env.power - 0.0025 * jnp.cos(3 * pos)
        next_velocity = jnp.clip(next_velocity, a_min=-self.env.max_speed, a_max=self.env.max_speed)
        next_position = pos + next_velocity
        next_position = jnp.clip(next_position, self.env.min_position, self.env.max_position)
        out_of_bounds = jnp.logical_and(next_position - self.env.min_position <= 0.0, next_velocity < 0.0)
        next_velocity = jnp.where(out_of_bounds, jnp.array(0.0, jnp.float32), next_velocity)
        next_obs = jnp.concatenate([next_position, next_velocity], axis=-1).reshape(-1, 2).squeeze()
        return next_obs

    @partial(jax.jit, static_argnums=0)
    def step(self,
             state: State,
             action: jax.Array) -> State:
        obs = state.obs
        assert obs.shape == (self.observation_size,)
        assert action.shape == (self.action_size,)
        next_obs = self.next_step(obs, action)
        reward = self.reward(obs, action, next_obs)
        next_state = State(pipeline_state=state.pipeline_state,
                           obs=next_obs,
                           reward=reward,
                           done=state.done,
                           metrics=state.metrics,
                           info=state.info)
        return next_state
    
    @property
    def dt(self):
        return  1.0

    @property
    def observation_size(self) -> int:
        return 2

    @property
    def action_size(self) -> int:
        return 1

    def backend(self) -> str:
        return 'positional'