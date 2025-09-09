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
        # Cache Gym constants as JAX float32 to avoid float64 upcasts
        self.min_position  = jnp.array(self.env.min_position,  jnp.float32)
        self.max_position  = jnp.array(self.env.max_position,  jnp.float32)
        self.max_speed     = jnp.array(self.env.max_speed,     jnp.float32)
        self.goal_position = jnp.array(self.env.goal_position, jnp.float32)
        self.goal_velocity = jnp.array(self.env.goal_velocity, jnp.float32)
        self.power         = jnp.array(self.env.power,         jnp.float32)
        self.max_action    = jnp.array(self.env.max_action,    jnp.float32)

    def reset(self,
              rng: jax.Array) -> State:
        pos0 = jr.uniform(key=rng, minval=-0.6, maxval=-0.4, shape=(), dtype=jnp.float32)
        obs = jnp.array([pos0, jnp.array(0.0, jnp.float32)], jnp.float32)
        state = State(
            pipeline_state=None,
            obs=obs,
            reward=jnp.array(0.0, jnp.float32),
            done=jnp.array(0.0, jnp.float32),
            metrics={},
            info={}
        )
        return state

    def reward(self,
               obs: Float[Array, '2'],
               action: Float[Array, '1'],
               next_obs: Float[Array, '2'], ) -> jax.Array:
        pos = next_obs[..., 0]
        vel = next_obs[..., 1]
        terminate = jnp.logical_and(pos >= self.goal_position, vel >= self.goal_velocity)
        reward = jnp.array(100.0, jnp.float32) * terminate.astype(jnp.float32)  # action cost handled elsewhere
        return reward.reshape(-1).astype(jnp.float32).squeeze()

    def next_step(self,
                  obs: Float[Array, '2'],
                  action: Float[Array, '1'], ) -> Float[Array, '2']:
        obs = jnp.atleast_2d(obs).reshape(-1, 2).astype(jnp.float32)
        action = jnp.atleast_2d(action).reshape(-1, 1).astype(jnp.float32)

        pos = obs[..., 0]
        vel = obs[..., 1]

        force = jnp.clip(action, a_min=-1.0, a_max=1.0) * self.max_action
        next_vel = vel + force * self.power - jnp.array(0.0025, jnp.float32) * jnp.cos(3.0 * pos)
        next_vel = jnp.clip(next_vel, a_min=-self.max_speed, a_max=self.max_speed)

        next_pos = pos + next_vel
        next_pos = jnp.clip(next_pos, self.min_position, self.max_position)

        out_of_bounds = jnp.logical_and(next_pos - self.min_position <= 0.0, next_vel < 0.0)
        next_vel = jnp.where(out_of_bounds, jnp.array(0.0, jnp.float32), next_vel)

        next_obs = jnp.concatenate([next_pos, next_vel], axis=-1).reshape(-1, 2).squeeze()
        return next_obs.astype(jnp.float32)

    @partial(jax.jit, static_argnums=0)
    def step(self, state: State, action: jax.Array) -> State:
        obs = state.obs
        assert obs.shape == (self.observation_size,)
        assert action.shape == (self.action_size,)

        next_obs = self.next_step(obs, action)
        reward = self.reward(obs, action, next_obs)

        # compute termination again here to set done
        pos = next_obs[..., 0]
        vel = next_obs[..., 1]
        terminate = jnp.logical_and(pos >= self.goal_position, vel >= self.goal_velocity)
        done = jnp.where(terminate, jnp.array(1.0, jnp.float32), state.done)

        next_state = State(pipeline_state=state.pipeline_state,
                           obs=next_obs,
                           reward=reward.astype(jnp.float32),
                           done=done.astype(jnp.float32),
                           metrics=state.metrics,
                           info=state.info)
        return next_state
    
    @property
    def dt(self):
        return 1.0

    @property
    def observation_size(self) -> int:
        return 2

    @property
    def action_size(self) -> int:
        return 1

    def backend(self) -> str:
        return 'positional'


def simple_policy_batched(obs_b: jnp.ndarray) -> jnp.ndarray:
    # obs_b: (B, 2)
    vel = obs_b[:, 1]
    a = jnp.where(vel >= 0.0, 1.0, -1.0)
    return a[:, None].astype(jnp.float32)  # (B, 1)

def main():
    from brax.envs import training 
    from wtc.wrappers.ih_switching_cost import IHSwitchCostWrapper, ConstantSwitchCost
    jax.config.update("jax_disable_jit", True)

    time_horizon = 10_000

    base_env = MountainCarEnv()
    env = IHSwitchCostWrapper(base_env,
                              num_integrator_steps=time_horizon,
                              min_time_between_switches=1,
                              max_time_between_switches=1,
                              switch_cost=ConstantSwitchCost(value=jnp.array(0.0)),
                              time_as_part_of_state=True)
    wrapped = training.wrap(env, episode_length=time_horizon, action_repeat=1)

    key = jr.PRNGKey(0)
    keys = jr.split(key, 1)
    state = wrapped.reset(keys)
    B = int(state.obs.shape[0])
    ACTION_SIZE = env.action_size          # = 1


    print("Running batched env (B=1) with training.wrap…")
    for t in range(10_000):
        raw_action = simple_policy_batched(state.obs)          # shape (B,) or (B,1)
        raw_action = jnp.asarray(raw_action, jnp.float32).reshape(B, 1)  # (B,1)

        # concat a constant second action component
        action_b = jnp.concatenate(
            [raw_action, jnp.ones((B, 1), dtype=jnp.float32)], axis=-1
    )  # -> (B, 2)
        prev_steps = int(state.info['steps'][0].item())
        if prev_steps == 108:
            pass
        state = wrapped.step(state, action_b)

        if int(state.done[0].item()) == 1:
            trunc = int(state.info['truncation'][0].item())
            print(f"[B=1] Episode terminated at step {prev_steps + 1} "
                  f"(reward={float(state.reward[0]):.1f}, truncation={trunc}).")
            print("Next episode starts at obs:", state.obs[0])
            break

if __name__ == "__main__":
    main()