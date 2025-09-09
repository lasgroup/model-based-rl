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
        # Check if the episode was already done
        done = state.done

        # 1. Calculate the potential next state based on physics
        potential_next_obs = self.next_step(state.obs, action)
        
        # 2. Determine if this new potential state meets the termination condition
        pos = potential_next_obs[0]
        velocity = potential_next_obs[1]
        just_terminated = jnp.logical_and(pos >= self.env.goal_position, velocity >= self.env.goal_velocity)
        
        # 3. Calculate the reward for this step
        # The reward is 100.0 only if we *just* terminated. If the episode was already done, the reward is 0.
        reward = jnp.where(done, 0.0, 100.0 * just_terminated)

        # 4. Determine the actual next observation
        # If the episode was already done, the observation does not change ("freezes").
        # Otherwise, we update to the new potential observation.
        # We need to reshape `done` to broadcast correctly with the `obs` array.
        next_obs = jnp.where(done.reshape(-1), state.obs, potential_next_obs)

        # 5. Update the done flag
        # The episode is done if it was already done OR if it just terminated.
        next_done = jnp.logical_or(done, just_terminated).astype(jnp.float32)
        
        # Construct the final next state
        next_state = State(pipeline_state=state.pipeline_state,
                           obs=next_obs,
                           reward=reward,
                           done=next_done,
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
    from mbrl.utils.ih_switching_cost import IHSwitchCostWrapper, ConstantSwitchCost
    jax.config.update("jax_disable_jit", True)

    time_horizon = 10_000

    base_env = MountainCarEnv()
    env = IHSwitchCostWrapper(base_env,
                              num_integrator_steps=time_horizon,
                              min_time_between_switches=1,
                              max_time_between_switches=1,
                              switch_cost=ConstantSwitchCost(value=jnp.array(0.1)),
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
        if prev_steps == 107:
            pass
        state = wrapped.step(state, action_b)
        print(state.obs[0], float(state.reward[0]), int(state.done[0]), int(state.info['steps'][0]), int(state.info['truncation'][0]))

        if int(state.done[0].item()) == 1:
            trunc = int(state.info['truncation'][0].item())
            print(f"[B=1] Episode terminated at step {prev_steps + 1} "
                  f"(reward={float(state.reward[0]):.1f}, truncation={trunc}).")
            print("Next episode starts at obs:", state.obs[0])
            break

if __name__ == "__main__":
    main()