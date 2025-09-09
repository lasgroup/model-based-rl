from abc import abstractmethod
from functools import partial
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from brax.envs.base import PipelineEnv, State, Env, base
from jax import jit
from jax.lax import while_loop, scan
from jaxtyping import Float, Array

EPS = 1e-10


class AugmentedPipelineState(NamedTuple):
    pipeline_state: base.State
    time: Float[Array, 'None']


class SwitchCost:
    @abstractmethod
    def __call__(self,
                 state: Float[Array, 'observation_dim'],
                 action: Float[Array, 'action_dim']) -> Float[Array, 'None']:
        pass


class ConstantSwitchCost(SwitchCost):

    def __init__(self, value: Float[Array, 'None']):
        self.value = value

    @partial(jit, static_argnums=(0,))
    def __call__(self,
                 state: Float[Array, 'observation_dim'],
                 action: Float[Array, 'action_dim']) -> Float[Array, 'None']:
        return self.value


class IHSwitchCostWrapper(Env):
    def __init__(self,
                 env: PipelineEnv,
                 num_integrator_steps: int,
                 min_time_between_switches: float,
                 max_time_between_switches: float | None = None,
                 switch_cost: SwitchCost = ConstantSwitchCost(value=jnp.array(1.0)),
                 discounting: float = 0.99,
                 time_as_part_of_state: bool = False,
                 ):
        self.env = env
        self.num_integrator_steps = num_integrator_steps
        self.switch_cost = switch_cost
        self.min_time_between_switches = min_time_between_switches
        assert min_time_between_switches >= self.env.dt, \
            'Min time between switches must be at least of the integration time dt'
        self.time_horizon = self.env.dt * self.num_integrator_steps
        if max_time_between_switches is None:
            max_time_between_switches = self.time_horizon
        self.max_time_between_switches = max_time_between_switches
        self.discounting = discounting
        self.time_as_part_of_state = time_as_part_of_state
        self.jitted_step_fn = jit(self.env.step)

    def reset(self, rng: jax.Array) -> State:
        """
        The augmented state is represented by concatenated vector:
         (state, time-to-go)
        """
        state = self.env.reset(rng)
        time = jnp.array(0.0)
        if self.time_as_part_of_state:
            augmented_obs = jnp.concatenate([state.obs, time.reshape(1)])
            augmented_state = state.replace(obs=augmented_obs)
        else:
            augmented_pipeline_state = AugmentedPipelineState(pipeline_state=state.pipeline_state,
                                                              time=time)
            augmented_state = state.replace(pipeline_state=augmented_pipeline_state)
        return augmented_state

    def compute_time(self,
                     pseudo_time: chex.Array,
                     dt: chex.Array,
                     t_lower: chex.Array,
                     t_upper: chex.Array,
                     ) -> chex.Array:
        time_for_action = ((t_upper - t_lower) / 2 * pseudo_time + (t_upper + t_lower) / 2)
        return (time_for_action // dt) * dt

    def step(self, state: State, action: jax.Array) -> State:
        u, pseudo_time_for_action = action[:-1], action[-1]
        if self.time_as_part_of_state:
            obs, time = state.obs[:-1], state.obs[-1]
        else:
            env_pipeline_state = state.pipeline_state.pipeline_state
            time = state.pipeline_state.time

        # Calculate the action time, i.e. Map pseudo_time_for_action from [-1, 1] to
        # time [self.min_time_between_switches, self.max_time_between_switches]
        time_for_action = self.compute_time(pseudo_time=pseudo_time_for_action,
                                            dt=self.env.dt,
                                            t_lower=self.min_time_between_switches,
                                            t_upper=self.max_time_between_switches,
                                            )

        done = time_for_action >= self.time_horizon - time
        # Calculate how many steps we need to take with action
        num_steps = jnp.minimum(time_for_action, self.time_horizon - time) // self.env.dt

        # Integrate dynamics forward for the num_steps
        if self.time_as_part_of_state:
            state = state.replace(obs=obs)
        else:
            state = state.replace(pipeline_state=env_pipeline_state)

        def body_integration_step(val):
            s, r, index = val
            next_state = self.env.step(s, u)
            # CORRECTED: Simply add the reward from the base environment
            next_reward = r + (self.discounting ** index) * next_state.reward
            return next_state, next_reward, index + 1

        def cond_integration_step(val):
            s, r, index = val
            # We continue if index is smaller that num_steps ant we are not done
            return jnp.bitwise_and(index < num_steps, jnp.bitwise_not(s.done.astype(bool)))

        init_val = (state, jnp.array(0.0), jnp.array(0))
        final_val = while_loop(cond_integration_step, body_integration_step, init_val)
        next_state, total_reward, _ = final_val
        next_done = 1 - (1 - next_state.done) * (1 - done)

        # Add switch cost to the total reward
        total_reward = total_reward - self.switch_cost(state=state.obs, action=u)

        # Prepare augmented obs
        next_time = (time + time_for_action).reshape(1)
        if self.time_as_part_of_state:
            augmented_next_obs = jnp.concatenate([next_state.obs, next_time])
            augmented_next_state = next_state.replace(obs=augmented_next_obs,
                                                      reward=total_reward,
                                                      done=next_done)
            return augmented_next_state
        else:
            augmented_pipeline_state = AugmentedPipelineState(pipeline_state=next_state.pipeline_state,
                                                              time=next_time.reshape())
            augmented_next_state = next_state.replace(reward=total_reward,
                                                      done=next_done,
                                                      pipeline_state=augmented_pipeline_state)
            return augmented_next_state

    def simulation_step(self, state: State, action: jax.Array) -> (State, State):
        u, pseudo_time_for_action = action[:-1], action[-1]
        if self.time_as_part_of_state:
            obs, time = state.obs[:-1], state.obs[-1]
        else:
            env_pipeline_state = state.pipeline_state.pipeline_state
            time = state.pipeline_state.time

        # Calculate the action time, i.e. Map pseudo_time_for_action from [-1, 1] to
        # time [self.min_time_between_switches, time_to_go]
        time_for_action = self.compute_time(pseudo_time=pseudo_time_for_action,
                                            dt=self.env.dt,
                                            t_lower=self.min_time_between_switches,
                                            t_upper=self.max_time_between_switches)
        done = time_for_action >= self.time_horizon - time

        # Calculate how many steps we need to take with action
        num_steps = jnp.minimum(time_for_action, self.time_horizon - time) // self.env.dt

        # Integrate dynamics forward for the num_steps
        if self.time_as_part_of_state:
            state = state.replace(obs=obs)
        else:
            state = state.replace(pipeline_state=env_pipeline_state)

        # Execute the action for the predicted number of integration steps
        step_index = 0
        cur_state = state
        all_states = []
        while step_index < num_steps and not cur_state.done:
            cur_state = self.jitted_step_fn(cur_state, u)
            all_states.append(cur_state)
            step_index += 1

        next_state = cur_state
        if len(all_states) == 0:
            all_states = [cur_state]
        inner_part = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *all_states)
        total_reward = jnp.sum(inner_part.reward)
        next_done = 1 - (1 - next_state.done) * (1 - done)

        # Add switch cost to the total reward
        total_reward = total_reward - self.switch_cost(state=state.obs, action=u)

        # Prepare augmented obs
        next_time = (time + time_for_action).reshape(1)
        if self.time_as_part_of_state:
            augmented_next_obs = jnp.concatenate([next_state.obs, next_time])
            augmented_next_state = next_state.replace(obs=augmented_next_obs,
                                                      reward=total_reward,
                                                      done=next_done)
            return augmented_next_state, inner_part
        else:
            augmented_pipeline_state = AugmentedPipelineState(pipeline_state=next_state.pipeline_state,
                                                              time=next_time.reshape())
            augmented_next_state = next_state.replace(reward=total_reward,
                                                      done=next_done,
                                                      pipeline_state=augmented_pipeline_state)
            return augmented_next_state, inner_part

    @property
    def observation_size(self) -> int:
        # +1 for time-to-go ant +1 for num remaining switches
        if self.time_as_part_of_state:
            return self.env.observation_size + 1
        else:
            return self.env.observation_size

    @property
    def action_size(self) -> int:
        # +1 for time that we apply action for
        return self.env.action_size + 1

    @property
    def backend(self) -> str:
        return self.env.backend

    @property
    def dt(self):
        return self.env.dt


if __name__ == '__main__':
    from brax import envs
    import jax.random as jr
    from jax import jit

    env_name = 'inverted_pendulum'
    backend = 'generalized'

    env = envs.get_environment(env_name=env_name,
                               backend=backend)

    env = IHSwitchCostWrapper(env,
                              num_integrator_steps=1000,
                              min_time_between_switches=env.dt,
                              # max_time_between_switches=10 * env.dt,
                              switch_cost=ConstantSwitchCost(value=jnp.array(1.0)),
                              discounting=1.0)

    key = jr.PRNGKey(42)
    key, subkey = jr.split(key)
    state = env.reset(subkey)

    wrapper = True

    u = jnp.array([0.1])
    time = jnp.array([0.0])
    augmented_action = jnp.concatenate([u, time]) if wrapper else u

    state, rest = env.simulation_step(state, augmented_action)
    state = env.step(state, augmented_action)
    # jitted_step = jit(env.step)

    # import time
    #
    # for i in range(10):
    #     start_time = time.time()
    #     state = jitted_step(state, augmented_action)
    #     print(f'elapsed_time: {time.time() - start_time} sec')
