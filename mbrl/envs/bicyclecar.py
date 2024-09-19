import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from brax import base
from brax.envs.base import State, Env
import chex

from jaxtyping import Float, Array
from functools import partial
from typing import Optional
import copy

from mbrl.utils.bicyclecar_utils import BicycleCarModel, BicycleCarReward

OBS_NOISE_STD_SIM_CAR: jnp.array = 0.1 * jnp.exp(jnp.array([-4.5, -4.5, -4., -2.5, -2.5, -1.]))

class BicycleEnv(Env):
    def __init__(self,
                 dynamics_model: BicycleCarModel = BicycleCarModel(dt=1/30.),
                 reward_model: BicycleCarReward = BicycleCarReward(),
                 init_noise_key: chex.PRNGKey = None,
                 render_mode : str = 'rgb_array',
                 use_obs_noise: bool = False,
                 init_pos: Optional[chex.PRNGKey] = jnp.array([1.42, -1.04, jnp.pi]),):
        self.render_mode = render_mode
        self.reward_model = reward_model
        self.dynamics_model = dynamics_model
        self.goal = jnp.asarray(self.reward_model.goal)
        self.init_noise_key = init_noise_key
        self.init_pos = init_pos
        self.use_obs_noise = use_obs_noise
        self._obs_noise_stds = OBS_NOISE_STD_SIM_CAR

        self.state_labels = [r'$x$', r'$y$', r'$sin(\theta)$', r'$cos(\theta)$', r'$\dot{x}$', r'$\dot{y}$', r'$\dot{\theta}$']
        self.state_derivative_labels = [r'$\dot{x}$', r'$\dot{y}$', r'$cos(\theta) \dot{\theta}$', r'$-sin(\theta) \dot{\theta}$', r'$\ddot{x}$', r'$\ddot{y}$', r'$\ddot{\theta}$']

        self.init_state = jnp.concatenate([self.init_pos[..., :2],
                                           jnp.sin(self.init_pos[2]).reshape(-1),
                                           jnp.cos(self.init_pos[2]).reshape(-1),
                                           jnp.zeros(3)])
        
    def reset(self,
              rng: jax.Array | None = None) -> State:
        first_info: dict = {'derivative': jnp.zeros(7),
                            't': jnp.array(0.0),
                            'dt': jnp.array(self.dynamics_model.dt),
                            'noise_key': self.init_noise_key}
        return State(pipeline_state=self.init_state,
                     obs=self.init_state,
                     reward=jnp.array(0.0),
                     done=jnp.array(0.0),
                     info=first_info)
    
    def reward(self,
               x: Float[Array, 'observation_dim'],
               u: Float[Array, 'action_dim']) -> Float[Array, 'None']:
        return self.reward_model.predict(x, u)
    
    @partial(jax.jit, static_argnums=0)
    def step(self,
             state: State,
             action: jax.Array) -> State:
        
        next_state = self.dynamics_model.predict(state.pipeline_state, action)

        reward = self.reward(next_state, action)

        next_info = copy.deepcopy(state.info)
        next_info['t'] += self.dynamics_model.dt
        next_info['derivative'] = (next_state - state.pipeline_state) / self.dynamics_model.dt
        next_info['dt'] = self.dynamics_model.dt
        
        if self.use_obs_noise:
            key, subkey = jr.split(next_info['noise_key'])
            x_reduced = self.dynamics_model.reduce_x(next_state)
            obs_reduced = x_reduced + self._obs_noise_stds * jr.normal(subkey, x_reduced.shape)
            obs = jnp.concatenate([obs_reduced[0:self.dynamics_model.angle_idx],
                                   jnp.sin(obs_reduced[self.dynamics_model.angle_idx]).reshape(-1),
                                   jnp.cos(obs_reduced[self.dynamics_model.angle_idx]).reshape(-1),
                                   obs_reduced[self.dynamics_model.angle_idx + 1:]], axis=-1)
            next_info['noise_key'] = key
        else:
            obs = next_state

        return State(pipeline_state=next_state,
                     obs=obs,
                     reward=reward,
                     done=state.done,
                     metrics=state.metrics,
                     info=next_info)
    
    @property
    def dt(self):
        return self.dynamics_model.dt

    @property
    def observation_size(self) -> int:
        return 7

    @property
    def action_size(self) -> int:
        return 2

    def backend(self) -> str:
        return 'positional'



if __name__ == "__main__":
    env = BicycleEnv(use_obs_noise=True,
                     init_noise_key=jr.PRNGKey(0))
    state = env.reset()

    observations = []
    actions = []
    for i in range(70):
        action = jnp.array([1.0, 0.7])
        state = env.step(state, action)
        observations.append(state.obs)
        actions.append(action)

    # Plot multiple stuff in one figure (multiple panels)
    # Panel 1: x-y plot of the car's trajectory with the car's orientation
    # Panel 2: Plot the throttle input over time and the cars velocity over time
    # Panel 3: Plot the steering input over time and the cars angular velocity over time

    import matplotlib.pyplot as plt
    obs = jnp.stack(observations)
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    axs[0].plot(obs[:, 0], obs[:, 1], label='Car Position')
    axs[0].quiver(obs[:, 0], obs[:, 1], obs[:, 3], obs[:, 2], color='r')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_title('Car Position and Orientation')
    axs[0].legend()
    axs[0].grid()
    axs[0].axis('equal')

    actions = jnp.stack(actions)
    axs[1].plot(actions[:, 1], label='Throttle')
    axs[1].plot(obs[:, 4], label='Car Velocity X')
    axs[1].plot(obs[:, 5], label='Car Velocity Y')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Value')
    axs[1].set_title('Throttle and Car Velocity')
    axs[1].legend()

    axs[2].plot(actions[:, 0], label='Steering')
    axs[2].plot(obs[:, 6], label='Angular Velocity')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Value')
    axs[2].set_title('Steering and Angular Velocity')
    axs[2].legend()

    plt.savefig('bicycle_sim_env.png')