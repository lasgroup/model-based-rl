import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Dict, Any

from functools import partial
from brax.envs.base import State, Env
import chex
import copy

from mbrl.utils.rccar_utils import RaceCar, RCCarEnvReward, RCCarDynamicsParams, encode_angles

OBS_NOISE_STD_SIM_CAR: jnp.array = 0.1 * jnp.exp(jnp.array([-4.5, -4.5, -4., -2.5, -2.5, -1.]))

class RCCarSimEnv(Env):
    max_steps: int = 200
    _dt: float = 1 / 30.
    dim_action: Tuple[int] = (2,)
    _goal: jnp.array = jnp.array([0.0, 0.0, 0.0])
    _init_pose: jnp.array = jnp.array([1.42, -1.04, jnp.pi])
    _angle_idx: int = 2
    _obs_noise_stds: jnp.array = OBS_NOISE_STD_SIM_CAR

    def __init__(self, ctrl_cost_weight: float = 0.005, encode_angle: bool = False, use_obs_noise: bool = True,
                 use_tire_model: bool = False, action_delay: float = 0.0, car_model_params: dict = None,
                 margin_factor: float = 10.0, max_throttle: float = 1.0, car_id: int = 2, ctrl_diff_weight: float = 0.0,
                 seed: int = 230492394, max_steps: int = 200):
        """
        Race car simulator environment

        Args:
            ctrl_cost_weight: weight of the control penalty
            encode_angle: whether to encode the angle as cos(theta), sin(theta)
            use_obs_noise: whether to use observation noise
            use_tire_model: whether to use the (high-fidelity) tire model, if False just uses a kinematic bicycle model
            action_delay: whether to delay the action by a certain amount of time (in seconds)
            car_model_params: dictionary of car model parameters that overwrite the default values
            seed: random number generator seed
        """
        self.dim_state: Tuple[int] = (7,) if encode_angle else (6,)
        self.encode_angle: bool = encode_angle
        self._rds_key = jax.random.PRNGKey(seed)
        self.max_throttle = jnp.clip(max_throttle, 0.0, 1.0)
        self.max_steps = max_steps

        # set car id and corresponding parameters
        assert car_id in [1, 2]
        self.car_id = car_id
        self._set_car_params()

        # initialize dynamics and observation noise models
        self._dynamics_model = RaceCar(dt=self._dt, encode_angle=False)

        self.use_tire_model = use_tire_model
        if use_tire_model:
            self._default_car_model_params = self._default_car_model_params_blend
        else:
            self._default_car_model_params = self._default_car_model_params_bicycle

        if car_model_params is None:
            _car_model_params = self._default_car_model_params
        else:
            _car_model_params = self._default_car_model_params
            _car_model_params.update(car_model_params)
        self._dynamics_params = RCCarDynamicsParams(**_car_model_params)
        self._next_step_fn = jax.jit(partial(self._dynamics_model.next_step, params=self._dynamics_params))

        self.use_obs_noise = use_obs_noise

        # initialize reward model
        self._reward_model = RCCarEnvReward(goal=self._goal,
                                            ctrl_cost_weight=ctrl_cost_weight,
                                            encode_angle=self.encode_angle,
                                            margin_factor=margin_factor)

        # set up action delay
        assert action_delay >= 0.0, "Action delay must be non-negative"
        self.action_delay = action_delay
        if abs(action_delay % self._dt) < 1e-8:
            self._act_delay_interpolation_weights = jnp.array([1.0, 0.0])
        else:
            # if action delay is not a multiple of dt, compute weights to interpolate
            # between temporally closest actions
            weight_first = (action_delay % self._dt) / self._dt
            self._act_delay_interpolation_weights = jnp.array([weight_first, 1.0 - weight_first])
        action_delay_buffer_size = int(jnp.ceil(action_delay / self._dt)) + 1
        self._action_buffer = jnp.zeros((action_delay_buffer_size, self.dim_action[0]))

        # initialize time and state
        self._time: int = 0
        self._state: jnp.array = jnp.zeros(self.dim_state)
        self.ctrl_diff_weight = ctrl_diff_weight

        if self.encode_angle:
            self.state_labels = [r'$x$', r'$y$', r'$sin(\theta)$', r'$cos(\theta)$', r'$\dot{x}$', r'$\dot{y}$', r'$\dot{\theta}$']
            self.state_derivative_labels = [r'$\dot{x}$', r'$\dot{y}$', r'$cos(\theta) \dot{\theta}$', r'$-sin(\theta) \dot{\theta}$', r'$\ddot{x}$', r'$\ddot{y}$', r'$\ddot{\theta}$']
        else:
            self.state_labels = [r'$x$', r'$y$', r'$\theta$', r'$\dot{x}$', r'$\dot{y}$', r'$\dot{\theta}$']
            self.state_derivative_labels = [r'$\dot{x}$', r'$\dot{y}$', r'$\dot{\theta}$', r'$\ddot{x}$', r'$\ddot{y}$', r'$\ddot{\theta}$']

    def reset(self, rng_key: Optional[jax.random.PRNGKey] = None) -> jnp.array:
        """ Resets the environment to a random initial state close to the initial pose.
        State has always this shape:
        [x, y, theta, x_dot, y_dot, theta_dot]
        Observation varies, is either (based on encode_angle):
        [x, y, theta, x_dot, y_dot, theta_dot] or
        [x, y, sin(theta), cos(theta), x_dot, y_dot, theta_dot]"""

        if rng_key is not None:
            # sample random initial state
            key_pos, key_theta, key_vel, key_obs, noise_key = jax.random.split(rng_key, 5)
            init_pos = self._init_pose[:2] + jax.random.uniform(key_pos, shape=(2,), minval=-0.10, maxval=0.10)
            init_theta = self._init_pose[2:] + \
                        jax.random.uniform(key_theta, shape=(1,), minval=-0.10 * jnp.pi, maxval=0.10 * jnp.pi)
            init_vel = jnp.zeros((3,)) + jnp.array([0.005, 0.005, 0.02]) * jax.random.normal(key_vel, shape=(3,))
        else:
            key_obs, noise_key = jax.random.split(self._rds_key)
            init_pos = self._init_pose[:2]
            init_theta = self._init_pose[2:]
            init_vel = jnp.zeros((3,))
        init_state = jnp.concatenate([init_pos, init_theta, init_vel])

        first_info: dict = {'derivative': jnp.zeros(7) if self.encode_angle else jnp.zeros(6),
                            't': jnp.array(self._time),
                            'dt': jnp.array(self._dt),
                            'noise_key': noise_key}

        return State(pipeline_state=init_state,
                     obs=self._state_to_obs(init_state, rng_key=key_obs),
                     reward=jnp.array(0.0),
                     done=jnp.array(0.0),
                     info=first_info)

    @partial(jax.jit, static_argnums=0)
    def step(self,
             state: State,
             action: chex.Array) \
            -> State:
        """ Performs one step in the environment
        
        Observation:
        
            Num     Observation               Min                     Max
            0       x position                -inf                    inf
            1       y position                -inf                    inf
            2       theta                     -pi                     pi
            3       x velocity                -inf                    inf
            4       y velocity                -inf                    inf
            5       angular velocity          -inf                    inf
        If encode_angle is True, the observation is instead:
        
            Num     Observation               Min                     Max
            0       x position                -inf                    inf
            1       y position                -inf                    inf
            2       sin(theta)                -1                      1
            3       cos(theta)                -1                      1
            4       x velocity                -inf                    inf
            5       y velocity                -inf                    inf
            6       angular velocity          -inf                    inf
            
        Actions:
        
            Num     Action                    Min                     Max
            0       Steering                  -1                      1
            1       Throttle                  -1                      1"""
        x = state.pipeline_state
        assert action.shape[-1:] == self.dim_action
        action = jnp.clip(action, -1.0, 1.0)
        action = action.at[0].set(self.max_throttle * action[0])


        jitter_reward = jnp.zeros_like(action).sum(-1)
        if self.action_delay > 0.0:
            # pushes action to action buffer and pops the oldest action
            # computes delayed action as a linear interpolation between the relevant actions in the past
            action, jitter_reward = self._get_delayed_action(action)

        # compute next state
        new_x = self._next_step_fn(x, action)

        next_info = copy.deepcopy(state.info)
        comp_dx = (new_x - x) / self._dt
        dx = jnp.concatenate([comp_dx[:self._angle_idx],
                              (jnp.cos(x[self._angle_idx]) * comp_dx[self._angle_idx]).reshape(-1,),
                              (-1 * jnp.sin(x[self._angle_idx]) * comp_dx[self._angle_idx]).reshape(-1,),
                              comp_dx[self._angle_idx+1:]])
        next_info['derivative'] = dx
        next_info['t'] = state.info['t'] + self._dt
        next_info['dt'] = self._dt

        key, subkey = jax.random.split(state.info['noise_key'], 2)
        obs = self._state_to_obs(new_x, subkey)
        next_info['noise_key'] = key

        next_state = State(pipeline_state=new_x,
                           obs=obs if self.use_obs_noise else (encode_angles(new_x, self._angle_idx) if self.encode_angle else new_x),
                           reward=self._reward_model.forward(obs=None, action=action, next_obs=obs) + jitter_reward,
                           done=state.done,
                           metrics=state.metrics,
                           info=next_info)
        return next_state

    def _state_to_obs(self, state: jnp.array, rng_key: Optional[jax.random.PRNGKey] = None) -> jnp.array:
        """ Adds observation noise to the state """
        assert state.shape[-1] == 6

        # add observation noise
        if self.use_obs_noise:
            obs = state + self._obs_noise_stds * jax.random.normal(rng_key, shape=state.shape)
        else:
            obs = state

        # encode angle to sin(theta) and cos(theta) if desired
        if self.encode_angle:
            obs = encode_angles(obs, self._angle_idx)
        assert (obs.shape[-1] == 7 and self.encode_angle) or (obs.shape[-1] == 6 and not self.encode_angle)
        return obs

    def _get_delayed_action(self, action: jnp.array) -> Tuple[jnp.array, jnp.array]:
        # push action to action buffer
        last_action = self._action_buffer[-1]
        reward = - self.ctrl_diff_weight * jnp.sum((action - last_action) ** 2)
        self._action_buffer = jnp.concatenate([self._action_buffer[1:], action[None, :]], axis=0)

        # get delayed action (interpolate between two actions if the delay is not a multiple of dt)
        delayed_action = jnp.sum(self._action_buffer[:2] * self._act_delay_interpolation_weights[:, None], axis=0)
        assert delayed_action.shape == self.dim_action
        return delayed_action, reward

    @property
    def time(self) -> float:
        return self._time
    
    @property
    def dt(self) -> float:
        return self._dt
    
    @property
    def observation_size(self) -> int:
        return self.dim_state[-1]
    
    @property
    def action_size(self) -> int:
        return self.dim_action[-1]
    
    def backend(self) -> str:
        return 'positional'

    def _set_car_params(self):
        from mbrl.utils.rccar_utils import (DEFAULT_PARAMS_BICYCLE_CAR1, DEFAULT_PARAMS_BLEND_CAR1,
                                                      DEFAULT_PARAMS_BICYCLE_CAR2, DEFAULT_PARAMS_BLEND_CAR2)
        if self.car_id == 1:
            self._default_car_model_params_bicycle: Dict = DEFAULT_PARAMS_BICYCLE_CAR1
            self._default_car_model_params_blend: Dict = DEFAULT_PARAMS_BLEND_CAR1
        elif self.car_id == 2:
            self._default_car_model_params_bicycle: Dict = DEFAULT_PARAMS_BICYCLE_CAR2
            self._default_car_model_params_blend: Dict = DEFAULT_PARAMS_BLEND_CAR2
        else:
            raise NotImplementedError(f'Car idx {self.car_id} not supported')
        


if __name__ == "__main__":
    env = RCCarSimEnv(encode_angle=True,
                      use_tire_model=True)
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

    plt.savefig('rccar_sim_env.png')

