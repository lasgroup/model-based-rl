from abc import ABC, abstractmethod

import chex
import jax.numpy as jnp
import jax.random as jr
from brax import base
from brax.envs.base import Env, State
from jaxtyping import Float, Array
from brax.training.types import Transition
import jax
import numpy as np
import json
import glob
from jax import vmap

from mbrl.envs.pendulum import PendulumEnv
from mbrl.envs.pendulum_ct import ContinuousPendulumEnv
from wtc.wrappers.ih_switching_cost import IHSwitchCostWrapper, ConstantSwitchCost, AugmentedPipelineState

from diff_smoothers.Base_Differentiator import BaseDifferentiator
from diff_smoothers.data_functions.data_creation import create_random_control_sequence
from diff_smoothers.data_functions.data_output import plot_data
from bsm.utils.normalization import Data
import matplotlib.pyplot as plt
import os

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

        first_info: dict = {'t': jnp.zeros(shape=(num_samples,)),}
        brax_state = State(pipeline_state=jnp.zeros(shape=(num_samples,)),
                           obs=states,
                           reward=jnp.zeros(shape=(num_samples,)),
                           done=jnp.zeros(shape=(num_samples,)),
                           info=first_info)

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

class DifferentiatorOfflineData(OfflineData):
    def __init__(self,
                 differentiator: BaseDifferentiator,
                 env: Env,
                 init_state_range: chex.Array):
        """Offline Data Generator with a differentiator
        Args:
            differentiator (BaseDifferentiator): Differentiator to fit
            env (Env): Environment to sample data from
            init_state_range (chex.Array): Range of initial states to sample from, dimension (2, observation_size)"""
        super().__init__(env=env)
        self.differentiator = differentiator
        self.init_state_range = init_state_range

    def _sample_states(self,
                       key: chex.PRNGKey,
                       num_samples: int) -> Float[Array, 'dim_batch dim_state']:
        key, init_key = jr.split(key)
        state_init_noise = jr.uniform(init_key, shape=(num_samples, self.env.observation_size), minval=-1, maxval=1)
        def f(noise):
            return self.init_state_range[0,:] + (self.init_state_range[1,:] - self.init_state_range[0,:]) * noise
        return vmap(f)(state_init_noise)
        

    def _sample_actions(self,
                        key: chex.PRNGKey,
                        num_samples: int) -> Float[Array, 'dim_batch dim_action']:
        actions = jr.uniform(key, shape=(num_samples, self.env.action_size), minval=-1, maxval=1)
        return actions
    
    def _sample_colored_actions(self, num_trajectories: int, num_points: int,
                                key: jr.PRNGKey, col_noise_exponent: float = 3) -> Array:
        return create_random_control_sequence(num_points=num_points,
                                              key=key, control_dim=num_trajectories,
                                              col_noise_exponent=col_noise_exponent)
    
    def sample_transitions(self,
                           key: Array,
                           num_samples: int,
                           trajectory_length: int,
                           plot_results: bool = False,
                           measurement_dt_ratio: int = 1,
                           state_data_source: str = 'smoother',
                           pathname: str = './results/offline_data',
                           ) -> Transition:
        num_trajectories = num_samples // trajectory_length * measurement_dt_ratio
        key_states, key_actions = jr.split(key, 2)
        init_states = self._sample_states(key_states, num_trajectories)
        colored_actions = self._sample_colored_actions(num_trajectories*self.env.action_size,
                                                       trajectory_length+measurement_dt_ratio,
                                                       key_actions).T
        # Scale the actions to -1 and 1
        def scale_actions(actions: chex.Array) -> chex.Array:
            return 2 * (actions - actions.min()) / (actions.max() - actions.min()) - 1
        colored_actions = vmap(scale_actions, in_axes=0, out_axes=0)(colored_actions)
        colored_actions = colored_actions.reshape(num_trajectories, self.env.action_size, -1)
        def run_full_sim(init_state, colored_action):
            # init_state has shape (observation_size,), colored_action has shape (action_size, trajectory_length)
            # Here we could use the init_state, but that does not work for all environments! (e.g. rccar)
            brax_state = self.env.reset()
            def f(carry, xs):
                next_state = self.env.step(carry, xs)
                return next_state, next_state
            _, next_states = jax.lax.scan(f, brax_state, colored_action.T)
            return next_states
        next_states = vmap(run_full_sim, in_axes=(0,0))(init_states, colored_actions)
        states = jnp.concatenate([init_states.reshape(num_trajectories, 1, -1), next_states.obs[:,:-1,:]], axis=1)
        # Resample the measurements according to the measurement_dt_ratio
        indices = jnp.arange(0, trajectory_length+measurement_dt_ratio, measurement_dt_ratio)

        # x and t are longer by 1, to be able to use next_state from smoother
        t = jnp.tile(indices*self.env.dt, (num_trajectories, 1)).reshape(num_trajectories, -1, 1)
        x = states.take(indices=indices, axis=1)

        u = colored_actions.take(indices=indices[:-1], axis=2).transpose(0, 2, 1)
        x_dot_true = next_states.info['derivative'].take(indices=indices[:-1], axis=1)
        next_states = states.take(indices=indices[:-1]+measurement_dt_ratio, axis=1)

        # Fit the differentiator
        if plot_results:
            fig = plot_data(t=t[:,:-1,:],
                            x=x[:,:-1,:],
                            u=u,
                            x_dot=x_dot_true,
                            title='Offline Data',
                            state_labels=self.env.state_labels)
            if not os.path.exists(pathname):
                os.makedirs(pathname)
            plt.savefig(f'{pathname}/colored_data.png')
            plt.close(fig)
        differentiator_keys = jr.split(key, num_trajectories)
        smoothed_state = jnp.zeros_like(x)
        smoothed_derivative = jnp.zeros_like(x)
        # Cant do vmapping here, since the differentiator is stateful
        for k01 in range(num_trajectories):
            data = Data(inputs=t[k01,:,:], outputs=x[k01,:,:])
            differentiator_state = self.differentiator.train(key=differentiator_keys[k01], data=data)
            differentiator_state, pred_x = self.differentiator.predict(state=differentiator_state, t=data.inputs)
            differentiator_state, pred_x_dot = self.differentiator.differentiate(state=differentiator_state, t=data.inputs)
            smoothed_state = smoothed_state.at[k01, :, :].set(pred_x)
            smoothed_derivative = smoothed_derivative.at[k01, :, :].set(pred_x_dot)
            if plot_results:
                fig, _ = self.differentiator.plot_fit(true_t=data.inputs[:-1,:],
                                                        pred_x=pred_x[:-1,:],
                                                        true_x=data.outputs[:-1,:],
                                                        pred_x_dot=pred_x_dot[:-1,:],
                                                        true_x_dot=x_dot_true[k01,:,:],
                                                        state_labels=self.env.state_labels)
                plt.savefig(f'{pathname}/trajectory_{k01}.png')
                plt.close(fig)

        total_num_samples = smoothed_state.shape[0]*(smoothed_state.shape[1] - 1)
        if state_data_source == 'smoother':
            transitions = Transition(observation=smoothed_state[:,:-1,:].reshape(total_num_samples, self.env.observation_size),
                              action=u.reshape(total_num_samples, self.env.action_size),
                              reward=jnp.zeros(shape=(total_num_samples,)),
                              discount=jnp.zeros(shape=(total_num_samples,)),
                              next_observation=smoothed_state[:,1:,:].reshape(total_num_samples, self.env.observation_size),
                              extras={'state_extras': {'t': t[:,:-1,:].reshape(total_num_samples, 1),
                                                       'derivative': smoothed_derivative[:,:-1,:].reshape(total_num_samples, self.env.observation_size),
                                                       'dt': jnp.ones((total_num_samples, 1)) * self.env.dt * measurement_dt_ratio,
                                                       'true_derivative': x_dot_true.reshape(total_num_samples, self.env.observation_size)},})
        elif state_data_source == 'true':
            transitions = Transition(observation=x[:,:-1,:].reshape(total_num_samples, self.env.observation_size),
                                     action=u.reshape(total_num_samples, self.env.action_size),
                                     reward=jnp.zeros(shape=(total_num_samples,)),
                                     discount=jnp.zeros(shape=(total_num_samples,)),
                                     next_observation=x[:,1:,:].reshape(total_num_samples, self.env.observation_size),
                                     extras={'state_extras': {'t': t[:,:-1,:].reshape(total_num_samples, 1),
                                                              'derivative': x_dot_true.reshape(total_num_samples, self.env.observation_size),
                                                              'dt': jnp.ones((total_num_samples, 1)) * self.env.dt * measurement_dt_ratio,
                                                              'true_derivative': x_dot_true.reshape(total_num_samples, self.env.observation_size)},})
            
        elif state_data_source == 'discrete':
            transitions = Transition(observation=x[:,:-1,:].reshape(total_num_samples, self.env.observation_size),
                                     action=u.reshape(total_num_samples, self.env.action_size),
                                     reward=jnp.zeros(shape=(total_num_samples,)),
                                     discount=jnp.zeros(shape=(total_num_samples,)),
                                     next_observation=x[:,1:,:].reshape(total_num_samples, self.env.observation_size),
                                     extras={'state_extras': {'t': t[:,:-1,:].reshape(total_num_samples, 1),
                                                              'derivative': x_dot_true.reshape(total_num_samples, self.env.observation_size),
                                                              'dt': jnp.ones((total_num_samples, 1)) * self.env.dt * measurement_dt_ratio},})

        
        return transitions


        


class WhenToControlWrapper(OfflineData):

    def __init__(self,
                 num_integrator_steps: int,
                 min_time_between_switches,
                 max_time_between_switches
                 ):
        base_env = PendulumEnv(reward_source='dm-control')
        env = IHSwitchCostWrapper(base_env,
                                  num_integrator_steps=num_integrator_steps,
                                  min_time_between_switches=min_time_between_switches,
                                  max_time_between_switches=max_time_between_switches,
                                  switch_cost=ConstantSwitchCost(value=jnp.array(0.0)),
                                  time_as_part_of_state=True,
                                  discounting=0.99
                                  )
        super().__init__(env=env)

    def _sample_states(self,
                       key: chex.PRNGKey,
                       num_samples: int) -> Float[Array, 'dim_batch dim_state']:
        key_angle, key_angular_velocity = jr.split(key)
        angles = jr.uniform(key_angle, shape=(num_samples,), minval=-jnp.pi, maxval=jnp.pi)
        cos, sin = jnp.cos(angles), jnp.sin(angles)
        angular_velocity = jr.uniform(key_angular_velocity, shape=(num_samples,), minval=-5, maxval=5)
        env_times = jnp.zeros(shape=(num_samples,))
        return jnp.stack([cos, sin, angular_velocity, env_times], axis=-1)

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

def save_array_to_file(array, filename):
    """Helper function to save a JAX array to a .npz file."""
    np.savez_compressed(filename, array=array)

def save_transitions(transitions: Transition, filename: str):
    """Save transitions to a file, including all the extras."""
    # Save the main transition data
    with open(filename, 'wb') as f:
        jnp.savez(f, observation=transitions.observation,
                  action=transitions.action,
                  reward=transitions.reward,
                  discount=transitions.discount,
                  next_observation=transitions.next_observation)

    # Save each entry in the extras dictionary
    extras = transitions.extras['state_extras']
    filename = filename.split('.')[0]

    def save_dict(d, prefix):
        """Recursively save dictionaries."""
        for key, value in d.items():
            if isinstance(value, dict):
                # Recursively save nested dictionaries
                save_dict(value, f'{prefix}-{key}')
            elif isinstance(value, jnp.ndarray):
                # Save arrays
                save_array_to_file(value, f'{filename}-{prefix}-{key}.npz')
            else:
                raise TypeError(f"Unsupported type {type(value)} for key {key}")

    save_dict(extras, 'state_extras')
        
def load_array_from_file(filename):
    """Helper function to load a JAX array from a .npz file."""
    with np.load(filename) as npz_file:
        return jax.device_put(npz_file['array'])  # Convert numpy array back to JAX array

def load_transitions(filename: str) -> Transition:
    """Load transitions from a file, including all the extras."""
    # Load the main transition data
    with open(filename, 'rb') as f:
        data = jnp.load(f, allow_pickle=True)
        observation = data['observation']
        action = data['action']
        reward = data['reward']
        discount = data['discount']
        next_observation = data['next_observation']
    
    # Load the extras dictionary
    extras = {}
    filename = filename.split('.')[0]
    

    def load_dict(prefix):
        """Recursively load dictionaries."""
        for file in glob.glob(f'{filename}-{prefix}-*.npz'):
            key = os.path.basename(file).split(f'{prefix}-')[1].replace('.npz', '')
            extras[key] = load_array_from_file(file)
        
        # Handle nested dictionaries
        for key in list(extras.keys()):
            if isinstance(extras[key], dict):
                # Recursively load nested dictionaries
                nested_prefix = f'{prefix}_{key}'
                extras[key] = load_dict(nested_prefix)
        
        return extras
    
    extras = load_dict('state_extras')
    extras = {'state_extras': extras}
    
    return Transition(observation=observation,
                      action=action,
                      reward=reward,
                      discount=discount,
                      next_observation=next_observation,
                      extras=extras)

if __name__ == '__main__':
    offline_data_gen = WhenToControlWrapper()
    key = jr.PRNGKey(0)

    data = offline_data_gen.sample_transitions(key=key, num_samples=5)
