import jax.numpy as jnp
import jax.random as jr
from typing import Tuple
import wandb
import types
import chex
import matplotlib.pyplot as plt

from brax.training.types import Transition
from brax.training.replay_buffers import UniformSamplingQueue, ReplayBufferState

from bsm.utils.normalization import Data
from bsm.utils import StatisticalModelState
from mbrl.model_based_agent.base_model_based_agent import ModelBasedAgentState
from diff_smoothers.smoother_net import SmootherNet
from diff_smoothers.data_functions.data_output import plot_derivative_data
from mbrl.model_based_agent.base_agent_wrapper import BaseAgentWrapper

# What is unclean about this implementation?
# The agent has its own agent attribute (so self.agent is an agent in and of itself...)

class SmootherWrapper(BaseAgentWrapper):
    def __init__(self, agent_type,
                 smoother_net: SmootherNet,
                 state_data_source: str = 'smoother',
                 **kwargs):
        super().__init__(agent_type, **kwargs)
        self.state_data_source = state_data_source
        self.smoother_model = smoother_net

        self.collected_data_buffer = self.prepare_data_buffers()

    # Override the prepare_data_buffers method to include the collected data buffer
    def prepare_data_buffers(self) -> UniformSamplingQueue:
        dummy_sample = Transition(observation=jnp.zeros(self.env.observation_size,),
                                  action=jnp.zeros(self.env.action_size,),
                                  reward=jnp.array(0.0),
                                  discount=jnp.array(0.99),
                                  next_observation=jnp.zeros(self.env.observation_size,),
                                  extras={'state_extras': {'t': jnp.array(0.0),
                                                           'derivative': jnp.zeros(self.env.observation_size,),
                                                           'true_derivative': jnp.zeros(self.env.observation_size,)}},
                                  )
        collected_data_buffer = UniformSamplingQueue(
            max_replay_size=self.max_collected_data_in_buffer,
            dummy_data_sample=dummy_sample,
            sample_batch_size=1)
        return collected_data_buffer

    # Override the simulate_on_true_env method to include the smoother
    def simulate_on_true_env(self,
                             agent_state: ModelBasedAgentState,
                             ) -> Tuple[ModelBasedAgentState, Transition]:
        key_agent, key_reset, key_smoother = jr.split(agent_state.key, 3)
        env_state = self.env_interactor.reset(key=key_reset)
        optimizer_state = agent_state.optimizer_state
        interaction = self.env_interactor.generate_rollouts(env_state=env_state,
                                                            actor_state=optimizer_state,
                                                            actor=self.actor
                                                            )
        final_state, optimizer_state, transitions = interaction
        transitions = self._get_dx(key_smoother, transitions)
        collected_data_buffer_state = agent_state.optimizer_state.true_buffer_state
        collected_data_buffer_state = self.collected_data_buffer.insert(buffer_state=collected_data_buffer_state,
                                                                        samples=transitions)
        optimizer_state = optimizer_state.replace(true_buffer_state=collected_data_buffer_state)
        env_steps = agent_state.env_steps + self.num_envs * self.episode_length
        return ModelBasedAgentState(optimizer_state=optimizer_state, env_steps=env_steps, key=key_agent), transitions

    # Override the _update_statistical_model method to include logging of the dynamics model
    def _update_statistical_model(self,
                                  statistical_model_state: StatisticalModelState,
                                  collected_data_buffer_state: ReplayBufferState,
                                  key: chex.PRNGKey):
        # We prepare data to train from the collected_data_buffer
        data, log_data = self._collected_buffer_to_train_data(collected_data_buffer_state)
        if self.reset_statistical_model:
            statistical_model_state = self.statistical_model.init(key=key)
        new_statistical_model_state = self.statistical_model.update(
            stats_model_state=statistical_model_state,
            data=data)
        if self.log_to_wandb:
            # Plot the data the dynamics model was trained on
            fig, _ = self.smoother_model.plot_fit(log_data['t'].reshape(-1, 1),
                                         log_data['x'],
                                         log_data['x'],
                                         log_data['x_dot_est'],
                                         log_data['x_dot_true'],
                                         state_labels=[r'$cos(\theta)$', r'$sin(\theta)$', r'$\omega$']
                                         )
            wandb.log({'dynamics_model/data': fig})
            plt.close(fig)
            # Plot the training performance of the dynamics model
            inputs = data.inputs
            pred_dx = self.statistical_model.predict_batch(inputs, new_statistical_model_state)
            fig = plot_derivative_data(t=log_data['t'].reshape(-1, 1),
                                       x = data.inputs[:, :self.env.observation_size],
                                       x_dot_true = data.outputs,
                                       x_dot_est=pred_dx.mean,
                                       x_dot_est_std=pred_dx.epistemic_std,
                                       source='Dyn. Model',
                                       beta = pred_dx.statistical_model_state.beta,
                                       state_labels=[r'$-sin(\theta) \dot{\theta}$', r'$cos(\theta) \dot{\theta}$', r'$\ddot{\theta}$'],
                                       )
            wandb.log({'dynamics_model/fit': wandb.Image(fig)})
            plt.close(fig)
        return new_statistical_model_state
    
    def _collected_buffer_to_train_data(self,
                                        collected_buffer_state: ReplayBufferState):
        idx = jnp.arange(start=collected_buffer_state.sample_position, stop=collected_buffer_state.insert_position)
        all_data = jnp.take(collected_buffer_state.data, idx, axis=0, mode='wrap')
        all_transitions = self.collected_data_buffer._unflatten_fn(all_data)
        obs = all_transitions.observation
        actions = all_transitions.action
        inputs = jnp.concatenate([obs, actions], axis=-1)
        outputs = all_transitions.extras['state_extras']['derivative']
        log_data = {}
        t = all_transitions.extras['state_extras']['t']
        log_data['t'] = t
        log_data['x'] = obs
        log_data['x_dot_true'] = all_transitions.extras['state_extras']['true_derivative']
        log_data['x_dot_est'] = outputs
        return Data(inputs=inputs, outputs=outputs), log_data

    def _get_dx(self,
               key: jr.PRNGKey,
               transitions: Transition,
               ) -> Transition:
        # Split transition into trajectories without resets (based on the time)
        timestamps = transitions.extras['state_extras']['t']
        indices = []
        for dt, i in enumerate(jnp.diff(timestamps)):
            if dt < 0:
                indices.append(i)
        # Split based on the indices
        trajectories = self._split_transitions(transitions, indices)
        # Fit Smoother for each trajectory (or only on longest one)
        longest_trajectory = max(trajectories, key=lambda x: len(x.observation))
        inputs = longest_trajectory.extras['state_extras']['t'].reshape(-1, 1)
        true_dx = longest_trajectory.extras['state_extras']['true_derivative'].reshape(-1, self.env.observation_size)
        outputs = longest_trajectory.observation
        data = Data(inputs, outputs)
        # Get dx from Smoother
        model_states = self.smoother_model.train_new_smoother(key, data)
        pred_x = self.smoother_model.predict_batch(inputs, model_states)
        ders = self.smoother_model.derivative_batch(inputs, model_states)

        # Log the smoother performance to wandb as a plot
        if self.log_to_wandb:
            fig, _ = self.smoother_model.plot_fit(inputs, pred_x.mean, outputs, true_dx, ders.mean,
                                                  state_labels=[r'$cos(\theta)$', r'$sin(\theta)$', r'$\omega$'])
            wandb.log({'smoother/fit': fig})
            plt.close(fig)

        # Use the smoothed trajectory in the transitions
        if self.state_data_source == 'smoother':
            transition = longest_trajectory._replace(observation=pred_x.mean)
        else:
            transition = longest_trajectory

        # Add the derivative to the transitions
        state_extras = transition.extras
        state_extras['state_extras']['derivative'] = ders.mean
        transition = transition._replace(extras=state_extras)
        return transition

    def _split_transitions(self,
                           transition: Transition,
                           indices: list[int],
                           ) -> list[Transition]:
         # Ensure indices are sorted and unique
        indices = sorted(set(indices))
        trajectories = []
        # Split the separate parts
        observations = transition.observation._split(indices)
        actions = transition.action._split(indices)
        rewards = transition.reward._split(indices)
        discounts = transition.discount._split(indices)
        next_observations = transition.next_observation._split(indices)
        # Split all entries in 'state_extras'
        state_extras = {}
        for key, value in transition.extras['state_extras'].items():
            if isinstance(value, jnp.ndarray):
                state_extras[key] = value._split(indices)

        for i in range(len(indices)):
            trajectories.append(Transition(
                observation=observations[i],
                action=actions[i],
                reward=rewards[i],
                discount=discounts[i],
                next_observation=next_observations[i],
                extras={'state_extras': {key: value[i] for key, value in state_extras.items()}}
            ))
        # Add the last part
        trajectories.append(Transition(
            observation=observations[-1],
            action=actions[-1],
            reward=rewards[-1],
            discount=discounts[-1],
            next_observation=next_observations[-1],
            extras={'state_extras': {key: value[-1] for key, value in state_extras.items()}}
        ))
        return trajectories