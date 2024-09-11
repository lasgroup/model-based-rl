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
from diff_smoothers.Base_Differentiator import BaseDifferentiator
from diff_smoothers.data_functions.data_output import plot_derivative_data, plot_data, plot_prediction_data, plot_data_reward
from mbrl.model_based_agent.base_agent_wrapper import BaseAgentWrapper
from mbrl.utils.brax_utils import EnvInteractor

class DifferentiatingAgent(BaseAgentWrapper):
    def __init__(self, agent_type,
                 differentiator: BaseDifferentiator,
                 state_data_source: str = 'smoother',
                 **kwargs):
        """Agent that uses a differentiator to estimate the state derivatives.
        Args:
            agent_type: Type of the agent to be wrapped (has to inherit from BaseAgent)
            differentiator: Differentiator to estimate the state derivatives (from diff-smoothers library)
            state_data_source: Source of the state data for the agent
                    - 'smoother':   the x and x_dot from the smoother are used in the transitions
                    - 'true':       the true x and x_dot are used in the transitions
                    - 'both':       the true x and the smoother x_dot are used in the transitions
            measurement_dt_ratio: Ratio of the measurement dt to the system dt, resamples the transitions
            **kwargs: Keyword arguments for the agent
        """
        super().__init__(agent_type, **kwargs)
        self.state_data_source = state_data_source
        self.differentiator = differentiator
        
        if self.log_mode > 0:
            wandb.define_metric("dynamics_model/data", summary="min")

        # Override the env_interactor so as not to try to pull true_derivatives from the environment
        extra_fields = list(self.state_extras_ref.keys())
        extra_fields.remove('true_derivative')
        self.env_interactor = EnvInteractor(
            env=self.env,
            eval_env=self.eval_env,
            num_envs=self.num_envs,
            num_eval_envs=self.num_eval_envs,
            episode_length=self.episode_length,
            action_repeat=self.action_repeat,
            key = self.env_interactor.evaluator._key,
            deterministic_policy_for_data_collection=self.deterministic_policy_for_data_collection,
            extra_fields=extra_fields,
        )

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
        if self.log_mode > 1:
            # Plot the data the dynamics model was trained on
            fig = plot_data(t=log_data['t'].reshape(-1, 1),
                               x=log_data['x'],
                               u=log_data['u'],
                               x_dot=log_data['x_dot_est'],
                               title='Data used for training the dynamics model',
                               state_labels=self.env.state_labels)
            wandb.log({'dynamics_model/data': wandb.Image(fig)})
            plt.close(fig)
            # Plot the training performance of the dynamics model
            inputs = data.inputs
            pred_dx = self.statistical_model.predict_batch(inputs, new_statistical_model_state)
            fig = plot_derivative_data(t=log_data['t'].reshape(-1, 1),
                                       x = data.inputs[:, :self.env.observation_size],
                                       x_dot_true = log_data['x_dot_true'],
                                       x_dot_est=pred_dx.mean,
                                       x_dot_est_std=pred_dx.epistemic_std,
                                       source='Dyn. Model',
                                       x_dot_smoother=log_data['x_dot_est'],
                                       beta = pred_dx.statistical_model_state.beta,
                                       num_trajectory_to_plot=-1,
                                       state_labels=self.env.state_derivative_labels,
                                       )
            wandb.log({'dynamics_model/fit': wandb.Image(fig)})
            plt.close(fig)
        if self.log_mode > 0:
            inputs = data.inputs
            pred_x  = self.statistical_model.predict_batch(inputs, new_statistical_model_state)
            dyn_mse = jnp.power((pred_x.mean - log_data['x_dot_true']), 2).mean(axis=0)
            wandb.log({"dynamics_model/error_mse": dyn_mse})
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
        log_data['u'] = actions
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
        # Resample the trajectories based on the measurement_dt_ratio
        if self.measurement_dt_ratio > 1:
            indices = range(0, len(longest_trajectory.observation), self.measurement_dt_ratio)
            longest_trajectory = self._resample_trajectories(longest_trajectory, indices)
        # Prepare the data for the differentiator
        inputs = longest_trajectory.extras['state_extras']['t'].reshape(-1, 1)
        true_dx = longest_trajectory.extras['state_extras']['derivative'].reshape(-1, self.env.observation_size)
        outputs = longest_trajectory.observation
        data = Data(inputs, outputs)
        # Get dx from Smoother
        differentiator_state = self.differentiator.train(key, data)
        differentiator_state, pred_x = self.differentiator.predict(differentiator_state, inputs)
        differentiator_state, ders = self.differentiator.differentiate(differentiator_state, inputs)

        # Log the smoother performance to wandb as a plot
        if self.log_mode > 1:
            fig, _ = self.differentiator.plot_fit(true_t=inputs,
                                                  pred_x=pred_x,
                                                  true_x=outputs,
                                                  pred_x_dot=ders,
                                                  true_x_dot=true_dx,
                                                  state_labels=self.env.state_labels)
            wandb.log({'smoother/fit': wandb.Image(fig)})
            plt.close(fig)

        # Use the smoothed trajectory in the transitions
        if self.state_data_source == 'smoother':
            transition = Transition(
                observation=pred_x,
                action=longest_trajectory.action,
                reward=longest_trajectory.reward,
                discount=longest_trajectory.discount,
                next_observation=longest_trajectory.next_observation,
                extras={'state_extras': {'t': longest_trajectory.extras['state_extras']['t'],
                                         'true_derivative': longest_trajectory.extras['state_extras']['derivative'],
                                         'derivative': ders,
                                         'dt': longest_trajectory.extras['state_extras']['dt']}}
            )
        elif self.state_data_source == 'true':
            transition = Transition(
                observation=longest_trajectory.observation,
                action=longest_trajectory.action,
                reward=longest_trajectory.reward,
                discount=longest_trajectory.discount,
                next_observation=longest_trajectory.next_observation,
                extras={'state_extras': {'t': longest_trajectory.extras['state_extras']['t'],
                                         'true_derivative': longest_trajectory.extras['state_extras']['derivative'],
                                         'derivative': longest_trajectory.extras['state_extras']['derivative'],
                                         'dt': longest_trajectory.extras['state_extras']['dt']}}
            )
        elif self.state_data_source == 'both':
            transition = Transition(
                observation=longest_trajectory.observation,
                action=longest_trajectory.action,
                reward=longest_trajectory.reward,
                discount=longest_trajectory.discount,
                next_observation=longest_trajectory.next_observation,
                extras={'state_extras': {'t': longest_trajectory.extras['state_extras']['t'],
                                         'true_derivative': longest_trajectory.extras['state_extras']['derivative'],
                                         'derivative': ders,
                                         'dt': longest_trajectory.extras['state_extras']['dt']}}
            )
        else:
            raise ValueError(f"Unknown state_data_source {self.state_data_source}")
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
    
    def plot_evaluation_data(self,
                             agent_state: ModelBasedAgentState,
                             data: Transition,
                             episode_idx: int):
        # Check what the dynamics model is predicting
        x_est = jnp.zeros((data.observation.shape[0], data.observation.shape[2]))
        x_dot_est = jnp.zeros((data.observation.shape[0], data.observation.shape[2]))
        input1 = jnp.concatenate([data.observation[0,:], data.action[0,:]], axis=-1).squeeze()
        x_est = x_est.at[0,:].set(data.observation[0,:].squeeze())
        initial_model_output = self.statistical_model(input1, agent_state.optimizer_state.system_params.dynamics_params.statistical_model_state)
        x_dot_est = x_dot_est.at[0,:].set(initial_model_output.mean.squeeze())
        for i in range(1, len(data.extras['state_extras']['t'])):
            new_x_est = x_est[i-1,:] + x_dot_est[i-1,:] * self.dynamics_dt
            x_est = x_est.at[i,:].set(new_x_est.squeeze())
            model_outputs = self.statistical_model(jnp.concatenate([x_est[i,:], data.action[i,:].reshape(-1,)]),
                                                            agent_state.optimizer_state.system_params.dynamics_params.statistical_model_state)
            x_dot_est = x_dot_est.at[i,:].set(model_outputs.mean)
        # Plot the predicted and true dynamics
        fig = plot_prediction_data(t=data.extras['state_extras']['t'].reshape(-1,1),
                                   x_true=data.observation.reshape(-1, data.observation.shape[-1]),
                                   x_est=x_est,
                                   x_est_std=jnp.zeros_like(x_est),
                                   beta=jnp.zeros((data.observation.shape[-1])),
                                   state_labels=self.env.state_labels,
                                   source='dyn.model')
        wandb.log({'eval_true_env/dyn_model_comparison': wandb.Image(fig)})
        plt.close(fig)
        # Plot the model predicted dynamics OFF OF THE TRUE STATE!
        pred_dx = self.statistical_model.predict_batch(jnp.concatenate([data.observation.reshape(-1, data.observation.shape[-1]), data.action.reshape(-1, data.action.shape[-1])], axis=-1),
                                        agent_state.optimizer_state.system_params.dynamics_params.statistical_model_state)
        fig = plot_derivative_data(t=data.extras['state_extras']['t'].reshape(-1, 1),
                                   x = data.observation.reshape(-1, data.observation.shape[-1]),
                                   x_dot_true = data.extras['state_extras']['derivative'].reshape(-1, data.observation.shape[-1]),
                                   x_dot_est=pred_dx.mean,
                                   x_dot_est_std=pred_dx.epistemic_std,
                                   source='Dyn. Model',
                                   beta = pred_dx.statistical_model_state.beta,
                                   num_trajectory_to_plot=-1,
                                   state_labels=self.env.state_derivative_labels,
                                   )
        wandb.log({'eval_true_env/dyn_model_derivative': wandb.Image(fig)})
        plt.close(fig)
        # Plot the actual actions and observations
        fig = plot_data_reward(t=data.extras['state_extras']['t'].reshape(-1, 1),
                               x=data.observation.reshape(-1, data.observation.shape[-1]),
                               reward=data.reward.reshape(-1, 1),
                               u=data.action.reshape(-1, data.action.shape[-1]),
                               title='Evaluation on True Env',
                               state_labels=self.env.state_labels,)
        wandb.log({'eval_true_env/evaluation_data': wandb.Image(fig)})
        plt.close(fig)