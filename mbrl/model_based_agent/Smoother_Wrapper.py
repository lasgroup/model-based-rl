import jax.numpy as jnp
import jax.random as jr
from typing import Tuple
import wandb
import types
import matplotlib.pyplot as plt

from brax.training.types import Transition
from brax.training.replay_buffers import UniformSamplingQueue

from bsm.utils.normalization import Data
from mbrl.model_based_agent.base_model_based_agent import ModelBasedAgentState
from diff_smoothers.smoother_net import SmootherNet
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
        dx, transitions = self._get_dx(key_smoother, transitions)
        state_extras = transitions.extras
        state_extras['state_extras']['derivative'] = dx
        transitions = transitions._replace(extras=state_extras)
        collected_data_buffer_state = agent_state.optimizer_state.true_buffer_state
        collected_data_buffer_state = self.collected_data_buffer.insert(buffer_state=collected_data_buffer_state,
                                                                        samples=transitions)
        optimizer_state = optimizer_state.replace(true_buffer_state=collected_data_buffer_state)
        env_steps = agent_state.env_steps + self.num_envs * self.episode_length
        return ModelBasedAgentState(optimizer_state=optimizer_state, env_steps=env_steps, key=key_agent), transitions

    def _get_dx(self,
               key: jr.PRNGKey,
               transitions: Transition,
               ) -> Tuple[jnp.ndarray, Transition]:
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
            fig, _ = self.smoother_model.plot_fit(inputs, pred_x.mean, outputs, true_dx, ders.mean)
            wandb.log({'smoother/fit': wandb.Image(fig)})

        # Use the smoothed trajectory in the transitions
        if self.state_data_source == 'smoother':
            transition = longest_trajectory._replace(observation=pred_x.mean)
        else:
            transition = longest_trajectory
        return ders.mean, transition

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