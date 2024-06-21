import jax.numpy as jnp
import jax.random as jr
from typing import Tuple
from jax import jit
from functools import partial
from brax.training.types import Transition
from brax.training.replay_buffers import UniformSamplingQueue
import types

from bsm.utils.normalization import Data
from mbrl.model_based_agent.base_model_based_agent import ModelBasedAgentState
from differentiators.nn_smoother import smoother_net

###
# To-Do:
# - Check the implementation of wrappers in gym
# - Change the way the agent is initiated (so that we dont run separate prepare_data_buffers)

def prepare_data_buffers(self) -> UniformSamplingQueue:
    dummy_sample = Transition(observation=jnp.zeros(shape=(self.env.observation_size,)),
                              action=jnp.zeros(shape=(self.env.action_size,)),
                              reward=jnp.array(0.0),
                              discount=jnp.array(0.99),
                              next_observation=jnp.zeros(shape=(self.env.observation_size,)),
                              extras={'state_extras': {'t': jnp.array(0.0),
                                                       'derivative': jnp.zeros(shape=(self.env.observation_size,))}},
                              )
    collected_data_buffer = UniformSamplingQueue(
        max_replay_size=self.max_collected_data_in_buffer,
        dummy_data_sample=dummy_sample,
        sample_batch_size=1)
    return collected_data_buffer

def get_dx(self,
           key: jr.PRNGKey,
           transitions: Transition,
           ) -> Tuple[jnp.ndarray, Transition]:
    # Split transition into trajectories without resets (based on the time)
    # Get the time for each trajectory
    timestamps = transitions.extras['state_extras']['t']
    # Get the indices of the transitions for each trajectory
    indices = []
    for dt, i in enumerate(jnp.diff(timestamps)):
        if dt < 0:
            indices.append(i)
    # Split based on the indices
    trajectories = split_transitions(transitions, indices)

    # Fit Smoother for each trajectory (or only on longest one)
    # Choose the longest one for now:
    longest_trajectory = max(trajectories, key=lambda x: len(x.observation))
    # Get the time as the input for the smoother
    # We should use the time as provided by the environment, since it could be non-uniformly sampled
    inputs = longest_trajectory.extras['state_extras']['t'].reshape(-1, 1)
    outputs = longest_trajectory.observation
    data = Data(inputs, outputs)
    # Get dx from Smoother
    model_states = self.smoother_model.train_new_smoother(key, data)
    pred_x = self.smoother_model.predict_batch(inputs, model_states)
    ders = self.smoother_model.derivative_batch(inputs, model_states)

    # Use the smoothed trajectory in the transitions
    if self.state_data_source == 'smoother':
        transition = longest_trajectory._replace(observation=pred_x.mean)
    else:
        transition = longest_trajectory
    
    return ders.mean, transition

def split_transitions(transition: Transition,
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
    time = transition.extras['state_extras']['t']._split(indices)

    for i in range(len(indices)):
        trajectories.append(Transition(
            observation=observations[i],
            action=actions[i],
            reward=rewards[i],
            discount=discounts[i],
            next_observation=next_observations[i],
            extras={'state_extras': {'t': time[i]}}
        ))
    # Add the last part
    trajectories.append(Transition(
        observation=observations[-1],
        action=actions[-1],
        reward=rewards[-1],
        discount=discounts[-1],
        next_observation=next_observations[-1],
        extras={'state_extras': {'t': time[-1]}}
    ))

    return trajectories
    
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
    dx, transitions = self.get_dx(key_smoother, transitions)
    state_extras = transitions.extras
    state_extras['state_extras']['derivative'] = dx
    transitions = transitions._replace(extras=state_extras)
    collected_data_buffer_state = agent_state.optimizer_state.true_buffer_state
    collected_data_buffer_state = self.collected_data_buffer.insert(buffer_state=collected_data_buffer_state,
                                                                    samples=transitions)
    optimizer_state = optimizer_state.replace(true_buffer_state=collected_data_buffer_state)
    env_steps = agent_state.env_steps + self.num_envs * self.episode_length
    return ModelBasedAgentState(optimizer_state=optimizer_state, env_steps=env_steps, key=key_agent), transitions

def Smoother_Wrapper(agent_class,
                     smoother: smoother_net,
                     state_data_source: str = 'smoother',
                     *args,
                     **kwargs):
    # Create an instance of the agent
    agent = agent_class(*args, **kwargs)
    agent.state_data_source = state_data_source

    # Add the prepare_data_buffers function to the agent
    agent.prepare_data_buffers = types.MethodType(prepare_data_buffers, agent)
    agent.collected_data_buffer = agent.prepare_data_buffers()

    # Initialise the smoother
    agent.smoother_model = smoother
    # Add the get_dx function to the agent
    agent.get_dx = types.MethodType(get_dx, agent)
    # Override the simulate_on_true_env function
    agent.simulate_on_true_env = types.MethodType(simulate_on_true_env, agent)

    return agent