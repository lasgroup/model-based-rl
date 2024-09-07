# Solo iCEM run on a continuous pendulum environment

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import wandb

from brax.envs.base import State
from brax.training.types import Transition

from mbpo.systems import System, SystemParams
from mbpo.optimizers.trajectory_optimizers.icem_optimizer import iCemTO, iCemParams

from mbrl.envs.bicyclecar import BicycleEnv

# Create a system class to wrap the environment
class EnvSystem:
    def __init__(self, env):
        self.env = env
        self.x_dim = env.observation_size
        self.u_dim = env.action_size

    def init_params(self, key):
        return None
    
    def step(self, state, action):
        return self.env.step(state, action)
        
    def reset(self):
        return env.reset()

# Define a custom rollout function for the iCEM optimizer
def rollout_actions(
        system: System,
        system_params: SystemParams,
        init_state: chex.Array,
        actions: chex.Array,
        horizon: int,
) -> Transition:
    
    assert actions.shape[0] == horizon
    state = system.reset()
    state = State(pipeline_state=state.pipeline_state,
                  obs=init_state,
                  reward=state.reward,
                  done=state.done,
                  info=state.info)

    def step(carry, acs):
        next_state = system.step(carry, acs)
        outs = [next_state.obs, next_state.reward]
        return next_state, outs
    
    _, (obs, rewards) = jax.lax.scan(step, state, actions)
    state = jnp.zeros_like(obs)
    state = state.at[0, ...].set(init_state)
    state = state.at[1:, ...].set(obs[:-1, ...])
    return Transition(
        observation=state,
        action=actions,
        reward=rewards,
        next_observation=obs,
        discount=jnp.ones_like(rewards),
    )


if __name__ == '__main__':

    opt_params = iCemParams(
        num_particles=5,
        num_samples=1000,
        num_elites=100,
        num_steps=10,
        exponent=5.0,)
    
    optimizer = iCemTO(horizon=50,
                       action_dim=2,
                       key=jr.PRNGKey(0),
                       opt_params=opt_params,
                       rollout_function=rollout_actions,)
    
    env = BicycleEnv(use_obs_noise=False)
    system = EnvSystem(env)

    optimizer.set_system(system)

    key = jr.PRNGKey(0)
    opt_state = optimizer.init(key)

    state = env.reset()
    
    # Create variable for logging
    actions = []
    observations = []
    rewards = []

    horizon = 200
    dt = env.dt

    for k01 in range(horizon):
        action, opt_state = optimizer.act(state.obs, opt_state)
        state = system.step(state, action)
        actions.append(action)
        observations.append(state.obs)
        rewards.append(state.reward)

    # Plot the results
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    time = jnp.arange(0, horizon*dt, dt)
    axs[0].plot(time, observations)
    axs[0].set_title('Observations')
    axs[0].legend(env.state_labels)
    axs[0].grid()
    axs[1].plot(time, actions)
    axs[1].set_title('Actions')
    axs[1].legend(['Steering', 'Throttle'])
    axs[1].grid()
    axs[2].plot(time, rewards)
    axs[2].set_title('Rewards ' + str(jnp.array(rewards).sum()))
    axs[2].set_xlabel('Time')
    axs[2].grid()
    plt.savefig('icem_optimizer.png')
