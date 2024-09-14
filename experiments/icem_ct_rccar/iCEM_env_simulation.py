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

from mbrl.envs.rccar import RCCarSimEnv

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
    state = State(pipeline_state=init_state,
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

    best_reward = -jnp.inf
    for i in range(10):
        opt_params = iCemParams(
            num_particles=1,
            num_samples=1000,
            num_elites=60,
            num_steps=30,
            exponent=1.0,)
        
        optimizer = iCemTO(horizon=55,
                        action_dim=2,
                        key=jr.PRNGKey(0),
                        opt_params=opt_params,
                        rollout_function=rollout_actions,)
        
        env = RCCarSimEnv(use_obs_noise=False,
                        use_tire_model=True)
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

        observations = jnp.array(observations)
        actions = jnp.array(actions)
        rewards = jnp.array(rewards)

        if rewards.sum() > best_reward:
            best_reward = rewards.sum()
            best_actions = actions
            best_observations = observations
            best_rewards = rewards
            print(f'!!! New best reward: {best_reward:.3f} for horizon {30 + i*5} !!!')
        else:
            print(f'Current reward: {rewards.sum():.3f} for horizon {30 + i*5}')


    actions = best_actions
    observations = best_observations
    rewards = best_rewards
    # Plot the results
    import matplotlib.pyplot as plt
    # TUM colors for the plots
    #          blue       orange      green     black      gray
    colors = ['#0065bd', '#e37222', '#a2ad00', '#000000', '#999999']
    fig, axs = plt.subplots(3, 1, figsize=(7, 7))
    time = jnp.arange(0, horizon*dt, dt)
    # Plot the observations
    axs[0].plot(time, observations[:,0], label=r'$pos_x$', color=colors[0])
    axs[0].plot(time, observations[:,1], label=r'$pos_y$', color=colors[1])
    axs[0].plot(time, observations[:,2], label=r'$\theta$', color=colors[2])
    axs[0].plot(time, jnp.sqrt(jnp.square(observations[:,3]) + jnp.square(observations[:,4])), label=r'$v$', color=colors[4])
    axs[0].legend()
    axs[0].set_title('Observations')
    axs[0].grid()

    axs[1].plot(time, actions[:,0], label='Steering', color=colors[0])
    axs[1].plot(time, actions[:,1], label='Throttle', color=colors[1])
    axs[1].set_title('Actions')
    axs[1].grid()
    axs[2].plot(time, rewards)
    axs[2].set_title('Rewards ' + str(jnp.array(rewards).sum()))
    axs[2].set_xlabel('Time')
    axs[2].grid()
    plt.savefig('icem_optimizer.png')

    # Plot an x-y trajectory with orientation
    
    plt.figure(figsize=(10, 10))
    # Plot the positions with the color representing the current velocity
    plt.scatter(observations[:, 0], observations[:, 1], c=jnp.sqrt(jnp.square(observations[:, 3]) + jnp.square(observations[:, 4])), cmap='viridis')
    # Add a labeled start and end point
    plt.scatter(observations[0, 0], observations[0, 1], c='r')
    plt.text(observations[0, 0], observations[0, 1], 'Start', fontsize=16, color='k')
    plt.scatter(env._goal[0], env._goal[1], c='g')
    plt.text(env._goal[0], env._goal[1], 'End', fontsize=16, color='k')
    plt.quiver(observations[::2, 0], observations[::2, 1], jnp.cos(observations[::2, 2]), jnp.sin(observations[::2, 2]),
               color=colors[4], alpha=0.5)
    #Show the color bar
    #plt.colorbar()
    plt.grid()
    plt.title('Trajectory')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('icem_trajectory.png')

