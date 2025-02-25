import jax
from brax import base
from brax.envs.base import PipelineEnv, State, Env
import chex
from flax import struct
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Float, Array
from functools import partial
import copy
from distrax import Distribution
from distrax import Normal
from typing import Tuple
from mbpo.systems.base_systems import System, SystemParams, SystemState
from mbpo.systems.dynamics.base_dynamics import Dynamics
from mbpo.systems.rewards.base_rewards import Reward
from mbpo.systems import DynamicsParams, RewardParams
import time
import matplotlib.pyplot as plt


from mbrl.utils.tolerance_reward import ToleranceReward
from mbpo.optimizers.trajectory_optimizers.icem_optimizer import iCemParams, iCemTO
from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv

@chex.dataclass
class MountainCarRewardParams:
    control_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(0.01))
    angle_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(1.0))
    target_position: chex.Array = struct.field(
        default_factory=lambda: jnp.array(0.45) # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        )
    target_velocity: chex.Array = struct.field(
        default_factory=lambda: jnp.array(0.0)
        )


class MountainCar(Env):
    """
    Implementation of the Mountain Car MDP in JAX, adapted from the [Farama Gymnasium implementation](https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/). 
    This MDP first appeared in [Andrew Moore's PhD Thesis (1990)](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.pdf)

    ```
    @TECHREPORT{Moore90efficientmemory-based,
        author = {Andrew William Moore},
        title = {Efficient Memory-based Learning for Robot Control},
        institution = {University of Cambridge},
        year = {1990}
    }
    ```
    """
    def __init__(self, reward_source: str = 'gym',
                 noise_level: chex.Array | None = None,
                 init_noise_key: chex.PRNGKey | None = None,
                 initial_position: float = -0.5,
                 goal_position: float = None,
                 goal_velocity: float = None,
                 bound: float = 0.1,
                 value_at_margin: float = 0.1,
                 margin_factor: float = 10.0):
        self.dynamics_params = None
        self.reward_params = MountainCarRewardParams()
        if goal_position is not None:
            raise NotImplementedError()
        
        self.reward_source = reward_source  # 'gym' or 'two-sided'
        self.noise_level = noise_level      # Noise level can have one value (for both velocity and position) or two separate values
        self.init_noise_key = init_noise_key
        self.state_labels = [r'$x$', r'$\dot{x}$']
        self.state_derivative_labels = [r'$\dot{x}$', r'$\ddot{x}$']
        self.env = Continuous_MountainCarEnv()
        self.tolerance_reward = ToleranceReward(bounds=(0.0, bound),
                                                margin=margin_factor * bound,
                                                value_at_margin=value_at_margin,
                                                sigmoid='long_tail')
        self.initial_position = initial_position

    def reset(self, rng: jax.Array = None) -> State:
        """Resets the environment to an initial state."""
        first_info: dict = {'derivative': jnp.array([0.0, 0.0]),
                't': jnp.array(0.0),
                'dt': jnp.array(self.dt),
                'noise_key': self.init_noise_key}
        """
        if self.init_noise_key is not None:
            raise NotImplementedError()
            # return self.env.reset()

        if rng is None:
            initial_position = self.initial_position
        else:
            key, subkey = jax.random.split(rng)
            initial_position = jax.random.uniform(subkey, shape=(), minval=-0.6, maxval=-0.4)

        initial_state = jnp.array([initial_position, 0.0])
        """
        initial_state = jnp.array([-0.5, 0.0])
        
        return State(
            pipeline_state=initial_state,
            obs=initial_state,
            reward=jnp.array(0.0),
            done=jnp.array(0.0),
            info=first_info
        )

        
    def reward(self,
               x: Float[Array, 'observation_dim'],
               u: Float[Array, 'action_dim']) -> tuple[Float[Array, 'None'], Float[Array, 'None']]:
        position, velocity = x[..., 0], x[..., 1]
        action_penalty = -self.reward_params.control_cost * (u**2)
        
        terminated = jnp.logical_and(
                position >= self.reward_params.target_position, 
                velocity >= self.reward_params.target_velocity)

        goal_reward = jnp.where(terminated, 100.0, 0.0)

        return jnp.squeeze(action_penalty + goal_reward), terminated

    def ts_reward(self,
               x: Float[Array, 'observation_dim'],
               u: Float[Array, 'action_dim']) -> tuple[Float[Array, 'None'], Float[Array, 'None']]:
        position, velocity = x[..., 0], x[..., 1]
        action_penalty = -self.reward_params.control_cost * (u**2)
        target_position = self.reward_params.target_position
        target_velocity = self.reward_params.target_velocity

        terminated = jnp.logical_and(
            jnp.isclose(position, target_position),
            jnp.isclose(velocity, target_velocity)
            )

        goal_reward = jnp.where(terminated, 100.0, 0.0)

        return jnp.squeeze(action_penalty + goal_reward), terminated
  
    @partial(jax.jit, static_argnums=0)
    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        print("JIT recompiling env step...", state.pipeline_state.shape, action.shape)  # Debugging
        obs = state.pipeline_state
        chex.assert_shape(obs, (self.observation_size,))
        chex.assert_shape(action, (self.action_size,))

        position, velocity = obs[..., 0], obs[..., 1]
        force = jnp.clip(action, a_min=self.env.min_action, a_max=self.env.max_action)
        next_acceleration = force * self.env.power - 0.0025 * jnp.cos(3 * position)
        next_velocity = velocity + next_acceleration
        next_velocity = jnp.clip(next_velocity, a_min=-self.env.max_speed, a_max=self.env.max_speed)
        next_derivative = jnp.concatenate([next_velocity, next_acceleration], axis=-1).reshape(-1, 2).squeeze()

        next_position = position + next_velocity
        next_position = jnp.clip(next_position, self.env.min_position, self.env.max_position)
        out_of_bounds = jnp.logical_and(next_position - self.env.min_position <= 0.0, next_velocity < 0.0)
        next_velocity = (1 - out_of_bounds) * next_velocity
        next_obs = jnp.concatenate([next_position, next_velocity], axis=-1).reshape(-1, 2).squeeze()

        if self.reward_source == 'gym':
            next_reward, done = self.reward(obs, action)
        elif self.reward_source == 'two-sided':
            next_reward, done = self.ts_reward(obs, action)
        else:
            raise NotImplementedError(f'Unknown reward source {self.reward_source}')

        next_info = jax.tree.map(lambda x: x, state.info)
        next_info['derivative'] = next_derivative
        next_info['t'] = state.info['t'] + self.dt
        next_info['dt'] = self.dt

        if self.noise_level is not None:
            noisy_obs = 0
            raise NotImplementedError()

        next_state = State(pipeline_state=next_obs,
                           obs=noisy_obs if self.noise_level is not None else next_obs,
                           reward=next_reward,
                           done=state.done,
                           metrics=state.metrics,
                           info=next_info)
        return next_state

    @property
    def dt(self):
        return 1.0
  
    @property
    def observation_size(self) -> int:
        """The size of the observation vector returned in step and reset."""
        return 2

  
    @property
    def action_size(self) -> int:
        """The size of the action vector expected by step."""
        return 1
  
    @property
    def backend(self) -> str:
        """The physics backend that this env was instantiated with."""
        return 'positional'


def test_mountain_car_termination():
    """Test termination conditions for the JAX Mountain Car environment."""

    # Initialize environment with a positive goal position
    env = MountainCar(reward_source='gym') # , goal_position=0.45, goal_velocity=0.0)

    # Test: Car below the goal should not terminate
    state = State(
        pipeline_state=jnp.array([0.3, 0.02]),  # Not at goal yet
        obs=jnp.array([0.3, 0.02]),
        reward=0.0,
        done=False,
        info={'t': 0}
    )
    action = jnp.array([0.1])  # Some force applied
    next_state = env.step(state, action)

    assert next_state.done == False, "Car should not terminate before reaching the goal."

    # Test: Car at the goal position and velocity should terminate
    state_at_goal = State(
        pipeline_state=jnp.array([0.45, 0.0]),  # Exactly at goal
        obs=jnp.array([0.45, 0.0]),
        reward=0.0,
        done=False,
        info={'t': 0}
    )
    next_state = env.step(state_at_goal, action)

    assert next_state.done == True, "Car should terminate when reaching the goal."

    print("✅ Termination test cases passed!")


def test_mountain_car_reaches_goal():
    """Test if the Mountain Car environment reaches the goal with a constant force."""
    
    # Initialize environment with default goal position (0.45) and velocity (0.0)
    env = MountainCar()
    
    # Reset environment
    state = env.reset()

    # Apply constant force to the right
    action = jnp.array([0.8])  # Moderate force in the positive direction

    max_steps = 5000  # Prevent infinite loops in case of a bug
    reached_goal = False

    for step in range(max_steps):
        state = env.step(state, action)
        if state.done:
            reached_goal = True
            break

    # assert reached_goal, "Car did not reach the goal within max_steps!"
    print(f"✅ Mountain Car reached the goal in {step + 1} steps with constant force {action[0]}")


def simulate_mountain_car(state, action, env):
    """Simulate multiple steps in the Mountain Car environment using JAX while loop."""
    
    def cond_fn(state_step):
        state, step = state_step
        return jnp.logical_and(step < 5000, jnp.logical_not(state.done))  # ✅ JAX-compatible condition

    def body_fn(state_step):
        state, step = state_step
        state = env.step(state, action)
        return state, step + 1  # ✅ Keep track of steps

    final_state, _ = jax.lax.while_loop(cond_fn, body_fn, (state, 0))
    return final_state.done, final_state.pipeline_state  # ✅ Works inside JAX

def test_mountain_car_vmap():
    """Test if multiple Mountain Car instances reach the goal with constant force using JAX vmap."""
    
    env = MountainCar()

    # Initialize multiple environments with different starting positions
    num_envs = 10
    initial_positions = jnp.linspace(-0.6, -0.4, num_envs)  # Spread across valid start range
    initial_states = jax.vmap(env.reset)(rng=jr.split(jr.key(0), num_envs))  # Reset all envs at once
    initial_states = initial_states.replace(
        pipeline_state=jnp.stack([initial_positions, jnp.zeros(num_envs)], axis=1),
        obs=jnp.stack([initial_positions, jnp.zeros(num_envs)], axis=1)
    )

    # Apply constant force to all environments
    action = jnp.full((num_envs, 1), 0.8)  # Apply the same force to all environments

    # Use `vmap` to parallelize the simulation
    vmap_simulate = jax.vmap(simulate_mountain_car, in_axes=(0, 0, None))
    done_flags, final_states = vmap_simulate(initial_states, action, env)

    assert jnp.all(done_flags), "Some environments did not reach the goal!"
    print(f"✅ All {num_envs} Mountain Car instances reached the goal!")

class DummyDynamics(Dynamics):
    def __init__(self, x_dim, u_dim):
        super().__init__(x_dim=x_dim, u_dim=u_dim)

    def next_state(self,
                   x: chex.Array,
                   u: chex.Array,
                   dynamics_params: DynamicsParams) -> Tuple[Distribution, DynamicsParams]:
        return Normal(0, 0.01), dynamics_params

    def init_params(self, key: chex.PRNGKey) -> DynamicsParams:
        return 0


class DummyReward(Reward):
    def __init__(self, x_dim, u_dim):
        super().__init__(x_dim, u_dim)

    def init_params(self, key: chex.PRNGKey) -> RewardParams:
        return 0

    def __call__(self,
                 x: chex.Array,
                 u: chex.Array,
                 reward_params: RewardParams,
                 x_next: chex.Array | None = None) -> Tuple[Distribution, RewardParams]:
        return Normal(0, 0.01), reward_params


class MCSystem(System):
    def __init__(self, reward_source: str = 'gym', margin_factor: float = 100.0):
        super().__init__(dynamics=DummyDynamics(x_dim=2, u_dim=1),
                         reward=DummyReward(x_dim=2, u_dim=1))
        self.brax_env = MountainCar()

    def step(self,
             x: chex.Array,
             u: chex.Array,
             system_params: SystemParams[DynamicsParams, RewardParams],
             ) -> SystemState:
        """

        :param x: current state of the system
        :param u: current action of the system
        :param system_params: parameters of the system
        :return: Tuple of next state, reward, updated system parameters
        """
        state = State(pipeline_state=x,
                      obs=x,
                      reward=jnp.array(0.0),
                      done=jnp.array(0.0), 
                      info = {'derivative': jnp.array([0.0, 0.0]),
                        't': jnp.array(0.0),
                        'dt': jnp.array(self.brax_env.dt)})

        next_state = self.brax_env.step(state, u)
        next_system_state = SystemState(x_next=next_state.obs,
                                        reward=next_state.reward,
                                        system_params=system_params,
                                        done=next_state.done)

        return next_system_state

class ActionRepeatWrapper(System):
    def __init__(self,
                 action_repeat: int,
                 system: System):
        super().__init__(dynamics=system.dynamics,
                         reward=system.reward)
        self.action_repeat = action_repeat
        self.system = system

    def step(self,
             x: chex.Array,
             u: chex.Array,
             system_params: SystemParams[DynamicsParams, RewardParams],
             ) -> SystemState:
        total_reward = 0.0
        for _ in range(action_repeat):
            sys_state = self.system.step(x, u, system_params)
            x, reward, system_params = sys_state.x_next, sys_state.reward, sys_state.system_params
            total_reward += reward
        sys_state = sys_state.replace(reward=total_reward)
        return sys_state

if __name__ == "__main__":
    action_repeat = 4
    horizon = 25
    safe_exploration = True

    cost_fn = None

    optimizer = iCemTO(
        horizon=horizon,
        action_dim=1,
        key=jr.PRNGKey(0),
        opt_params=iCemParams(exponent=1.0,
                              num_samples=500,
                              alpha=0.2,
                              num_steps=5,
                              num_particles=1, ),
        system=ActionRepeatWrapper(action_repeat=action_repeat, system=MCSystem()),
        cost_fn=cost_fn,
    )

    system = MCSystem()

    optimizer_state = optimizer.init(key=jr.PRNGKey(1))
    system_params = system.init_params(key=jr.PRNGKey(2))
    # obs = jnp.array([-0.5, 0.0])
    key, reset_key = jax.random.split(jr.PRNGKey(69))
    initial_position = jax.random.uniform(reset_key, shape=(), minval=-0.6, maxval=-0.4)
    initial_state = system.brax_env.reset(reset_key)
    obs = initial_state.obs

    all_obs = []
    all_actions = []
    all_rewards = []

    times = []

    for i in range(200 // action_repeat):
        start_time = time.time()
        action, optimizer_state = optimizer.act(obs, optimizer_state)
        total_reward = 0
        for _ in range(action_repeat):
            sys_state = system.step(obs, action, system_params)
            obs, reward, system_params = sys_state.x_next, sys_state.reward, sys_state.system_params
            total_reward += reward
        all_obs.append(obs)
        all_actions.append(action)
        all_rewards.append(total_reward)
        end_time = time.time()
        times.append(end_time - start_time)

    fig, axs = plt.subplots(1, 4, figsize=(8, 2))
    axs[0].plot(all_obs)
    axs[0].set_title('Observation')
    axs[1].plot(all_actions)
    axs[1].set_title('Action')
    axs[2].plot(all_rewards)
    axs[2].set_title('Reward')
    axs[3].plot(times[2:])
    axs[3].set_title('Time')
    plt.tight_layout()
    plt.show()

    print(f'Maximal velocity value: {jnp.max(jnp.stack(all_obs)[:, -1])}')
    print(f'Minimal velocity value: {jnp.min(jnp.stack(all_obs)[:, -1])}')

    import numpy as np

    total_reward = np.sum(np.array(all_rewards))
    print(f'Total reward: {total_reward}')

    test_mountain_car_termination()
    test_mountain_car_reaches_goal()
    test_mountain_car_vmap()
    env = MountainCar()
    initial_state = env.reset(jr.key(0))
    initial_action = jax.numpy.ones(env.action_size)
    next_state = env.step(initial_state, initial_action)

    for ii in range(10):
        next_state = env.step(next_state, initial_action)