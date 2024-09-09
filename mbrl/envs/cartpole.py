import jax
import jax.numpy as jnp
from brax.envs.base import PipelineEnv, State, Env
import chex
from flax import struct
from functools import partial
import copy

from mbrl.utils.tolerance_reward import ToleranceReward

@chex.dataclass
class CartpoleDynamicsParams:
    max_lin_speed: chex.Array = struct.field(default_factory=lambda: jnp.array(10.0))
    max_ang_speed: chex.Array = struct.field(default_factory=lambda: jnp.array(10.0))
    max_force: chex.Array = struct.field(default_factory=lambda: jnp.array(10.0))
    dt: chex.Array = struct.field(default_factory=lambda: jnp.array(0.05))
    g: chex.Array = struct.field(default_factory=lambda: jnp.array(9.81))
    m_cart: chex.Array = struct.field(default_factory=lambda: jnp.array(1.0))
    m_pole: chex.Array = struct.field(default_factory=lambda: jnp.array(0.1))
    l: chex.Array = struct.field(default_factory=lambda: jnp.array(0.5))

@chex.dataclass
class CartpoleRewardParams:
    control_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(0.02))
    angle_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(1.0))
    target_angle: chex.Array = struct.field(default_factory=lambda: jnp.array(0.0))


class ContinuousCartpoleEnv(Env):
    '''Continuous Cartpole Evironment
    This environment is the cartpole environment based on the work done by Barto,
    Sutton, and Anderson in
    ["Neuronlike adaptive elements that can solve difficult learning control
    problems"](https://ieeexplore.ieee.org/document/6313077).
    
    ### Action Space
    The agent take a 1-element vector for actions. The action space is a
    continuous `(action)` in `[-1, 1]`, where `action` represents the numerical
    force applied to the cart (with magnitude representing the amount of force and
    sign representing the direction)

    ### Observation Space
    The state space consists of positional values of different body parts of the
    pendulum system, followed by the velocities of those individual parts (their
    derivatives) with all the positions ordered before all the velocities.

    The observation is a `ndarray` with shape `(5,)` where the elements correspond
    to the following:

    | Num | Observation                                   
    |-----|-----------------------------------------------
    | 0   | position of the cart along the linear surface
    | 1   | Cosine of the vertical angle of the pole on the cart
    | 2   | Sine of the vertical angle of the pole on the cart
    | 3   | linear velocity of the cart
    | 4   | angular velocity of the pole on the cart
    
    '''
    
    def __init__(self, reward_source: str = 'gym',
                 noise_level: chex.Array | None = None,
                 init_noise_key: chex.PRNGKey | None = None):
        self.dynamics_params = CartpoleDynamicsParams()
        self.reward_params = CartpoleRewardParams()
        bound = 0.1
        value_at_margin = 0.1
        margin_factor = 10.0
        self.reward_source = reward_source
        if noise_level is not None:
            chex.assert_shape(noise_level, (4,))
        self.noise_level = noise_level
        self.init_noise_key = init_noise_key
        self.state_labels = [r'$x$', r'$cos(\theta)$', r'$sin(\theta)$', r'$\dot{x}$', r'$\dot{\theta}$']
        self.state_derivative_labels = [r'$\dot{x}$', r'$-sin(\theta) \dot{\theta}$', r'$cos(\theta) \dot{\theta}$',
                                        r'$\ddot{x}$', r'$\ddot{\theta}$']
        self.tolerance_reward = ToleranceReward(bounds=(0.0, bound),
                                                margin=margin_factor * bound,
                                                value_at_margin=value_at_margin,
                                                sigmoid='long_tail')
        
    def reset(self,
              rng: jax.Array | None = None) -> State:
        first_info: dict = {'derivative': jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
                            't': jnp.array(0.0),
                            'dt': jnp.array(self.dynamics_params.dt),
                            'noise_key': self.init_noise_key}
        return State(pipeline_state=jnp.array([0.0, -1.0, 0.0, 0.0, 0.0]),
                     obs=jnp.array([0.0, -1.0, 0.0, 0.0, 0.0]),
                     reward=jnp.array(0.0),
                     done=jnp.array(0.0),
                     info=first_info)
    
    def reward(self,
               x: chex.Array,
               u: chex.Array) -> chex.Array:
        theta, omega = jnp.arctan2(x[2], x[1]), x[-1]
        target_angle = self.reward_params.target_angle
        diff_th = theta - target_angle
        diff_th = ((diff_th + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        reward = -(self.reward_params.angle_cost * diff_th ** 2 +
                   0.1 * omega ** 2) - self.reward_params.control_cost * u ** 2
        return reward.squeeze()
    
    def dm_reward(self,
                    x: chex.Array,
                    u: chex.Array) -> chex.Array:
        theta, omega = jnp.arctan2(x[2], x[1]), x[-1]
        target_angle = self.reward_params.target_angle
        diff_th = theta - target_angle
        diff_th = ((diff_th + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        reward = self.tolerance_reward(jnp.sqrt(self.reward_params.angle_cost * diff_th ** 2 +
                                       0.1 * omega ** 2)) - self.reward_params.control_cost * u ** 2
        return reward.squeeze()
        
    @partial(jax.jit, static_argnums=0)
    def step(self,
             state: State,
             action: jax.Array) -> State:
        # State: (x, cos(theta), sin(theta), x_dot, theta_dot)
        x = state.pipeline_state
        chex.assert_shape(x, (5,))
        chex.assert_shape(action, (1,))
        th = jnp.arctan2(x[2], x[1])
        dt = self.dynamics_params.dt
        dx_compressed = self.ode(x, action).reshape(-1,)
        dx = jnp.array([x[3],
                        -jnp.sin(th)*x[4],
                        jnp.cos(th)*x[4],
                        dx_compressed[0],
                        dx_compressed[1]])
        new_x = jnp.array([x[0] + dx[0]*dt,
                           jnp.cos(th + x[4]*dt),
                           jnp.sin(th + x[4]*dt),
                           x[3] + dx[3]*dt,
                           x[4] + dx[4]*dt])

        if self.reward_source == 'gym':
            next_reward = self.reward(x, action)
        elif self.reward_source == 'dm-control':
            next_reward = self.dm_reward(x, action)
        else:
            raise NotImplementedError(f'Unknown reward source {self.reward_source}')
        
        next_info = copy.deepcopy(state.info)
        next_info['derivative'] = dx
        next_info['t'] += dt
        next_info['dt'] = dt

        if self.noise_level is not None:
            next_info['noise_key'], subkey = jax.random.split(next_info['noise_key'])
            x_compressed = jnp.array([new_x[0], jnp.arctan2(new_x[2], new_x[1]), new_x[3], new_x[4]])
            noisy_x_compressed = x_compressed + self.noise_level * jax.random.normal(subkey, (4,))
            noisy_obs = jnp.array([noisy_x_compressed[0],
                                   jnp.cos(noisy_x_compressed[1]),
                                   jnp.sin(noisy_x_compressed[1]),
                                   noisy_x_compressed[2],
                                   noisy_x_compressed[3]])
        
        return State(pipeline_state=new_x,
                     obs=noisy_obs if self.noise_level is not None else new_x,
                     reward=next_reward,
                     done=state.done,
                     metrics=state.metrics,
                     info=next_info)
    
    def ode(self,
            x: chex.Array,
            u: chex.Array) -> chex.Array:
        chex.assert_shape(x, (self.observation_size,))
        chex.assert_shape(u, (self.action_size,))

        g = self.dynamics_params.g
        m_cart = self.dynamics_params.m_cart
        m_pole = self.dynamics_params.m_pole
        l = self.dynamics_params.l
        dt = self.dynamics_params.dt
        force = jnp.clip(u, -1, 1) * self.dynamics_params.max_force

        newthddot_num = g * x[2] + x[1] * ((-force - m_pole * l * jnp.power(x[4],2)*x[2])/(m_cart + m_pole))
        newthddot_den = l * (4/3 - m_pole * jnp.power(x[1],2)/(m_cart + m_pole))
        newthddot = newthddot_num / newthddot_den

        newxddot_num = force + m_pole * l * (jnp.power(x[4],2)*x[2] - newthddot*x[1])
        newxddot_den = m_cart + m_pole
        newxddot = newxddot_num / newxddot_den

        return jnp.array([newxddot, newthddot])

    @property
    def dt(self):
        return self.dynamics_params.dt

    @property
    def observation_size(self) -> int:
        return 5

    @property
    def action_size(self) -> int:
        return 1

    def backend(self) -> str:
        return 'positional'
    
if __name__ == '__main__':
    env = ContinuousCartpoleEnv()
    state = env.reset()
    obs_log = []
    for i in range(100):
        action = jnp.sin(i*6*jnp.pi/100).reshape(-1,)
        state = env.step(state, action)
        obs_log.append(state.obs)
    import matplotlib.pyplot as plt
    obs_log = jnp.array(obs_log)
    plt.plot(obs_log[:, 0], label='Cart Position')
    plt.plot(obs_log[:, 1], label='Cosine of Pole Angle')
    plt.plot(obs_log[:, 2], label='Sine of Pole Angle')
    plt.plot(obs_log[:, 3], label='Cart Velocity')
    plt.plot(obs_log[:, 4], label='Pole Angular Velocity')
    plt.legend()
    plt.grid()
    plt.savefig('cartpole.png')
    print("1")