import jax
from brax.envs.base import PipelineEnv, State, Env
import chex
from flax import struct
import jax.numpy as jnp
from jaxtyping import Float, Array
from functools import partial

from mbrl.utils.tolerance_reward import ToleranceReward


@chex.dataclass
class PendulumDynamicsParams:
    max_speed: chex.Array = struct.field(default_factory=lambda: jnp.array(8.0))
    max_torque: chex.Array = struct.field(default_factory=lambda: jnp.array(2.0))
    dt: chex.Array = struct.field(default_factory=lambda: jnp.array(0.05))
    g: chex.Array = struct.field(default_factory=lambda: jnp.array(9.81))
    m: chex.Array = struct.field(default_factory=lambda: jnp.array(1.0))
    l: chex.Array = struct.field(default_factory=lambda: jnp.array(1.0))


@chex.dataclass
class PendulumRewardParams:
    control_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(0.02))
    angle_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(1.0))
    target_angle: chex.Array = struct.field(default_factory=lambda: jnp.array(0.0))


class PendulumEnv(Env):
    def __init__(self,
                 reward_source: str = 'gym',
                 margin_factor: float = 10.0,
                 ):
        self.dynamics_params = PendulumDynamicsParams()
        self.reward_params = PendulumRewardParams()
        bound = 0.01
        value_at_margin = 0.8
        self.reward_source = reward_source  # 'dm-control' or 'gym'
        self.tolerance_reward = ToleranceReward(bounds=(0.0, bound),
                                                margin=margin_factor * bound,
                                                value_at_margin=value_at_margin,
                                                sigmoid='long_tail')

    def reset(self,
              rng: jax.Array) -> State:
        return State(pipeline_state=None,
                     obs=jnp.array([-1.0, 0.0, 0.0]),
                     reward=jnp.array(0.0),
                     done=jnp.array(0.0), )

    def reward(self,
               x: Float[Array, 'observation_dim'],
               u: Float[Array, 'action_dim']) -> Float[Array, 'None']:
        theta, omega = jnp.arctan2(x[1], x[0]), x[-1]
        target_angle = self.reward_params.target_angle
        diff_th = theta - target_angle
        diff_th = ((diff_th + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        reward = -(self.reward_params.angle_cost * diff_th ** 2 +
                   0.1 * omega ** 2) - self.reward_params.control_cost * u ** 2
        reward = reward.squeeze()
        return reward

    def dm_reward(self,
                  x: Float[Array, 'observation_dim'],
                  u: Float[Array, 'action_dim']) -> Float[Array, 'None']:
        theta, omega = jnp.arctan2(x[1], x[0]), x[-1]
        target_angle = self.reward_params.target_angle
        diff_th = theta - target_angle
        diff_th = ((diff_th + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        reward = self.tolerance_reward(jnp.sqrt(self.reward_params.angle_cost * diff_th ** 2 +
                                       0.1 * omega ** 2)) - self.reward_params.control_cost * u ** 2
        reward = reward.squeeze()
        return reward

    @partial(jax.jit, static_argnums=0)
    def step(self,
             state: State,
             action: jax.Array) -> State:
        x = state.obs
        chex.assert_shape(x, (self.observation_size,))
        chex.assert_shape(action, (self.action_size,))
        th = jnp.arctan2(x[1], x[0])
        thdot = x[-1]
        dt = self.dynamics_params.dt
        x_compressed = jnp.array([th, thdot])
        dx = self.ode(x_compressed, action)
        newth = th + dx[0] * dt
        newthdot = thdot + dx[-1] * dt
        newthdot = jnp.clip(newthdot, -self.dynamics_params.max_speed, self.dynamics_params.max_speed)
        next_obs = jnp.asarray([jnp.cos(newth), jnp.sin(newth), newthdot]).reshape(-1)
        if self.reward_source == 'gym':
            next_reward = self.reward(x, action)
        elif self.reward_source == 'dm-control':
            next_reward = self.dm_reward(x, action)
        else:
            raise NotImplementedError(f'Unknown reward source {self.reward_source}')

        next_state = State(pipeline_state=state.pipeline_state,
                           obs=next_obs,
                           reward=next_reward,
                           done=state.done,
                           metrics=state.metrics,
                           info=state.info)
        return next_state

    def ode(self, x_compressed: chex.Array, u: chex.Array) -> chex.Array:
        chex.assert_shape(x_compressed, (self.observation_size - 1,))
        chex.assert_shape(u, (self.action_size,))
        thdot = x_compressed[-1]
        th = x_compressed[0]

        g = self.dynamics_params.g
        m = self.dynamics_params.m
        l = self.dynamics_params.l
        dt = self.dynamics_params.dt
        u = jnp.clip(u, -1, 1) * self.dynamics_params.max_torque
        newthddot = (3 * g / (2 * l) * jnp.sin(th) + 3.0 / (m * l ** 2) * u)
        newthdot = thdot + newthddot * dt
        newthdot = jnp.clip(newthdot, -self.dynamics_params.max_speed, self.dynamics_params.max_speed)
        return jnp.asarray([newthdot, newthddot])

    @property
    def dt(self):
        return self.dynamics_params.dt

    @property
    def observation_size(self) -> int:
        return 3

    @property
    def action_size(self) -> int:
        return 1

    def backend(self) -> str:
        return 'positional'
