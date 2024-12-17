from typing import Generic, Tuple

import jax.random
from distrax import Distribution, Normal
import chex
import jax.numpy as jnp
import jax.random as jr

from mbpo.systems.base_systems import SystemParams, SystemState, System
from mbpo.systems.dynamics.base_dynamics import Dynamics
from mbpo.systems.dynamics.base_dynamics import DynamicsParams as DummyDynamicsParams
from bsm.statistical_model import StatisticalModel
from bsm.utils.type_aliases import ModelState, StatisticalModelState
from mbpo.systems.rewards.base_rewards import Reward, RewardParams


@chex.dataclass
class DynamicsParams(Generic[ModelState, DummyDynamicsParams]):
    key: chex.PRNGKey
    statistical_model_state: StatisticalModelState[ModelState]


class ContinuousPetsDynamics(Dynamics, Generic[ModelState]):
    def __init__(self,
                 x_dim: int,
                 u_dim: int,
                 statistical_model: StatisticalModel,
                 aleatoric_noise_in_prediction: bool = False,
                 predict_difference: bool = False,
                 dt: float = 0.05,
                 ):
        Dynamics.__init__(self, x_dim=x_dim, u_dim=u_dim)
        self.statistical_model = statistical_model
        self.aleatoric_noise_in_prediction = aleatoric_noise_in_prediction
        self.predict_difference = predict_difference
        assert (self.predict_difference == False), \
            f'self.predict_difference must be False for continuous dynamics, got: {self.predict_difference}.'
        self.dt = dt

    def vmap_input_axis(self, data_axis: int = 0) -> DynamicsParams:
        return DynamicsParams(
            key=data_axis,
            statistical_model_state=self.statistical_model.vmap_input_axis(data_axis=data_axis)
        )

    def vmap_output_axis(self, data_axis=0) -> tuple[int, DynamicsParams]:
        return (data_axis, DynamicsParams(key=data_axis,
                                          statistical_model_state=self.statistical_model.vmap_input_axis(
                                              data_axis=data_axis)))

    def next_state(self,
                   x: chex.Array,
                   u: chex.Array,
                   dynamics_params: DynamicsParams) -> Tuple[Distribution, DynamicsParams]:
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        # Create state-action pair
        z = jnp.concatenate([x, u])
        next_key, key_sample_x_next = jr.split(dynamics_params.key)

        model_output = self.statistical_model(input=z,
                                              statistical_model_state=dynamics_params.statistical_model_state)
        scale_std = model_output.epistemic_std
        dx_dist = Normal(loc=model_output.mean, scale=scale_std)
        dx_next = dx_dist.sample(seed=key_sample_x_next)
        x_next = x + dx_next * self.dt

        # Concatenate state and last num_frame_stack actions
        new_dynamics_params = dynamics_params.replace(key=next_key,
                                                      statistical_model_state=model_output.statistical_model_state)
        aleatoric_std = model_output.aleatoric_std * self.dt
        if not self.aleatoric_noise_in_prediction:
            aleatoric_std = 0 * aleatoric_std
        return Normal(loc=x_next, scale=aleatoric_std), new_dynamics_params

    def init_params(self, key: chex.PRNGKey) -> DynamicsParams:
        param_key, model_state_key = jr.split(key, 2)
        model_state = self.statistical_model.init(model_state_key)
        return DynamicsParams(key=key, statistical_model_state=model_state)


class ContinuousOptimisticDynamics(ContinuousPetsDynamics, Generic[ModelState]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.u_dim = self.x_dim + self.u_dim

    def next_state(self,
                   x: chex.Array,
                   u: chex.Array,
                   dynamics_params: DynamicsParams) -> Tuple[Distribution, DynamicsParams]:
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        # Create state-action pair
        a, eta = jnp.split(u, axis=-1, indices_or_sections=[self.u_dim - self.x_dim])
        z = jnp.concatenate([x, a])
        next_key, key_sample_x_next = jr.split(dynamics_params.key)
        model_output = self.statistical_model(input=z,
                                              statistical_model_state=dynamics_params.statistical_model_state)

        dx_next = model_output.mean + dynamics_params.statistical_model_state.beta * model_output.epistemic_std * eta
        x_next = x + dx_next * self.dt

        # Concatenate state and last num_frame_stack actions
        aleatoric_std = model_output.aleatoric_std * self.dt
        if not self.aleatoric_noise_in_prediction:
            aleatoric_std = 0 * aleatoric_std
        new_dynamics_params = dynamics_params.replace(key=next_key,
                                                      statistical_model_state=model_output.statistical_model_state)
        return Normal(loc=x_next, scale=aleatoric_std), new_dynamics_params
    
    
class ContinuousMeanDynamics(ContinuousPetsDynamics, Generic[ModelState]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def next_state(self,
                   x: chex.Array,
                   u: chex.Array,
                   dynamics_params: DynamicsParams) -> Tuple[Distribution | DynamicsParams]:
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        z = jnp.concatenate([x, u])
        next_key, key_sample_x_next = jr.split(dynamics_params.key)
        model_output = self.statistical_model(input=z,
                                              statistical_model_state=dynamics_params.statistical_model_state)
        dx_next = model_output.mean
        x_next = x + dx_next * self.dt

        # Concatenate state and last num_frame_stack actions
        aleatoric_std = model_output.aleatoric_std * self.dt
        if not self.aleatoric_noise_in_prediction:
            aleatoric_std = 0 * aleatoric_std
        new_dynamics_params = dynamics_params.replace(key=next_key,
                                                      statistical_model_state=model_output.statistical_model_state)
        return Normal(loc=x_next, scale=aleatoric_std), new_dynamics_params


class ContinuousPetsExplorationDynamics(ContinuousPetsDynamics, Generic[ModelState]):
    def __init__(self, use_log: bool = True, scale_with_aleatoric_std: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_log = use_log
        self.scale_with_aleatoric_std = scale_with_aleatoric_std

    def get_intrinsic_reward(self, epistemic_std: chex.Array, aleatoric_std: chex.Array) -> chex.Array:
        if self.scale_with_aleatoric_std:
            # sigma^2_ep / sigma^2_al
            intrinsic_reward = jnp.square(epistemic_std / jnp.clip(aleatoric_std, a_min=1e-4))
        else:
            # sigma^2_ep
            intrinsic_reward = jnp.square(epistemic_std)
        if self.use_log:
            # use log transform
            intrinsic_reward = jnp.log(1 + intrinsic_reward)
        # sum over the state axis
        return jnp.sum(intrinsic_reward, axis=0)

    def next_state(self,
                   x: chex.Array,
                   u: chex.Array,
                   dynamics_params: DynamicsParams) -> Tuple[Distribution, DynamicsParams]:
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        # Create state-action pair
        z = jnp.concatenate([x, u])
        next_key, key_sample_x_next = jr.split(dynamics_params.key)

        model_output = self.statistical_model(input=z,
                                              statistical_model_state=dynamics_params.statistical_model_state)
        scale_std = model_output.epistemic_std
        dx_dist = Normal(loc=model_output.mean, scale=scale_std)
        dx_next = dx_dist.sample(seed=key_sample_x_next)
        x_next = x + dx_next * self.dt

        aleatoric_std = model_output.aleatoric_std # TODO * self.dt?
        intrinsic_reward = self.get_intrinsic_reward(scale_std, aleatoric_std)
        intrinsic_reward = jnp.atleast_1d(intrinsic_reward)

        # Concatenate state and last num_frame_stack actions
        new_dynamics_params = dynamics_params.replace(key=next_key,
                                                      statistical_model_state=model_output.statistical_model_state)
        aleatoric_std = model_output.aleatoric_std * self.dt # TODO
        if not self.aleatoric_noise_in_prediction:
            aleatoric_std = 0 * aleatoric_std
        # add intrinsic reward to the next state
        x_next_with_reward = jnp.concatenate([x_next, intrinsic_reward], axis=-1)
        aleatoric_std_with_reward = jnp.concatenate([aleatoric_std, jnp.zeros_like(intrinsic_reward)], axis=-1)
        return Normal(loc=x_next_with_reward, scale=aleatoric_std_with_reward), new_dynamics_params


class ContinuousOptimisticExplorationDynamics(ContinuousPetsExplorationDynamics, Generic[ModelState]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.u_dim = self.x_dim + self.u_dim
    
    def next_state(self,
                   x: chex.Array,
                   u: chex.Array,
                   dynamics_params: DynamicsParams) -> Tuple[Distribution, DynamicsParams]:
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        # Create state-action pair
        a, eta = jnp.split(u, axis=-1, indices_or_sections=[self.u_dim - self.x_dim])
        z = jnp.concatenate([x, a])
        next_key, key_sample_x_next = jr.split(dynamics_params.key)
        model_output = self.statistical_model(input=z,
                                              statistical_model_state=dynamics_params.statistical_model_state)

        dx_next = model_output.mean + dynamics_params.statistical_model_state.beta * model_output.epistemic_std * eta
        x_next = x + dx_next * self.dt

        # Concatenate state and last num_frame_stack actions
        aleatoric_std = model_output.aleatoric_std # TODO: dt?
        intrinsic_reward = self.get_intrinsic_reward(model_output.epistemic_std, aleatoric_std)
        intrinsic_reward = jnp.atleast_1d(intrinsic_reward)
        new_dynamics_params = dynamics_params.replace(key=next_key,
                                                      statistical_model_state=model_output.statistical_model_state)
        aleatoric_std = model_output.aleatoric_std * self.dt # TODO
        if not self.aleatoric_noise_in_prediction:
            aleatoric_std = 0 * aleatoric_std
        # add intrinsic reward to the next state
        x_next_with_reward = jnp.concatenate([x_next, intrinsic_reward], axis=-1)
        aleatoric_std_with_reward = jnp.concatenate([aleatoric_std, jnp.zeros_like(intrinsic_reward)], axis=-1)
        return Normal(loc=x_next_with_reward, scale=aleatoric_std_with_reward), new_dynamics_params


class ContinuousMeanExplorationDynamics(ContinuousPetsExplorationDynamics, Generic[ModelState]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.u_dim = self.x_dim + self.u_dim
    
    def next_state(self,
                   x: chex.Array,
                   u: chex.Array,
                   dynamics_params: DynamicsParams) -> Tuple[Distribution, DynamicsParams]:
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        z = jnp.concatenate([x, u])
        next_key, key_sample_x_next = jr.split(dynamics_params.key)
        model_output = self.statistical_model(input=z,
                                              statistical_model_state=dynamics_params.statistical_model_state)
        dx_next = model_output.mean
        x_next = x + dx_next * self.dt

        epistemic_std = model_output.epistemic_std
        aleatoric_std = model_output.aleatoric_std # TODO * self.dt?
        intrinsic_reward = self.get_intrinsic_reward(epistemic_std, aleatoric_std)
        intrinsic_reward = jnp.atleast_1d(intrinsic_reward)

        # Concatenate state and last num_frame_stack actions
        new_dynamics_params = dynamics_params.replace(key=next_key,
                                                      statistical_model_state=model_output.statistical_model_state)
        aleatoric_std = model_output.aleatoric_std * self.dt # TODO
        if not self.aleatoric_noise_in_prediction:
            aleatoric_std = 0 * aleatoric_std
        # add intrinsic reward to the next state
        x_next_with_reward = jnp.concatenate([x_next, intrinsic_reward], axis=-1)
        aleatoric_std_with_reward = jnp.concatenate([aleatoric_std, jnp.zeros_like(intrinsic_reward)], axis=-1)
        return Normal(loc=x_next_with_reward, scale=aleatoric_std_with_reward), new_dynamics_params


class ContinuousPetsSystem(System, Generic[ModelState, RewardParams]):
    def __init__(self, dynamics: ContinuousPetsDynamics[ModelState], reward: Reward[RewardParams]):
        super().__init__(dynamics, reward)
        self.dynamics = dynamics
        self.reward = reward
        self.x_dim = dynamics.x_dim
        self.u_dim = dynamics.u_dim

    def get_reward(self,
                   x: chex.Array,
                   u: chex.Array,
                   reward_params: RewardParams,
                   x_next: chex.Array,
                   key: jax.random.PRNGKey):
        reward_dist, new_reward_params = self.reward(x, u, reward_params, x_next)
        reward = reward_dist.sample(seed=key)
        return reward, new_reward_params

    def step(self,
             x: chex.Array,
             u: chex.Array,
             system_params: SystemParams[ModelState, RewardParams],
             ) -> SystemState:
        """

        :param x: current state of the system
        :param u: current action of the system
        :param system_params: parameters of the system
        :return: Tuple of next state, reward, updated system parameters
        """
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        x_next_dist, new_dynamics_params = self.dynamics.next_state(x, u, system_params.dynamics_params)
        next_state_key, reward_key, new_systems_key = jr.split(system_params.key, 3)
        x_next = x_next_dist.sample(seed=next_state_key)
        reward, new_reward_params = self.get_reward(x, u, system_params.reward_params, x_next, reward_key)
        new_systems_params = system_params.replace(dynamics_params=new_dynamics_params,
                                                   reward_params=new_reward_params,
                                                   key=new_systems_key)
        new_system_state = SystemState(
            x_next=x_next,
            reward=reward,
            system_params=new_systems_params,
            done=jnp.array(0.0),
        )
        return new_system_state

    def vmap_input_axis(self, data_axis: int = 0) -> SystemParams[ModelState, RewardParams]:
        return SystemParams(
            dynamics_params=self.dynamics.vmap_input_axis(data_axis),
            reward_params=None,
            key=data_axis,
        )

    def vmap_output_axis(self, data_axis: int = 0) -> SystemState[ModelState, RewardParams]:
        return SystemState(
            x_next=data_axis,
            reward=data_axis,
            system_params=self.vmap_input_axis(data_axis),
            done=data_axis,
        )

    def system_params_vmap_axes(self, axes: int = 0):
        return self.vmap_input_axis(data_axis=axes)


class ContinuousOptimisticSystem(ContinuousPetsSystem, Generic[ModelState, RewardParams]):
    def __init__(self, dynamics: ContinuousOptimisticDynamics[ModelState], reward: Reward[RewardParams]):
        super().__init__(dynamics, reward)

    def get_reward(self,
                   x: chex.Array,
                   u: chex.Array,
                   reward_params: RewardParams,
                   x_next: chex.Array,
                   key: jax.random.PRNGKey):
        reward_dist, new_reward_params = self.reward(x, u[:self.u_dim - self.x_dim], reward_params, x_next)
        reward = reward_dist.sample(seed=key)
        return reward, new_reward_params

class ContinuousMeanSystem(ContinuousPetsSystem, Generic[ModelState, RewardParams]):
    def __init__(self, dynamics: ContinuousMeanDynamics[ModelState], reward: Reward[RewardParams]):
        super().__init__(dynamics, reward)

@chex.dataclass
class ExplorationRewardParams:
    action_cost: chex.Array | float = 0.0 # TODO: increase


class ContinuousExplorationReward(Reward, ExplorationRewardParams):
    def __init__(self, x_dim: int, u_dim: int):
        super().__init__(x_dim=x_dim, u_dim=u_dim)

    def __call__(self,
                 x: chex.Array,
                 u: chex.Array,
                 reward_params: ExplorationRewardParams,
                 x_next: chex.Array | None = None) -> Tuple[Distribution, RewardParams]:
        chex.assert_shape(x, (self.x_dim,))
        chex.assert_shape(u, (self.u_dim,))
        chex.assert_shape(x_next, (self.x_dim + 1,))
        # get intrinsic reward out
        intrinsic_reward = x_next[-1]
        total_reward = intrinsic_reward - reward_params.action_cost * jnp.sum(jnp.square(u), axis=0)
        return Normal(loc=total_reward, scale=jnp.zeros_like(total_reward)), reward_params

    def init_params(self, key: chex.PRNGKey) -> ExplorationRewardParams:
        return ExplorationRewardParams()


class ContinuousPetsExplorationSystem(ContinuousPetsSystem, Generic[ModelState, RewardParams]):
    def __init__(self, dynamics: ContinuousPetsExplorationDynamics[ModelState], reward: Reward[RewardParams] | None = None):
        if reward is None:
            # TODO: evtl pass control cost here. Currently zero.
            reward = ContinuousExplorationReward(x_dim=dynamics.x_dim, u_dim=dynamics.u_dim)
        super().__init__(dynamics, reward)

    def get_reward(self,
                   x: chex.Array,
                   u: chex.Array,
                   reward_params: RewardParams,
                   x_next: chex.Array,
                   key: jax.random.PRNGKey):
        # x_next includes the next state and the intrinsic reward
        chex.assert_shape(x_next, (self.x_dim + 1,))
        if isinstance(self.reward, ContinuousExplorationReward):
            # include the intrinsic reward in x_next
            reward_dist, new_reward_params = self.reward(x, u, reward_params, x_next)
        else:
            # ignore the last state in x_next which is the intrinsic reward
            reward_dist, new_reward_params = self.reward(x, u, reward_params, x_next[:-1])
        reward = reward_dist.sample(seed=key)
        return reward, new_reward_params

    def step(self,
             x: chex.Array,
             u: chex.Array,
             system_params: SystemParams[ModelState, RewardParams],
             ) -> SystemState:
        """

        :param x: current state of the system
        :param u: current action of the system
        :param system_params: parameters of the system
        :return: Tuple of next state, reward, updated system parameters
        """
        new_system_state = super().step(x, u, system_params)
        # remove the intrinsic reward from the state
        new_system_state = new_system_state.replace(
            x_next=new_system_state.x_next[:-1]
        )
        return new_system_state


class ContinuousOptimisticExplorationSystem(ContinuousPetsExplorationSystem, Generic[ModelState, RewardParams]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_reward(self,
                   x: chex.Array,
                   u: chex.Array,
                   reward_params: RewardParams,
                   x_next: chex.Array,
                   key: jax.random.PRNGKey):
        # x_next includes the next state and the intrinsic reward
        chex.assert_shape(x_next, (self.x_dim + 1,))
        if isinstance(self.reward, ContinuousExplorationReward):
            # include the intrinsic reward in x_next
            # exclude etas from the reward calculation
            reward_dist, new_reward_params = self.reward(x, u[:self.u_dim - self.x_dim], reward_params, x_next)
        else:
            # ignore the last state in x_next which is the intrinsic reward
            # exclude etas from the reward calculation
            reward_dist, new_reward_params = self.reward(x, u[:self.u_dim - self.x_dim], reward_params, x_next[:-1])
        reward = reward_dist.sample(seed=key)
        return reward, new_reward_params


class ContinuousMeanExplorationSystem(ContinuousPetsExplorationSystem, Generic[ModelState, RewardParams]):
    pass
