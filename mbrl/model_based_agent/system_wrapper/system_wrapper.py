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


class PetsDynamics(Dynamics, Generic[ModelState]):
    def __init__(self,
                 x_dim: int,
                 u_dim: int,
                 statistical_model: StatisticalModel,
                 aleatoric_noise_in_prediction: bool = True,
                 predict_difference: bool = True,
                 ):
        Dynamics.__init__(self, x_dim=x_dim, u_dim=u_dim)
        self.statistical_model = statistical_model
        self.aleatoric_noise_in_prediction = aleatoric_noise_in_prediction
        self.predict_difference = predict_difference

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
        if self.predict_difference:
            model_output = self.statistical_model(input=z,
                                                  statistical_model_state=dynamics_params.statistical_model_state)
            scale_std = model_output.epistemic_std
            delta_x_dist = Normal(loc=model_output.mean, scale=scale_std)
            delta_x = delta_x_dist.sample(seed=key_sample_x_next)
            x_next = x + delta_x
        else:
            model_output = self.statistical_model(input=z,
                                                  statistical_model_state=dynamics_params.statistical_model_state)
            scale_std = model_output.epistemic_std
            x_next_dist = Normal(loc=model_output.mean, scale=scale_std)
            x_next = x_next_dist.sample(seed=key_sample_x_next)

        # Concatenate state and last num_frame_stack actions
        new_dynamics_params = dynamics_params.replace(key=next_key,
                                                      statistical_model_state=model_output.statistical_model_state)
        aleatoric_std = model_output.aleatoric_std
        if not self.aleatoric_noise_in_prediction:
            aleatoric_std = 0 * aleatoric_std
        return Normal(loc=x_next, scale=aleatoric_std), new_dynamics_params

    def init_params(self, key: chex.PRNGKey) -> DynamicsParams:
        param_key, model_state_key = jr.split(key, 2)
        model_state = self.statistical_model.init(model_state_key)
        return DynamicsParams(key=key, statistical_model_state=model_state)


class OptimisticDynamics(PetsDynamics, Generic[ModelState]):
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
        if self.predict_difference:
            model_output = self.statistical_model(input=z,
                                                  statistical_model_state=dynamics_params.statistical_model_state)
            delta_x = model_output.mean + dynamics_params.statistical_model_state.beta * model_output.epistemic_std * eta
            x_next = x + delta_x
        else:
            model_output = self.statistical_model(input=z,
                                                  statistical_model_state=dynamics_params.statistical_model_state)
            x_next = model_output.mean + dynamics_params.statistical_model_state.beta * model_output.epistemic_std * eta

        # Concatenate state and last num_frame_stack actions
        new_dynamics_params = dynamics_params.replace(key=next_key,
                                                      statistical_model_state=model_output.statistical_model_state)
        return Normal(loc=x_next, scale=model_output.aleatoric_std), new_dynamics_params


class PetsSystem(System, Generic[ModelState, RewardParams]):
    def __init__(self, dynamics: PetsDynamics[ModelState], reward: Reward[RewardParams]):
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


class OptimisticSystem(PetsSystem, Generic[ModelState, RewardParams]):
    def __init__(self, dynamics: OptimisticDynamics[ModelState], reward: Reward[RewardParams]):
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
