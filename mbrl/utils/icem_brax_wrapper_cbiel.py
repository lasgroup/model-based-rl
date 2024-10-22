from functools import partial
from typing import Tuple, List, Generic

import chex
import jax
from jax import jit
import jax.random as jr
import jax.numpy as jnp
from brax.training import types
from brax.training.types import Transition

from mbpo.optimizers.base_optimizer import BaseOptimizer
from mbrl.utils.icem import iCemTO, iCemParams, iCemOptimizerState
from mbpo.systems.base_systems import System, SystemParams

class iCEMOptimizer(BaseOptimizer):
    def __init__(self,
                 horizon: int,
                 opt_params: iCemParams = iCemParams(),
                 system: System | None = None,
                 key: jr.PRNGKey = jr.PRNGKey(0),
                 **agent_kwargs):
        super().__init__(system, key)
        self.horizon = horizon
        self.key = key
        self.opt_params = opt_params
        self.agent_class = iCemTO
        self.agent_kwargs = agent_kwargs
        if system is not None:
            self.set_system(system)

    def set_system(self, system: System):
        super().set_system(system)

    def init(self,
             key: chex.PRNGKey,
             true_buffer_state = None) -> iCemOptimizerState:
        # true_buffer_state is used to ensure compatibilitywith SAC and PPO
        self.agent = self.agent_class(horizon=self.horizon,
                                      action_dim=self.system.u_dim,
                                      key = self.key,
                                      opt_params=self.opt_params,
                                      **self.agent_kwargs)
        if self.system is not None:
            self.agent.set_system(self.system)
        if true_buffer_state is None:
            dummy_buffer_key, key = jr.split(key, 2)
            true_buffer_state = self.dummy_true_buffer_state(dummy_buffer_key)
        agent_state = self.agent.init(key)
        agent_state.true_buffer_state = true_buffer_state
        return agent_state

    @partial(jit, static_argnums=(0, 3))
    def act(self,
            obs: chex.Array,
            opt_state: iCemOptimizerState,
            evaluate: bool = True) -> Tuple[chex.Array, iCemOptimizerState]:
        assert self.system is not None, "Brax optimizer requires system to be defined."
        action, opt_state = self.agent.act(obs.reshape(-1,), opt_state, evaluate)
        return action.reshape(1, -1), opt_state
    
def rollout_actions(
        system: System,
        system_params: SystemParams,
        init_state: chex.Array,
        actions: chex.Array,
        horizon: int,
    ) -> Transition:

    assert actions.shape[0] == horizon
    
    def step(carry, acs):
        obs = carry[0]
        sys_params = carry[-1]
        
        system_output = system.step(
            x=obs.reshape(-1,),
            u=acs.reshape(-1,),
            system_params=sys_params,
        )
        next_obs = system_output.x_next
        reward = system_output.reward
        next_sys_params = system_output.system_params
        carry = [next_obs, next_sys_params]
        outs = [next_obs, reward]
        return carry, outs
    
    ins = actions
    carry = [init_state, system_params]
    _, outs = jax.lax.scan(step, carry, ins, length=horizon)
    next_state = outs[0]
    state = jnp.zeros_like(next_state)
    state = state.at[0, ...].set(init_state)
    state = state.at[1:, ...].set(next_state[:-1, ...])
    rewards = outs[1]
    transitions = Transition(
        observation=state,
        action=actions,
        reward=rewards,
        next_observation=next_state,
        discount=jnp.ones_like(rewards),
    )
    return transitions