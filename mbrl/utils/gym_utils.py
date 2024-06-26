import jax
import gymnasium as gym
from gymnasium import Env, State
from typing import Sequence, Tuple

from brax.training.types import Transition

from mbpo.optimizers.base_optimizer import OptimizerState
from mbrl.model_based_agent.optimizer_wrapper import Actor

def env_step(
        env: Env,
        env_state: State,
        actor: Actor,
        actor_state: OptimizerState,
        extra_fields: Sequence[str] = (),
        evaluate: bool = False,
) -> Tuple[State, OptimizerState, Transition]:
    """Collect data."""
    action, new_actor_state = actor.act(env_state.obs, opt_state=actor_state, evaluate=evaluate)
    next_env_state = env.step(env_state, action)
    
    return next_env_state, new_actor_state, Transition(
        observation=env_state.obs,
        action=action,
        reward=next_env_state.reward,
        discount=1 - next_env_state.done,
        next_observation=next_env_state.pipeline_state.qd,
        extras={'state_extras': {x: next_env_state.info[x] for x in extra_fields}})

def generate_unroll(
        env: Env,
        env_state: State,
        actor: Actor,
        actor_state: OptimizerState,
        unroll_length: int,
        extra_fields: Sequence[str] = ('t'),
        evaluate: bool = False,
) -> Tuple[State, OptimizerState, Transition]:
    """Collect trajectories of given unroll_length."""

    @jax.jit
    def f(carry, unused_t):
        state, opt_state = carry
        nstate, new_opt_state, transition = env_step(
            env, state, actor, opt_state, extra_fields=extra_fields, evaluate=evaluate)
        return (nstate, new_opt_state), transition

    return jax.lax.scan(f, (env_state, actor_state), jnp.arange(unroll_length)
)