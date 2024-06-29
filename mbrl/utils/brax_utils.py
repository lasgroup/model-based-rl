import time
from functools import partial
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs import Env as BraxEnv
from brax.envs import State
from brax.envs import training
from brax.training.types import Metrics
from brax.training.types import Transition
import jax.tree_util as jtu
from mbpo.optimizers.base_optimizer import OptimizerState
from mbrl.model_based_agent.optimizer_wrapper import Actor


def env_step(
        env: BraxEnv,
        env_state: State,
        actor: Actor,
        actor_state: OptimizerState,
        extra_fields: Sequence[str] = (),
        evaluate: bool = False,
) -> Tuple[State, OptimizerState, Transition]:
    """Collect data.

    Note: discount = 1 - next_env_state.done because true environment has no discounting. At terminal state s_T
    we have V(s_T) = r(s_T, a) instead of the typical V(s) = r(s, a) + discount * V(s')
    Setting discount = 1 - done gives the desired behavior.
    """
    action, new_actor_state = actor.act(env_state.obs, opt_state=actor_state, evaluate=evaluate)
    next_env_state = env.step(env_state, action)
    state_extras = {}
    state_extras['true_derivative'] = next_env_state.info['true_derivative']
    state_extras['t'] = env_state.info['t']
    # state_extras = {x: next_env_state.info[x] for x in extra_fields}
    return next_env_state, new_actor_state, Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=env_state.obs,
        action=action,
        reward=next_env_state.reward,
        discount=1 - next_env_state.done,
        next_observation=next_env_state.obs,
        extras={'state_extras': state_extras})


def generate_unroll(
        env: BraxEnv,
        env_state: State,
        actor: Actor,
        actor_state: OptimizerState,
        unroll_length: int,
        extra_fields: Sequence[str] = ('t', 'true_derivative'),
        evaluate: bool = False,
) -> Tuple[State, OptimizerState, Transition]:
    """Collect trajectories of given unroll_length."""

    @jax.jit
    def f(carry, unused_t):
        state, opt_state = carry
        nstate, new_opt_state, transition = env_step(
            env, state, actor, opt_state, extra_fields=extra_fields, evaluate=evaluate)
        return (nstate, new_opt_state), transition

    (final_state, final_opt_state), data = jax.lax.scan(
        f, (env_state, actor_state), (), length=unroll_length)
    return final_state, final_opt_state, data


class Evaluator:
    """Class to run evaluations."""

    def __init__(self,
                 eval_env: BraxEnv,
                 num_eval_envs: int,
                 episode_length: int,
                 action_repeat: int,
                 key: jax.random.PRNGKey):
        """Init.

    Args:
      eval_env: Batched environment to run evals on.
      num_eval_envs: Each env will run 1 episode in parallel for each eval.
      episode_length: Maximum length of an episode.
      action_repeat: Number of physics steps per env step.
      key: RNG key.
    """
        self._key = key
        self._eval_wall_time = 0.

        self.eval_env = training.EvalWrapper(eval_env)
        self.num_eval_envs = num_eval_envs
        self.action_repeat = action_repeat
        self.episode_length = episode_length
        self._steps_per_unroll = episode_length * num_eval_envs

    @partial(jax.jit, static_argnums=(0, 2))
    def generate_eval_unroll(self,
                             opt_state: OptimizerState,
                             actor: Actor,
                             rng: jax.random.PRNGKey) -> State:
        reset_keys = jax.random.split(rng, self.num_eval_envs)
        eval_first_state = self.eval_env.reset(reset_keys)
        state = generate_unroll(
            env=self.eval_env,
            env_state=eval_first_state,
            actor=actor,
            actor_state=opt_state,
            unroll_length=self.episode_length // self.action_repeat,
            evaluate=True)[0]
        return state

    def run_evaluation(self,
                       actor_state: OptimizerState,
                       actor: Actor,
                       aggregate_episodes: bool = True) -> Metrics:
        """Run one epoch of evaluation."""
        self._key, unroll_key, opt_key = jax.random.split(self._key, 3)
        eval_opt_state = actor_state.replace(key=opt_key)

        t = time.time()
        eval_state = self.generate_eval_unroll(eval_opt_state, actor, unroll_key)
        eval_metrics = eval_state.info['eval_metrics']
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        metrics = {}
        for fn in [np.mean, np.std]:
            suffix = '_std' if fn == np.std else ''
            metrics.update(
                {
                    f'eval_true_env/episode_{name}{suffix}': (
                        fn(value) if aggregate_episodes else value
                    )
                    for name, value in eval_metrics.episode_metrics.items()
                }
            )
        metrics['eval_true_env/avg_episode_length'] = np.mean(eval_metrics.episode_steps)
        metrics['eval_true_env/epoch_eval_time'] = epoch_eval_time
        metrics['eval_true_env/sps'] = self._steps_per_unroll / epoch_eval_time
        self._eval_wall_time = self._eval_wall_time + epoch_eval_time
        metrics = {
            'eval_true_env/wall_time': self._eval_wall_time,
            **metrics
        }

        return metrics  # pytype: disable=bad-return-type  # jax-ndarray


class EnvInteractor:
    def __init__(self,
                 env: BraxEnv,
                 key: jax.random.PRNGKey,
                 episode_length: int,
                 action_repeat: int = 1,
                 num_envs: int = 1,
                 num_eval_envs: int = 128,
                 deterministic_policy_for_data_collection: bool = True,
                 eval_env: BraxEnv | None = None,
                 ):
        self.episode_length = episode_length
        self.action_repeat = action_repeat
        self.num_envs = num_envs
        self.num_eval_envs = num_eval_envs
        self.deterministic_policy_for_data_collection = deterministic_policy_for_data_collection
        wrap_for_training = training.wrap
        self.env = wrap_for_training(
            env,
            episode_length=self.episode_length,
            action_repeat=self.action_repeat,
        )

        if not eval_env:
            eval_env = env

        eval_env = wrap_for_training(
            eval_env,
            episode_length=self.episode_length,
            action_repeat=self.action_repeat,
        )

        self.evaluator = Evaluator(
            eval_env,
            num_eval_envs=self.num_eval_envs,
            episode_length=self.episode_length,
            action_repeat=self.action_repeat,
            key=key)

    def _make_one_env_step(self,
                           env_state: State,
                           opt_state: OptimizerState,
                           actor: Actor,
                           ) -> Tuple[State, OptimizerState, Transition]:
        env_state, new_opt_state, transitions = env_step(
            self.env, env_state, actor, opt_state,
            extra_fields=('t', 'true_derivative'),
            evaluate=self.deterministic_policy_for_data_collection)
        return env_state, new_opt_state, transitions

    def generate_rollouts(self,
                          env_state: State,
                          actor_state: OptimizerState,
                          actor: Actor,
                          unroll_length: int | None = None
                          ) -> Tuple[State, OptimizerState, Transition]:
        def get_rollouts(carry, _):
            opt_state, env_state = carry
            new_env_state, new_opt_state, transition = self._make_one_env_step(
                env_state=env_state,
                opt_state=opt_state,
                actor=actor
            )
            carry = (new_opt_state, new_env_state)
            outs = transition
            return carry, outs

        if unroll_length:
            carry, transitions = jax.lax.scan(
                get_rollouts, (actor_state, env_state), (),
                length=unroll_length)
        else:
            carry, transitions = jax.lax.scan(
                get_rollouts, (actor_state, env_state), (),
                length=self.episode_length // self.action_repeat)
        new_actor_state, env_state = carry
        return env_state, new_actor_state, jtu.tree_map(jnp.concatenate, transitions)

    def reset(self, key: jax.random.PRNGKey) -> State:
        env_keys = jax.random.split(key, self.num_envs)
        return self.env.reset(env_keys)

    def run_evaluation(self,
                       actor_state: OptimizerState,
                       actor: Actor) -> Metrics:
        return self.evaluator.run_evaluation(
            actor_state=actor_state,
            actor=actor,
        )

    @property
    def env_steps_per_actor_step(self):
        return self.action_repeat * self.num_envs
