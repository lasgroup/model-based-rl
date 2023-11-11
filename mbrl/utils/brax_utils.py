from brax.envs import Env as BraxEnv
from brax.envs import State
from brax.training.types import Transition
from brax.envs import training
from brax.training.types import Metrics
from typing import Sequence, Tuple
import time
from mbpo.optimizers.base_optimizer import BaseOptimizer, OptimizerState
import jax
import numpy as np
import jax.numpy as jnp
from functools import partial


def flatten_transitions(transition: Transition):
    def convert_array(x: jax.Array):
        if x.ndim == 3:
            return x.reshape(-1, x.shape[-1])
        else:
            return x.reshape(-1)
    new_transition = jax.tree_util.tree_map(lambda x: convert_array(x), transition)
    return new_transition


def env_step(
        env: BraxEnv,
        env_state: State,
        optimizer: BaseOptimizer,
        optimizer_state: OptimizerState,
        extra_fields: Sequence[str] = (),
        evaluate: bool = False,
) -> Tuple[State, OptimizerState, Transition]:
    """Collect data."""
    actions, new_optimizer_state = optimizer.act(env_state.obs, opt_state=optimizer_state, evaluate=evaluate)
    nstate = env.step(env_state, actions)
    state_extras = {x: nstate.info[x] for x in extra_fields}
    return nstate, new_optimizer_state, Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=env_state.obs,
        action=actions,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation=nstate.obs,
        extras={
            'state_extras': state_extras
        })


def generate_unroll(
        env: BraxEnv,
        env_state: State,
        optimizer: BaseOptimizer,
        optimizer_state: OptimizerState,
        unroll_length: int,
        extra_fields: Sequence[str] = (),
        evaluate: bool = False,
) -> Tuple[State, OptimizerState, Transition]:
    """Collect trajectories of given unroll_length."""

    @jax.jit
    def f(carry, unused_t):
        state, opt_state = carry
        nstate, new_opt_state, transition = env_step(
            env, state, optimizer, opt_state, extra_fields=extra_fields, evaluate=evaluate)
        return (nstate, new_opt_state), transition

    (final_state, final_opt_state), data = jax.lax.scan(
        f, (env_state, optimizer_state), (), length=unroll_length)
    return final_state, final_opt_state, data


class BraxEvaluator:
    """Class to run evaluations."""

    def __init__(self, eval_env: BraxEnv,
                 num_eval_envs: int,
                 episode_length: int,
                 action_repeat: int,
                 key: jax.random.PRNGKey):
        """Init.

    Args:
      eval_env: Batched environment to run evals on.
      eval_policy_fn: Function returning the policy from the policy parameters.
      num_eval_envs: Each env will run 1 episode in parallel for each eval.
      episode_length: Maximum length of an episode.
      action_repeat: Number of physics steps per env step.
      key: RNG key.
    """
        self._key = key
        self._eval_walltime = 0.

        self.eval_env = training.EvalWrapper(eval_env)
        self.num_eval_envs = num_eval_envs
        self.action_repeat = action_repeat
        self.episode_length = episode_length
        self._steps_per_unroll = episode_length * num_eval_envs

    @partial(jax.jit, static_argnums=(0, 2))
    def generate_eval_unroll(self,
                             opt_state: OptimizerState,
                             optimizer: BaseOptimizer,
                             rng: jax.random.PRNGKey) -> State:
        reset_keys = jax.random.split(rng, self.num_eval_envs)
        eval_first_state = self.eval_env.reset(reset_keys)
        state = generate_unroll(
            env=self.eval_env,
            env_state=eval_first_state,
            optimizer=optimizer,
            optimizer_state=opt_state,
            unroll_length=self.episode_length // self.action_repeat,
            evaluate=True)[0]
        return state

    def run_evaluation(self,
                       optimizer_state: OptimizerState,
                       optimizer: BaseOptimizer,
                       aggregate_episodes: bool = True) -> Metrics:
        """Run one epoch of evaluation."""
        self._key, unroll_key, opt_key = jax.random.split(self._key, 3)
        eval_opt_state = optimizer_state.replace(key=opt_key)

        t = time.time()
        eval_state = self.generate_eval_unroll(eval_opt_state, optimizer, unroll_key)
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
        self._eval_walltime = self._eval_walltime + epoch_eval_time
        metrics = {
            'eval_true_env/walltime': self._eval_walltime,
            **metrics
        }

        return metrics  # pytype: disable=bad-return-type  # jax-ndarray


class BraxEnvCollector:
    def __init__(self,
                 env: BraxEnv,
                 key: jax.random.PRNGKey,
                 episode_length: int,
                 action_repeat: int = 1,
                 num_envs: int = 1,
                 num_eval_envs: int = 128,
                 env_steps_per_update: int = 1,
                 num_evals: int = 1,
                 eval_env: BraxEnv | None = None,
                 ):
        self.episode_length = episode_length
        self.action_repeat = action_repeat
        self.num_envs = num_envs
        self.num_eval_envs = num_eval_envs
        self.env_steps_per_update = env_steps_per_update
        self.num_evals = num_evals
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

        self.evaluator = BraxEvaluator(
            eval_env,
            num_eval_envs=self.num_eval_envs,
            episode_length=self.episode_length,
            action_repeat=self.action_repeat,
            key=key)

    def get_experience_brax(
            self,
            env_state: State,
            opt_state: OptimizerState,
            optimizer: BaseOptimizer,
    ) -> Tuple[State, OptimizerState, Transition]:
        env_state, new_opt_state, transitions = env_step(
            self.env, env_state, optimizer, opt_state, extra_fields=(), evaluate=False)
        return env_state, new_opt_state, transitions

    def generate_rollouts(self,
                          env_state: State,
                          optimizer_state: OptimizerState,
                          optimizer: BaseOptimizer,
                          num_env_steps: int | None = None
                          ):
        def get_rollouts(carry, unused):
            del unused
            opt_state, env_state = carry[0], carry[1]
            new_env_state, new_opt_state, transitions = self.get_experience_brax(
                env_state=env_state,
                opt_state=opt_state,
                optimizer=optimizer
            )
            carry = [new_opt_state, new_env_state]
            outs = transitions
            return carry, outs

        if num_env_steps:
            carry, transitions = jax.lax.scan(
                get_rollouts, [optimizer_state, env_state], (),
                length=num_env_steps)
        else:
            carry, transitions = jax.lax.scan(
                get_rollouts, [optimizer_state, env_state], (),
                length=self.env_steps_per_update)
        new_optimizer_state, env_state = carry[0], carry[1]
        return new_optimizer_state, env_state, flatten_transitions(transitions)

    def reset(self, env_key: jax.random.PRNGKey):
        env_keys = jax.random.split(env_key, self.num_envs)
        return self.env.reset(env_keys)

    def run_evaluation(self, optimizer_state: OptimizerState, optimizer: BaseOptimizer) -> Metrics:
        return self.evaluator.run_evaluation(
            optimizer_state=optimizer_state,
            optimizer=optimizer,
        )

    @property
    def env_steps_per_actor_step(self):
        return self.action_repeat * self.num_envs
