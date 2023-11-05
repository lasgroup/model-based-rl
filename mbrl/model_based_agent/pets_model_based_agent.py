from abc import ABC
from typing import Generic, Union, Optional, Tuple
import jax
from gym import Env as GymEnv
from brax.envs import Env as BraxEnv
from brax import envs
from brax.envs import State
import jax.numpy as jnp
import chex
from bsm.statistical_model import StatisticalModel
from bsm.utils.type_aliases import ModelState
from mbpo.utils.type_aliases import OptimizerState
from mbpo.optimizers.base_optimizer import BaseOptimizer
import jax.random as jr
from brax.training.replay_buffers import UniformSamplingQueue, ReplayBufferState
from brax.training.types import Transition
from brax.training import pmap
import wandb
from mbrl.model_based_agent.system_wrapper import LearnedModelSystem, LearnedDynamics
from mbpo.systems.rewards.base_rewards import Reward, RewardParams
from bsm.utils.normalization import Data
from absl import logging
from functools import partial
from mbrl.model_based_agent.training_utils import env_step, BraxEvaluator, save_params, load_params
import time

_PMAP_AXIS_NAME = 'i'


@chex.dataclass
class ModelBasedAgentState(Generic[ModelState]):
    buffer_state: ReplayBufferState
    optimizer_state: OptimizerState[ModelState]
    env_steps: chex.Array = jnp.zeros(())


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


class PetsModelBasedAgent(ABC, Generic[ModelState, RewardParams]):
    def __init__(self,
                 env: Union[BraxEnv, GymEnv],
                 model: StatisticalModel[ModelState],
                 reward_model: Reward[RewardParams],
                 optimizer: BaseOptimizer[RewardParams],
                 episode_length: int,
                 action_repeat: int = 1,
                 num_envs: int = 1,
                 num_eval_envs: int = 128,
                 env_steps_per_update: int = 1,
                 num_evals: int = 1,
                 offline_data: Transition = None,
                 predict_difference: bool = True,
                 reset_bnn: bool = True,
                 return_best_bnn: bool = True,
                 bnn_training_test_ratio: float = 0.2,
                 return_best_optimizer: bool = True,
                 discounting: float = 0.99,
                 min_replay_size: int = 0,
                 max_replay_size_true_data_buffer: int = 10 ** 4,
                 key: chex.PRNGKey = jr.PRNGKey(0),
                 checkpoint_logdir: Optional[str] = None,
                 eval_env: Optional[Union[BraxEnv, GymEnv]] = None,
                 ):
        self.env = env
        self.model = model
        self.reward_model = reward_model
        self.optimizer = optimizer
        self.bnn_training_test_ratio = bnn_training_test_ratio
        self.predict_difference = predict_difference
        self.return_best_optimizer = return_best_optimizer
        self.return_best_bnn = return_best_bnn
        self.reset_bnn = reset_bnn
        self.discounting = discounting
        self.key = key
        self.state_dim, self.action_dim = self.env.observation_space.shape[0], self.env.action_space.shape[0]

        self.min_replay_size = min_replay_size

        dummy_obs = jnp.zeros(shape=(self.state_dim,))
        self.dummy_sample = Transition(observation=dummy_obs,
                                       action=jnp.zeros(shape=(self.action_dim,)),
                                       reward=jnp.array(0.0),
                                       discount=jnp.array(0.99),
                                       next_observation=dummy_obs)

        self.true_data_buffer = UniformSamplingQueue(
            max_replay_size=max_replay_size_true_data_buffer,
            dummy_data_sample=self.dummy_sample,
            sample_batch_size=1)

        # We now insert data into the true data buffer
        self.key, key_bf_init = jr.split(self.key)
        true_data_bf_state = self.true_data_buffer.init(key_bf_init)

        if offline_data:
            assert offline_data.observation.shape[-1] == self.state_dim and \
                   offline_data.action.shape[-1] == self.action_dim
            true_data_buffer_state = self.add_data_to_buffer(data=offline_data,
                                                             true_data_buffer_state=true_data_bf_state)
        else:
            true_data_buffer_state = true_data_bf_state

        self.init_states_buffer = UniformSamplingQueue(
            max_replay_size=max_replay_size_true_data_buffer,  # Should be larger than the number of episodes we run
            dummy_data_sample=self.dummy_sample,
            sample_batch_size=1)
        self.dynamics = LearnedDynamics(self.model)
        self.system = LearnedModelSystem(
            dynamics=self.dynamics,
            reward=self.reward_model,
        )
        self.init_true_data_buffer_state = true_data_buffer_state
        self.optimizer.set_system(system=self.system)

        # env variables
        self.action_repeat = action_repeat
        self.num_envs = num_envs
        self.num_eval_envs = num_eval_envs
        self.episode_length = episode_length
        self.env_steps_per_update = env_steps_per_update

        self.eval_env = eval_env
        self.checkpoint_logdir = checkpoint_logdir
        self.num_evals = num_evals

    def init(self, key: chex.Array):
        init_optimizer_state = self.optimizer.init(key)
        return ModelBasedAgentState(buffer_state=self.init_true_data_buffer_state,
                                    optimizer_state=init_optimizer_state,
                                    env_steps=0
                                    )

    def init_state_for_brax(self, key: chex.Array, local_devices_to_use: int):
        agent_state = self.init(key)
        return jax.device_put_replicated(agent_state,
                                         jax.local_devices()[:local_devices_to_use])

    def add_data_to_buffer(self, data: Transition, agent_state: ModelBasedAgentState) -> ModelBasedAgentState:
        new_buffer_state = self.true_data_buffer.insert(agent_state.buffer_state, data)
        new_agent_state = agent_state.replace(buffer_state=new_buffer_state)
        return new_agent_state

    @staticmethod
    def define_wandb_metrics():
        wandb.define_metric('x_axis/episode')

    def get_data_for_model_training_from_buffer_state(self, buffer_state: ReplayBufferState):
        idx = jnp.arange(start=buffer_state.sample_position, stop=buffer_state.insert_position)
        all_data = jnp.take(buffer_state.data, idx, axis=0, mode='wrap')
        all_transitions = self.true_data_buffer._unflatten_fn(all_data)
        obs = all_transitions.observation
        actions = all_transitions.action
        inputs = jnp.concatenate([obs, actions], axis=-1)
        next_obs = all_transitions.next_observation
        if self.predict_difference:
            outputs = next_obs - obs
        else:
            outputs = next_obs
        return Data(inputs=inputs, outputs=outputs)

    def train_dynamics_model(self,
                             agent_state: ModelBasedAgentState,
                             ) -> ModelBasedAgentState:
        # Prepare data
        buffer_state = agent_state.buffer_state
        data = self.get_data_for_model_training_from_buffer_state(buffer_state)
        new_model_state = self.model.update(
            statistical_model_state=agent_state.optimizer_state.system_params.dynamics_params.model_state,
            data=data)
        new_dynamics_params = agent_state.optimizer_state.system_params.dynamics_params.replace(
            model_state=new_model_state)
        new_system_params = agent_state.optimizer_state.system_params.replace(
            dynamics_params=new_dynamics_params)
        new_optimizer_state = agent_state.optimizer_state.replace(system_params=new_system_params)
        new_agent_state = agent_state.replace(optimizer_state=new_optimizer_state)
        return new_agent_state

    def train_policy(self, agent_state: ModelBasedAgentState) -> ModelBasedAgentState:
        optimizer_state = agent_state.optimizer_state
        optimizer_state = optimizer_state.replace(true_buffer_state=agent_state.buffer_state)

        optimizer_output = self.optimizer.train(
            opt_state=optimizer_state
        )
        new_agent_state = agent_state.replace(optimizer_state=optimizer_output.optimizer_state)
        return new_agent_state

    @partial(jax.jit, static_argnum=0)
    def train_step(self, agent_state: ModelBasedAgentState) -> ModelBasedAgentState:
        agent_state_after_model_training = self.train_dynamics_model(agent_state)
        trained_agent_state = self.train_policy(agent_state_after_model_training)
        return trained_agent_state

    def train_brax_env(self, num_timesteps: int, max_devices_per_host: Optional[int] = None, ):
        process_id = jax.process_index()
        local_devices_to_use = jax.local_device_count()
        if max_devices_per_host is not None:
            local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
        device_count = local_devices_to_use * jax.process_count()
        logging.info('local_device_count: %s; total_device_count: %s',
                     local_devices_to_use, device_count)

        if self.min_replay_size >= num_timesteps:
            raise ValueError(
                'No training will happen because min_replay_size >= num_timesteps')

        env_steps_per_actor_step = self.action_repeat * self.num_envs
        # equals to ceil(min_replay_size / env_steps_per_actor_step)
        num_prefill_actor_steps = -(-self.min_replay_size // self.num_envs)

        num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
        assert num_timesteps - num_prefill_env_steps >= 0
        num_evals_after_init = max(self.num_eval_envs - 1, 1)

        num_training_steps_per_epoch = -(
                -(num_timesteps - num_prefill_env_steps) //
                (num_evals_after_init * env_steps_per_actor_step))

        assert self.num_envs % device_count == 0
        wrap_for_training = envs.training.wrap

        rng, self.key = jax.random.split(self.key)

        env = wrap_for_training(
            self.env,
            episode_length=self.episode_length,
            action_repeat=self.action_repeat,
        )

        def get_experience(
                opt_state: OptimizerState,
                env_state: State,
                buffer_state: ReplayBufferState
        ) -> Tuple[State, OptimizerState, ReplayBufferState]:
            env_state, new_opt_state, transitions = env_step(
                env, env_state, self.optimizer, opt_state, extra_fields=('truncation',), evaluate=False)

            buffer_state = self.true_data_buffer.insert(buffer_state, transitions)
            return env_state, new_opt_state, buffer_state

        def get_rollouts(carry, unused):
            del unused
            ag_state, env_state = carry
            new_env_state, new_opt_state, new_buffer_state = get_experience(
                opt_state=ag_state.optimizer_state,
                env_state=env_state,
                buffer_state=ag_state.buffer_state
            )
            new_ag_state = ag_state.replace(
                env_steps=ag_state.env_steps + env_steps_per_actor_step,
                buffer_state=new_buffer_state,
                optimizer_state=new_opt_state,
            )
            return (new_ag_state, new_env_state), ()

        def training_step(
                agent_state: ModelBasedAgentState, env_state: State,
        ) -> Tuple[ModelBasedAgentState, State]:

            (new_agent_state, env_state), _ = jax.lax.scan(
                get_rollouts, (agent_state, env_state), (),
                length=self.env_steps_per_update)

            trained_agent_state = self.train_step(new_agent_state)
            return trained_agent_state, env_state

        def prefill_replay_buffer(
                agent_state: ModelBasedAgentState, env_state: State,
        ) -> Tuple[ModelBasedAgentState, State]:

            return jax.lax.scan(
                get_rollouts, (agent_state, env_state), (),
                length=num_prefill_actor_steps)[0]

        prefill_replay_buffer = jax.pmap(
            prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME)

        def training_epoch(
                agent_state: ModelBasedAgentState, env_state: State
        ) -> Tuple[ModelBasedAgentState, State]:

            def f(carry: Tuple[ModelBasedAgentState, State], unused_t):
                del unused_t
                ags, es = carry
                nags, es = training_step(ags, es)
                return (nags, es), ()

            (new_agent_state, env_state), _ = jax.lax.scan(
                f, (agent_state, env_state), (),
                length=num_training_steps_per_epoch)
            return new_agent_state, env_state

        def training_epoch_with_timing(
                agent_state: ModelBasedAgentState, env_state: envs.State
        ) -> Tuple[ModelBasedAgentState, envs.State]:
            nonlocal training_walltime
            t = time.time()
            new_agent_state, new_env_state = training_epoch(agent_state, env_state)
            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time
            return new_agent_state, new_env_state  # pytype: disable=bad-return-type  # py311-upgrade

        # Training state init

        global_key, local_key = jax.random.split(rng)
        local_key = jax.random.fold_in(local_key, process_id)
        agent_state = self.init_state_for_brax(global_key)
        del global_key

        rb_key, env_key, eval_key = jax.random.split(local_key, 3)

        # Env init
        env_keys = jax.random.split(env_key, self.num_envs // jax.process_count())
        env_keys = jnp.reshape(env_keys,
                               (local_devices_to_use, -1) + env_keys.shape[1:])
        env_state = jax.pmap(env.reset)(env_keys)

        # Replay buffer init
        # buffer_state = jax.pmap(self.true_data_buffer.init)(
        #    jax.random.split(rb_key, local_devices_to_use))
        # buffer_state_transitions = self.true_data_buffer._unflatten_fn(agent_state.buffer_state)
        # buffer_state = jax.pmap(self.true_data_buffer.insert(buffer_state,
        #                                                     buffer_state_transitions))(buffer_state)
        # agent_state = agent_state.replace(buffer_state=buffer_state)

        if not self.eval_env:
            eval_env = self.env
        else:
            eval_env = self.eval_env
        eval_env = wrap_for_training(
            eval_env,
            episode_length=self.episode_length,
            action_repeat=self.action_repeat,
        )

        evaluator = BraxEvaluator(
            eval_env,
            optimizer=self.optimizer,
            num_eval_envs=self.num_eval_envs,
            episode_length=self.episode_length,
            action_repeat=self.action_repeat,
            key=eval_key)

        # Run initial eval
        metrics = {}
        if process_id == 0 and self.num_evals > 1:
            metrics = evaluator.run_evaluation(_unpmap(agent_state.optimizer_state))
            logging.info(metrics)

        # Create and initialize the replay buffer.
        t = time.time()
        prefill_key, local_key = jax.random.split(local_key)
        prefill_keys = jax.random.split(prefill_key, local_devices_to_use)
        agent_state, env_state, _ = prefill_replay_buffer(
            agent_state, env_state, prefill_keys)

        replay_size = jnp.sum(jax.vmap(
            self.true_data_buffer.size)(agent_state.buffer_state)) * jax.process_count()
        logging.info('replay size after prefill %s', replay_size)
        assert replay_size >= self.min_replay_size
        training_walltime = time.time() - t

        current_step = 0
        for _ in range(num_evals_after_init):
            logging.info('step %s', current_step)

            # Optimization
            agent_state, env_state = training_epoch_with_timing(agent_state=agent_state, env_state=env_state)
            current_step = int(_unpmap(agent_state.env_steps))

        if process_id == 0:
            if self.checkpoint_logdir:
                # Save current policy.
                saved_agent_state = _unpmap(agent_state)
                path = f'{self.checkpoint_logdir}_PETS_{current_step}.pkl'
                save_params(path, saved_agent_state)

                # model.save_params(path, params)

            # Run evals.
            metrics = evaluator.run_evaluation(
                _unpmap(
                    agent_state.optimizer_state))
            logging.info(metrics)

        total_steps = current_step
        assert total_steps >= num_timesteps

        final_state = _unpmap(agent_state)
        # If there was no mistakes the training_state should still be identical on all
        # devices.
        pmap.assert_is_replicated(agent_state)
        logging.info('total steps: %s', total_steps)
        pmap.synchronize_hosts()
        return final_state
