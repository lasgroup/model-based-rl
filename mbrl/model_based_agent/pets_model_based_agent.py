from abc import ABC
from typing import Union, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import wandb
from brax.envs import Env as BraxEnv
from brax.envs import State
from brax.training.replay_buffers import UniformSamplingQueue, ReplayBufferState
from brax.training.types import Transition
from bsm.statistical_model import StatisticalModel
from bsm.utils.normalization import Data
from gym import Env as GymEnv
from mbpo.optimizers.base_optimizer import BaseOptimizer
from mbpo.systems.rewards.base_rewards import Reward, RewardParams
from mbpo.utils.type_aliases import OptimizerState

from mbrl.model_based_agent.system_wrapper import LearnedModelSystem, LearnedDynamics
from mbrl.utils.brax_utils import EnvInteractor
from mbrl.utils.training_utils import save_params, metrics_to_float


@chex.dataclass
class ModelBasedAgentState:
    buffer_state: ReplayBufferState
    optimizer_state: OptimizerState
    env_steps: chex.Array


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


class PetsModelBasedAgent(ABC):
    def __init__(self,
                 env: Union[BraxEnv, GymEnv],
                 statistical_model: StatisticalModel,
                 reward_model: Reward,
                 optimizer: BaseOptimizer,
                 episode_length: int,
                 action_repeat: int = 1,
                 num_envs: int = 1,
                 num_eval_envs: int = 128,
                 env_steps_per_update: int = 1,
                 num_evals: int = 1,
                 offline_data: Transition | None = None,
                 predict_difference: bool = True,
                 reset_bnn: bool = True,
                 return_best_bnn: bool = True,
                 bnn_training_test_ratio: float = 0.2,
                 discounting: float = 0.99,
                 min_replay_size: int = 0,
                 max_replay_size_true_data_buffer: int = 10 ** 4,
                 key: chex.PRNGKey = jr.PRNGKey(0),
                 checkpoint_logdir: Optional[str] = None,
                 eval_env: Optional[Union[BraxEnv, GymEnv]] = None,
                 log_to_wandb: bool = False,
                 return_best_agent: bool = False
                 ):
        self.model = statistical_model
        self.reward_model = reward_model
        self.optimizer = optimizer
        self.bnn_training_test_ratio = bnn_training_test_ratio
        self.predict_difference = predict_difference
        self.return_best_bnn = return_best_bnn
        self.reset_bnn = reset_bnn
        self.discounting = discounting
        self.key = key
        if isinstance(env, GymEnv):
            self.state_dim, self.action_dim = env.observation_space.shape[0], env.action_space.shape[0]
        else:
            self.state_dim, self.action_dim = env.observation_size, env.action_size

        self.min_replay_size = min_replay_size

        dummy_obs = jnp.zeros(shape=(self.state_dim,))
        self.dummy_sample = Transition(observation=dummy_obs,
                                       action=jnp.zeros(shape=(self.action_dim,)),
                                       reward=jnp.array(0.0),
                                       discount=jnp.array(0.99),
                                       next_observation=dummy_obs,
                                       )

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
            true_data_buffer_state = self.add_data_to_buffer(data=offline_data, buffer_state=true_data_bf_state)
        else:
            true_data_buffer_state = true_data_bf_state

        self.init_states_buffer = UniformSamplingQueue(
            max_replay_size=max_replay_size_true_data_buffer,  # Should be larger than the number of episodes we run
            dummy_data_sample=self.dummy_sample,
            sample_batch_size=1)
        self.dynamics = LearnedDynamics(statistical_model=self.model, x_dim=self.state_dim, u_dim=self.action_dim)
        self.system = LearnedModelSystem(
            dynamics=self.dynamics,
            reward=self.reward_model,
        )
        self.init_true_data_buffer_state = true_data_buffer_state
        self.optimizer.set_system(system=self.system)

        if eval_env is None:
            eval_env = env

        self.key, data_collector_key = jax.random.split(self.key)
        if isinstance(env, BraxEnv):
            assert isinstance(eval_env, BraxEnv), "Evaluation env must be of the same type as the training env."
            self.data_collector = EnvInteractor(
                env=env,
                key=data_collector_key,
                episode_length=episode_length,
                action_repeat=action_repeat,
                num_envs=num_envs,
                num_eval_envs=num_eval_envs,
                env_steps_per_update=env_steps_per_update,
                num_evals=num_evals,
                eval_env=eval_env,
            )
        else:
            self.data_collector = EnvInteractor(
                env=env,
                key=data_collector_key,
                episode_length=episode_length,
                action_repeat=action_repeat,
                num_envs=num_envs,
                num_eval_envs=num_eval_envs,
                env_steps_per_update=env_steps_per_update,
                num_evals=num_evals,
                eval_env=eval_env,
            )

        self.checkpoint_logdir = checkpoint_logdir
        self.wandb_logging = log_to_wandb
        self.return_best_agent = return_best_agent

    def init(self, key: chex.Array):
        init_optimizer_state = self.optimizer.init(key)
        return ModelBasedAgentState(buffer_state=self.init_true_data_buffer_state,
                                    optimizer_state=init_optimizer_state,
                                    env_steps=0,
                                    )

    @property
    def agent_name(self) -> str:
        return 'PETS'

    def add_data_to_buffer(self, data: Transition, buffer_state: ReplayBufferState) -> ReplayBufferState:
        new_buffer_state = self.true_data_buffer.insert(buffer_state, data)
        return new_buffer_state

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
            stats_model_state=agent_state.optimizer_state.system_params.dynamics_params.model_state,
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

    def train_step(self, agent_state: ModelBasedAgentState) -> ModelBasedAgentState:
        agent_state_after_model_training = self.train_dynamics_model(agent_state)
        trained_agent_state = self.train_policy(agent_state_after_model_training)
        return trained_agent_state

    def train(self, num_timesteps: int):

        if self.min_replay_size >= num_timesteps:
            raise ValueError(
                'No training will happen because min_replay_size >= num_timesteps')

        env_steps_per_actor_step = self.data_collector.env_steps_per_actor_step
        num_prefill_actor_steps = -(-self.min_replay_size // self.data_collector.num_envs)

        num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
        assert num_timesteps - num_prefill_env_steps >= 0
        num_evals_after_init = max(self.data_collector.num_evals - 1, 1)

        num_training_steps = -(
                -(num_timesteps - num_prefill_env_steps) // env_steps_per_actor_step)

        eval_frequency = num_training_steps // num_evals_after_init
        rng, self.key = jax.random.split(self.key)

        def training_step(
                agent_state: ModelBasedAgentState, env_state: State,
        ) -> Tuple[ModelBasedAgentState, State]:
            new_optimizer_state, env_state, transitions = self.data_collector.generate_rollouts(
                env_state=env_state,
                optimizer_state=agent_state.optimizer_state,
                optimizer=self.optimizer,
            )

            buffer_state = agent_state.buffer_state
            new_buffer_state = self.true_data_buffer.insert(buffer_state, transitions)
            total_env_steps = agent_state.env_steps + self.data_collector.env_steps_per_update \
                              * self.data_collector.num_envs
            new_agent_state = agent_state.replace(
                buffer_state=new_buffer_state,
                env_steps=total_env_steps,
                optimizer_state=new_optimizer_state,
            )

            trained_agent_state = self.train_step(new_agent_state)
            return trained_agent_state, env_state

        def prefill_replay_buffer(
                agent_state: ModelBasedAgentState, env_state: State,
        ) -> Tuple[ModelBasedAgentState, State]:

            new_optimizer_state, env_state, transitions = self.data_collector.generate_rollouts(
                env_state=env_state,
                optimizer_state=agent_state.optimizer_state,
                optimizer=self.optimizer,
                unroll_length=num_prefill_env_steps,
            )

            buffer_state = agent_state.buffer_state
            new_buffer_state = self.true_data_buffer.insert(buffer_state, transitions)
            total_env_steps = agent_state.env_steps + num_prefill_env_steps
            new_agent_state = agent_state.replace(
                buffer_state=new_buffer_state,
                env_steps=total_env_steps,
                optimizer_state=new_optimizer_state,
            )
            return new_agent_state, env_state

        # Training state init

        global_key, local_key = jax.random.split(rng)
        agent_state = self.init(global_key)
        del global_key

        rb_key, env_key, eval_key = jax.random.split(local_key, 3)

        # Env init
        env_state = self.data_collector.reset(env_key)

        # Replay buffer init
        # buffer_state = jax.pmap(self.true_data_buffer.init)(
        #    jax.random.split(rb_key, local_devices_to_use))
        # buffer_state_transitions = self.true_data_buffer._unflatten_fn(agent_state.buffer_state)
        # buffer_state = jax.pmap(self.true_data_buffer.insert(buffer_state,
        #                                                     buffer_state_transitions))(buffer_state)
        # agent_state = agent_state.replace(buffer_state=buffer_state)

        all_metrics = []
        highest_eval_episode_reward = jnp.array(-jnp.inf)
        best_state = agent_state

        # Run initial eval
        if self.data_collector.num_evals > 1:
            metrics = self.data_collector.run_evaluation(agent_state.optimizer_state, self.optimizer)
            highest_eval_episode_reward = metrics['eval_true_env/episode_reward']
            # logging.info(metrics)
            if self.wandb_logging:
                metrics = metrics_to_float(metrics)
                wandb.log(metrics)

            all_metrics.append(metrics)

        # Create and initialize the replay buffer.
        agent_state, env_state = prefill_replay_buffer(
            agent_state, env_state)

        replay_size = self.true_data_buffer.size(agent_state.buffer_state)
        # logging.info('replay size after prefill %s', replay_size)
        assert replay_size >= self.min_replay_size

        current_step = 0
        for ts in range(num_training_steps):
            # logging.info('step %s', current_step)

            # Optimization
            agent_state, env_state = training_step(agent_state=agent_state, env_state=env_state)
            current_step = int(agent_state.env_steps)
            if ts % eval_frequency == 0:
                metrics = self.data_collector.run_evaluation(agent_state.optimizer_state, self.optimizer)
                if metrics['eval_true_env/episode_reward'] > highest_eval_episode_reward:
                    highest_eval_episode_reward = metrics['eval_true_env/episode_reward']
                    best_state = agent_state
                else:
                    best_state = best_state.replace(
                        buffer_state=agent_state.buffer_state,
                        env_steps=agent_state.env_steps,
                    )
                if self.wandb_logging:
                    metrics = metrics_to_float(metrics)
                    wandb.log(metrics)
                all_metrics.append(metrics)

        if self.checkpoint_logdir:
            # Save current policy.
            path = f'{self.checkpoint_logdir}_{self.agent_name}_{current_step}.pkl'
            save_params(path, agent_state)
            path = f'{self.checkpoint_logdir}_{self.agent_name}_{current_step}.pkl'
            save_params(path, best_state)

        if self.return_best_agent:
            final_agent_state = best_state
        else:
            final_agent_state = agent_state

        # If there was no mistakes the training_state should still be identical on all
        # devices.
        # logging.info('total steps: %s', total_steps)
        if self.wandb_logging:
            wandb.log(metrics_to_float({'total steps': int(agent_state.env_steps)}))
        return final_agent_state, all_metrics


if __name__ == '__main__':
    from bsm.statistical_model.bnn_statistical_model import BNNStatisticalModel
    from brax.envs.simple_pendulum import Pendulum
    from mbpo.optimizers import SACOptimizer
    from distrax import Normal

    env = Pendulum()


    class PendulumReward(Reward):
        def __init__(self):
            super().__init__(x_dim=2, u_dim=1)

        def __call__(self,
                     x: chex.Array,
                     u: chex.Array,
                     reward_params: RewardParams,
                     x_next: chex.Array | None = None
                     ):
            assert x.shape == (2,) and u.shape == (1,)
            u = 4 * u
            cost = jnp.sum(x ** 2) + jnp.sum(u ** 2)
            reward = -cost * reward_params['dt']
            reward_dist = Normal(reward, jnp.zeros_like(reward))
            return reward_dist, reward_params

        def init_params(self, key: chex.PRNGKey) -> RewardParams:
            return {'dt': env.dt}


    horizon = 200
    model = BNNStatisticalModel(
        input_dim=env.observation_size + env.action_size,
        output_dim=env.observation_size,
        num_training_steps=2000,
        output_stds=jnp.ones(env.observation_size),
        features=(64, 64, 64),
        num_particles=5,
        logging_wandb=False,
    )
    sac_kwargs = {
        'num_timesteps': 20_000,
        'episode_length': horizon,
    }
    max_replay_size_true_data_buffer = 10 ** 4
    dummy_sample = Transition(observation=jnp.ones(env.observation_size),
                              action=jnp.zeros(shape=(env.action_size,)),
                              reward=jnp.array(0.0),
                              discount=jnp.array(0.99),
                              next_observation=jnp.ones(env.observation_size))
    sac_buffer = UniformSamplingQueue(
        max_replay_size=max_replay_size_true_data_buffer,
        dummy_data_sample=dummy_sample,
        sample_batch_size=1)
    optimizer = SACOptimizer(system=None, true_buffer=sac_buffer, **sac_kwargs)

    agent = PetsModelBasedAgent(
        env=env,
        statistical_model=model,
        optimizer=optimizer,
        reward_model=PendulumReward(),
        episode_length=horizon,
        num_envs=32,
        env_steps_per_update=horizon,
    )
    agent.train(num_timesteps=10_000)
