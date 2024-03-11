from abc import ABC
from functools import partial

import chex
import jax.numpy as jnp
import jax.random as jr
import wandb
from brax.envs import Env as BraxEnv
from brax.training.replay_buffers import UniformSamplingQueue, ReplayBufferState
from brax.training.types import Transition
from bsm.statistical_model import StatisticalModel
from bsm.utils import StatisticalModelState
from bsm.utils.normalization import Data
from jax import jit
from jax.nn import swish
from mbpo.optimizers.base_optimizer import BaseOptimizer
from mbpo.systems.base_systems import System
from mbpo.systems.rewards.base_rewards import Reward, RewardParams
from mbpo.utils.type_aliases import OptimizerState

from mbrl.model_based_agent.system_wrapper import LearnedModelSystem, LearnedDynamics
from mbrl.utils.brax_utils import EnvInteractor


@chex.dataclass
class ModelBasedAgentState:
    optimizer_state: OptimizerState
    env_steps: chex.Array
    key: chex.Array


class PetsModelBasedAgent(ABC):

    def __init__(self,
                 env: BraxEnv,
                 statistical_model: StatisticalModel,
                 reward_model: Reward,
                 optimizer: BaseOptimizer,
                 episode_length: int,
                 eval_env: BraxEnv,
                 num_envs: int = 1,
                 num_eval_envs: int = 1,
                 action_repeat: int = 1,
                 offline_data: Transition | None = None,
                 max_collected_data_in_buffer: int = 10 ** 4,
                 max_episodes: int = 100,
                 predict_difference: bool = True,
                 reset_statistical_model: bool = True,
                 key: chex.PRNGKey = jr.PRNGKey(0),
                 log_to_wandb: bool = False,
                 ):
        self.env = env
        self.statistical_model = statistical_model
        self.reward_model = reward_model
        self.optimizer = optimizer
        self.episode_length = episode_length
        self.eval_env = eval_env
        self.num_envs = num_envs
        self.num_eval_envs = num_eval_envs
        self.action_repeat = action_repeat
        self.offline_data = offline_data
        self.max_collected_data_in_buffer = max_collected_data_in_buffer
        self.max_episodes = max_episodes
        self.predict_difference = predict_difference
        self.reset_statistical_model = reset_statistical_model
        self.key = key
        self.log_to_wandb = log_to_wandb

        self.key, subkey = jr.split(self.key)
        self.env_interactor = EnvInteractor(env=self.env,
                                            eval_env=self.eval_env,
                                            num_envs=self.num_envs,
                                            num_eval_envs=self.num_eval_envs,
                                            episode_length=self.episode_length,
                                            action_repeat=self.action_repeat,
                                            key=subkey)

        self.collected_data_buffer = self.prepare_data_buffers()
        self.learned_system = self.prepare_learned_system()
        self.optimizer.set_system(system=self.learned_system)

    def prepare_data_buffers(self) -> UniformSamplingQueue:
        dummy_sample = Transition(observation=jnp.zeros(shape=(self.env.observation_size,)),
                                  action=jnp.zeros(shape=(self.env.action_size,)),
                                  reward=jnp.array(0.0),
                                  discount=jnp.array(0.99),
                                  next_observation=jnp.zeros(shape=(self.env.observation_size,)),
                                  )

        collected_data_buffer = UniformSamplingQueue(
            max_replay_size=self.max_collected_data_in_buffer,
            dummy_data_sample=dummy_sample,
            sample_batch_size=1)

        return collected_data_buffer

    def init(self, key: chex.Array):
        key_state, key_data_buffers, key_optimizer = jr.split(key, 3)
        collected_data_buffer_state = self._init_data_buffer_states(key_data_buffers)
        init_optimizer_state = self.optimizer.init(key=key_optimizer,
                                                   true_buffer_state=collected_data_buffer_state)
        return ModelBasedAgentState(optimizer_state=init_optimizer_state,
                                    env_steps=0,
                                    key=key_state)

    def prepare_learned_system(self) -> System:
        learned_dynamics = LearnedDynamics(statistical_model=self.statistical_model,
                                           x_dim=self.env.observation_size,
                                           u_dim=self.env.action_size)
        system = LearnedModelSystem(dynamics=learned_dynamics,
                                    reward=self.reward_model, )
        return system

    def train_policy(self,
                     agent_state: ModelBasedAgentState,
                     episode_idx: int) -> ModelBasedAgentState:
        optimizer_state = agent_state.optimizer_state
        # TODO: here we always start training the optimizer from scratch, we might want to just continue training after
        #  the data buffer surpasses certain margin
        optimizer_output = self.optimizer.train(opt_state=optimizer_state)
        new_agent_state = agent_state.replace(optimizer_state=optimizer_output.optimizer_state)
        return new_agent_state

    def train_dynamics_model(self,
                             agent_state: ModelBasedAgentState,
                             episode_idx: int, ) -> ModelBasedAgentState:
        statistical_model_state = agent_state.optimizer_state.system_params.dynamics_params.statistical_model_state
        collected_data_buffer_state = agent_state.optimizer_state.true_buffer_state
        key, key_sms = jr.split(agent_state.key)

        statistical_model_state = self._update_statistical_model(
            statistical_model_state=statistical_model_state,
            collected_data_buffer_state=collected_data_buffer_state,
            key=key_sms)

        new_dynamics_params = agent_state.optimizer_state.system_params.dynamics_params.replace(
            statistical_model_state=statistical_model_state)
        new_system_params = agent_state.optimizer_state.system_params.replace(
            dynamics_params=new_dynamics_params)
        new_optimizer_state = agent_state.optimizer_state.replace(system_params=new_system_params)
        new_agent_state = agent_state.replace(optimizer_state=new_optimizer_state,
                                              key=key)
        return new_agent_state

    @partial(jit, static_argnums=0)
    def simulate_on_true_env(self,
                             agent_state: ModelBasedAgentState,
                             ) -> ModelBasedAgentState:
        key_agent, key_reset = jr.split(agent_state.key)
        env_state = self.env_interactor.reset(key=key_reset)
        optimizer_state = agent_state.optimizer_state
        interaction = self.env_interactor.generate_rollouts(env_state=env_state,
                                                            optimizer_state=optimizer_state,
                                                            optimizer=self.optimizer
                                                            )
        final_state, optimizer_state, transitions = interaction
        collected_data_buffer_state = agent_state.optimizer_state.true_buffer_state
        collected_data_buffer_state = self.collected_data_buffer.insert(buffer_state=collected_data_buffer_state,
                                                                        samples=transitions)
        optimizer_state = optimizer_state.replace(true_buffer_state=collected_data_buffer_state)
        env_steps = agent_state.env_steps + self.num_envs * self.episode_length
        return ModelBasedAgentState(optimizer_state=optimizer_state,
                                    env_steps=env_steps,
                                    key=key_agent, )

    def do_episode(self,
                   agent_state: ModelBasedAgentState,
                   episode_idx: int) -> ModelBasedAgentState:
        if episode_idx > 0 or self.offline_data:
            # If we collected some data already then we train dynamics model and the policy
            print(f'Start of dynamics training')
            agent_state = self.train_dynamics_model(agent_state=agent_state,
                                                    episode_idx=episode_idx)
            print(f'End of dynamics training')
            print(f'Start of policy training')
            agent_state = self.train_policy(agent_state=agent_state,
                                            episode_idx=episode_idx)
            print(f'End of policy training')
        # We collect new data with the current policy
        print(f'Start of data collection')
        agent_state = self.simulate_on_true_env(agent_state=agent_state)
        print(f'End of data collection')
        print(f'Start with evaluation of the policy')
        metrics = self.env_interactor.run_evaluation(optimizer=self.optimizer,
                                                     optimizer_state=agent_state.optimizer_state)
        if self.log_to_wandb:
            wandb.log(metrics)
        else:
            print(metrics)
        print(f'End with evaluation of the policy')
        return agent_state

    def run_episodes(self,
                     num_episodes: int,
                     start_from_scratch: bool = True,
                     key: chex.PRNGKey = jr.PRNGKey(0),
                     agent_state: ModelBasedAgentState | None = None) -> ModelBasedAgentState:
        if start_from_scratch:
            # If we start collecting the data and need to initialize the agent state
            agent_state = self.init(key)
        for episode_idx in range(num_episodes):
            print(f'Starting with Episode {episode_idx}')
            agent_state = self.do_episode(agent_state=agent_state,
                                          episode_idx=episode_idx)
            print(f'End of Episode {episode_idx}')
        return agent_state

    def _collected_buffer_to_train_data(self,
                                        collected_buffer_state: ReplayBufferState):
        idx = jnp.arange(start=collected_buffer_state.sample_position, stop=collected_buffer_state.insert_position)
        all_data = jnp.take(collected_buffer_state.data, idx, axis=0, mode='wrap')
        all_transitions = self.collected_data_buffer._unflatten_fn(all_data)
        obs = all_transitions.observation
        actions = all_transitions.action
        inputs = jnp.concatenate([obs, actions], axis=-1)
        next_obs = all_transitions.next_observation
        if self.predict_difference:
            outputs = next_obs - obs
        else:
            outputs = next_obs
        return Data(inputs=inputs, outputs=outputs)

    def _init_data_buffer_states(self,
                                 key: chex.PRNGKey) -> ReplayBufferState:
        key_collected_data_bs, key_init_state_bs = jr.split(key)
        collected_data_buffer_state = self.collected_data_buffer.init(key_collected_data_bs)

        # If we have offline data we insert in the collected data buffer
        if self.offline_data:
            assert self.offline_data.observation.shape[-1] == self.env.observation_size
            assert self.offline_data.action.shape[-1] == self.env.action_size
            collected_data_buffer_state = self.collected_data_buffer.insert(collected_data_buffer_state,
                                                                            self.offline_data)

        return collected_data_buffer_state

    def _update_statistical_model(self,
                                  statistical_model_state: StatisticalModelState,
                                  collected_data_buffer_state: ReplayBufferState,
                                  key: chex.PRNGKey):
        # We prepare data to train from the collected_data_buffer
        data = self._collected_buffer_to_train_data(collected_data_buffer_state)
        if self.reset_statistical_model:
            statistical_model_state = self.statistical_model.init(key=key)
        new_statistical_model_state = self.statistical_model.update(
            stats_model_state=statistical_model_state,
            data=data)
        return new_statistical_model_state


if __name__ == "__main__":
    from mbrl.envs.pendulum import PendulumEnv
    from bsm.statistical_model.bnn_statistical_model import BNNStatisticalModel
    from mbpo.optimizers import SACOptimizer
    from distrax import Normal
    from mbrl.utils.offline_data import PendulumOfflineData

    ENTITY = 'trevenl'

    env = PendulumEnv(reward_source='dm-control')


    class PendulumReward(Reward):
        def __init__(self):
            super().__init__(x_dim=2, u_dim=1)

        def __call__(self,
                     x: chex.Array,
                     u: chex.Array,
                     reward_params: RewardParams,
                     x_next: chex.Array | None = None
                     ):
            assert x.shape == (3,) and u.shape == (1,)
            theta, omega = jnp.arctan2(x[1], x[0]), x[-1]
            target_angle = env.reward_params.target_angle
            diff_th = theta - target_angle
            diff_th = ((diff_th + jnp.pi) % (2 * jnp.pi)) - jnp.pi
            reward = env.tolerance_reward(jnp.sqrt(env.reward_params.angle_cost * diff_th ** 2 +
                                                   0.1 * omega ** 2)) - env.reward_params.control_cost * u ** 2
            reward = reward.squeeze()
            reward_dist = Normal(reward, jnp.zeros_like(reward))
            return reward_dist, reward_params

        def init_params(self, key: chex.PRNGKey) -> RewardParams:
            return {'dt': env.dt}


    offline_data_gen = PendulumOfflineData()
    key = jr.PRNGKey(0)
    offline_data = offline_data_gen.sample_transitions(key=key, num_samples=10_000)

    horizon = 200
    model = BNNStatisticalModel(
        input_dim=env.observation_size + env.action_size,
        output_dim=env.observation_size,
        num_training_steps=50_000,
        output_stds=0.01 * jnp.ones(env.observation_size),
        features=(64, 64, 64),
        num_particles=5,
        logging_wandb=True,
    )

    sac_kwargs = {
        'num_timesteps': 40_000,
        'episode_length': 200,
        'num_env_steps_between_updates': 20,
        'num_envs': 32,
        'num_eval_envs': 4,
        'lr_alpha': 3e-4,
        'lr_policy': 3e-4,
        'lr_q': 3e-4,
        'wd_alpha': 0.,
        'wd_policy': 0.,
        'wd_q': 0.,
        'max_grad_norm': 1e5,
        'discounting': 0.99,
        'batch_size': 32,
        'num_evals': 20,
        'normalize_observations': True,
        'reward_scaling': 1.,
        'tau': 0.005,
        'min_replay_size': 10 ** 4,
        'max_replay_size': 10 ** 5,
        'grad_updates_per_step': 20 * 32,  # should be num_envs * num_env_steps_between_updates
        'deterministic_eval': True,
        'init_log_alpha': 0.,
        'policy_hidden_layer_sizes': (32,) * 5,
        'policy_activation': swish,
        'critic_hidden_layer_sizes': (128,) * 4,
        'critic_activation': swish,
        'wandb_logging': True,
        'return_best_model': True,
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
    optimizer = SACOptimizer(system=None,
                             true_buffer=sac_buffer,
                             **sac_kwargs)

    wandb.init(project="Model-based Agent",
               dir='/cluster/scratch/' + ENTITY,
               )

    agent = PetsModelBasedAgent(
        env=env,
        eval_env=env,
        statistical_model=model,
        optimizer=optimizer,
        reward_model=PendulumReward(),
        episode_length=horizon,
        offline_data=offline_data,
        num_envs=1,
        num_eval_envs=1,
        log_to_wandb=True,
    )

    agent_state = agent.run_episodes(num_episodes=20,
                                     start_from_scratch=True,
                                     key=jr.PRNGKey(0))

    wandb.finish()
