import chex
import jax.numpy as jnp
import jax.random as jr
import wandb
from brax.training.replay_buffers import UniformSamplingQueue
from brax.training.types import Transition
from jax.nn import swish
from mbpo.optimizers.base_optimizer import BaseOptimizer
from mbpo.systems.rewards.base_rewards import Reward, RewardParams
from mbpo.utils.type_aliases import OptimizerState

from mbrl.model_based_agent.optimizer_wrapper import Actor, PetsActor
from mbrl.model_based_agent.system_wrapper import WtsScPetsDynamics, WtcScPetsSystem
from mbrl.model_based_agent.wtc_base import WtcBaseModelBasedAgent


@chex.dataclass
class ModelBasedAgentState:
    optimizer_state: OptimizerState
    env_steps: chex.Array
    key: chex.Array


class WtcPets(WtcBaseModelBasedAgent):

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_actor(self,
                      optimizer: BaseOptimizer,
                      ) -> Actor:
        return self.prepare_wtc_actor(optimizer=optimizer,
                                      dynamics=WtsScPetsDynamics,
                                      system=WtcScPetsSystem,
                                      actor=PetsActor)


if __name__ == "__main__":
    from mbrl.envs.pendulum import PendulumEnv
    from bsm.statistical_model.bnn_statistical_model import BNNStatisticalModel
    from mbpo.optimizers import SACOptimizer
    from distrax import Normal
    from mbrl.utils.offline_data import WhenToControlWrapper
    from wtc.wrappers.ih_switching_cost import IHSwitchCostWrapper, ConstantSwitchCost
    from wtc.utils import discrete_to_continuous_discounting
    from bsm.bayesian_regression import DeterministicEnsemble

    log_wandb = True
    ENTITY = 'trevenl'

    base_env = PendulumEnv(reward_source='dm-control')

    min_time_between_switches = 1 * base_env.dt
    max_time_between_switches = 30 * base_env.dt
    num_integrator_steps = 100
    switch_cost = 0.1

    env = IHSwitchCostWrapper(base_env,
                              num_integrator_steps=num_integrator_steps,
                              min_time_between_switches=min_time_between_switches,
                              max_time_between_switches=max_time_between_switches,
                              switch_cost=ConstantSwitchCost(value=jnp.array(0.0)),
                              time_as_part_of_state=True)

    episode_time = base_env.dt * num_integrator_steps


    class TransitionReward(Reward):
        def __init__(self):
            super().__init__(x_dim=2, u_dim=1)

        def __call__(self,
                     x: chex.Array,
                     u: chex.Array,
                     reward_params: RewardParams,
                     x_next: chex.Array | None = None
                     ):
            reward = jnp.array(-switch_cost)
            reward_dist = Normal(reward, jnp.zeros_like(reward))
            return reward_dist, reward_params

        def init_params(self, key: chex.PRNGKey) -> RewardParams:
            return {'dt': 0.05}


    offline_data_gen = WhenToControlWrapper(
        num_integrator_steps=num_integrator_steps,
        min_time_between_switches=min_time_between_switches,
        max_time_between_switches=max_time_between_switches
    )
    key = jr.PRNGKey(0)

    offline_data = offline_data_gen.sample_transitions(key=key,
                                                       num_samples=100)
    running_reward_max_bound = 20.0
    running_reward_min_bound = -5
    horizon = 100
    model = BNNStatisticalModel(
        input_dim=env.observation_size + env.action_size - 1,  # -1 since we don't input env_time
        output_dim=env.observation_size + 1 - 1,  # +1 for the reward -1 for env time
        num_training_steps=30_000,
        output_stds=1e-3 * jnp.ones(env.observation_size + 1 - 1),  # +1 for the reward -1 for env_time
        beta=2.0 * jnp.ones(shape=(env.observation_size + 1 - 1,)),
        features=(64,) * 3,
        bnn_type=DeterministicEnsemble,
        num_particles=5,
        logging_wandb=log_wandb,
        return_best_model=True,
        eval_batch_size=64,
        train_share=0.8,
        eval_frequency=5_000,
        weight_decay=0.0,
    )

    discount_factor = 0.99
    continuous_discounting = discrete_to_continuous_discounting(discrete_discounting=discount_factor,
                                                                dt=env.dt)
    sac_kwargs = {
        'num_timesteps': 100_000,
        'episode_length': 100,
        'num_env_steps_between_updates': 10,
        'num_envs': 64,
        'num_eval_envs': 4,
        'lr_alpha': 3e-4,
        'lr_policy': 3e-4,
        'lr_q': 3e-4,
        'wd_alpha': 0.,
        'wd_policy': 0.,
        'wd_q': 0.,
        'max_grad_norm': 1e5,
        'discounting': 0.99,
        'batch_size': 64,
        'num_evals': 20,
        'normalize_observations': True,
        'reward_scaling': 1.,
        'tau': 0.005,
        'min_replay_size': 10 ** 3,
        'max_replay_size': 10 ** 5,
        'grad_updates_per_step': 10 * 64,  # should be num_envs * num_env_steps_between_updates
        'deterministic_eval': True,
        'init_log_alpha': 0.,
        'policy_hidden_layer_sizes': (32,) * 5,
        'policy_activation': swish,
        'critic_hidden_layer_sizes': (128,) * 3,
        'critic_activation': swish,
        'wandb_logging': log_wandb,
        'return_best_model': True,
        'non_equidistant_time': True,
        'continuous_discounting': continuous_discounting,
        'min_time_between_switches': min_time_between_switches,
        'max_time_between_switches': max_time_between_switches,
        'env_dt': env.dt,
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
    if log_wandb:
        wandb.init(project="ModelBasedTest",
                   dir='/cluster/scratch/' + ENTITY,
                   )

    agent = WtcPets(
        env=env,
        eval_env=env,
        statistical_model=model,
        optimizer=optimizer,
        reward_model=TransitionReward(),
        episode_length=horizon,
        offline_data=offline_data,
        num_envs=1,
        num_eval_envs=1,
        log_to_wandb=log_wandb,
        dt=env.dt,
        min_time_between_switches=min_time_between_switches,
        max_time_between_switches=max_time_between_switches,
        episode_time=episode_time,
        running_reward_max_bound=running_reward_max_bound,
        running_reward_min_bound=running_reward_min_bound
    )

    agent_state = agent.run_episodes(num_episodes=20,
                                     start_from_scratch=True,
                                     key=jr.PRNGKey(0))

    wandb.finish()
