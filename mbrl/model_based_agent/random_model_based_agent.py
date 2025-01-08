from .base_model_based_agent import BaseModelBasedAgent
from .continuous_base_model_based_agent import ContinuousBaseModelBasedAgent
from mbpo.optimizers.base_optimizer import BaseOptimizer
from mbrl.model_based_agent.optimizer_wrapper import Actor,  RandomActor
from mbrl.model_based_agent.system_wrapper import MeanSystem, MeanDynamics, ContinuousMeanSystem, ContinuousMeanDynamics

class RandomModelBasedAgent(BaseModelBasedAgent):
    def __init__(self, *args, **kwarsg):
        super().__init__(*args, **kwarsg)

    def prepare_actor(self,
                      optimizer: BaseOptimizer,
                      ) -> Actor:
        dynamics, system, actor = MeanDynamics, MeanSystem, RandomActor
        dynamics = dynamics(statistical_model=self.statistical_model,
                            x_dim=self.env.observation_size,
                            u_dim=self.env.action_size,
                            predict_difference=self.predict_difference)
        system = system(dynamics=dynamics,
                        reward=self.reward_model, )
        actor = actor(env=self.env,
                      env_observation_size=self.env.observation_size,
                      env_action_size=self.env.action_size,
                      optimizer=optimizer)
        actor.set_system(system=system)
        return actor

class ContinuousRandomModelBasedAgent(ContinuousBaseModelBasedAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_actor(self,
                      optimizer: BaseOptimizer,
                      ) -> Actor:
        dynamics, system, actor = ContinuousMeanDynamics, ContinuousMeanSystem, RandomActor
        dynamics = dynamics(statistical_model=self.statistical_model,
                            x_dim=self.env.observation_size,
                            u_dim=self.env.action_size,
                            predict_difference=self.predict_difference,
                            dt=self.dt) # self.dynamics_dt
        system = system(dynamics=dynamics,
                        reward=self.reward_model, )
        actor = actor(env=self.env,
                      env_observation_size=self.env.observation_size,
                      env_action_size=self.env.action_size,
                      optimizer=optimizer)
        actor.set_system(system=system)
        return actor