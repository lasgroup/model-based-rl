from mbrl.model_based_agent.base_model_based_agent import BaseModelBasedAgent
from mbpo.optimizers.base_optimizer import BaseOptimizer
from mbrl.model_based_agent.optimizer_wrapper import Actor
from typing import Type

class BaseAgentWrapper(BaseModelBasedAgent):
    def __init__(self, agent_type: Type[BaseModelBasedAgent],
                 **kwargs):
        # Get the args and kwargs from the agent
        self.agent = agent_type(**kwargs)
        super().__init__(**kwargs)

    # Only change the prepare_actor method (since this is what varies between the different agents)
    def prepare_actor(self, optimizer: BaseOptimizer) -> Actor:
        return self.agent.prepare_actor(optimizer) 