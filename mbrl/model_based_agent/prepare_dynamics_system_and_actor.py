from mbpo.systems.base_systems import System
from mbpo.systems.dynamics.base_dynamics import Dynamics

from mbrl.model_based_agent.optimizer_wrapper import Actor, OptimisticActor, PetsActor
from mbrl.model_based_agent.system_wrapper import OptimisticSystem, OptimisticDynamics
from mbrl.model_based_agent.system_wrapper import PetsSystem, PetsDynamics


def prepare_dynamics_system_actor(learning_style: str) -> (Dynamics, System, Actor):
    if learning_style == "Pets":
        return PetsDynamics, PetsSystem, PetsActor
    elif learning_style == "Optimistic":
        return OptimisticDynamics, OptimisticSystem, OptimisticActor
    else:
        raise ValueError(f"Unknown model name: {learning_style}, available options are Pets, Optimistic")
