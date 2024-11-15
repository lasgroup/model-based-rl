import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

ENTITY = 'kiten'

general_configs = {
    'project_name': ['OMBRL_Pendulum_Nov14'],
    'num_offline_samples': [0, ],
    'icem_horizon': [20, ],
    'num_particles': [10, ],
    'num_samples': [200, ],
    'num_elites': [50, ],
    'init_std': [0.5],
    'num_steps': [10, ],
    'exponent': [1.0],
    'seed': list(range(5)),
    'num_episodes': [50, ],
    'min_bnn_steps': [15_000],
    'max_bnn_steps': [16_000],
    'linear_scheduler_steps': [20_000],
    'exploration': ['optimistic', 'pets', 'hucrl'],
    'reset_statistical_model': [1],
    'regression_model': ['deterministic_ensemble', 'probabilistic_ensemble'],
    'exploration_factor': [20, ],
    'horizon': [100],
    'log_wandb': [1],
    'entity': [ENTITY, ],
    'int_reward_weight': [0.1, 1.0]
}


def main():
    command_list = []
    flags_combinations = dict_permutations(general_configs)
    for flags in flags_combinations:
        cmd = generate_base_command(exp, flags=flags)
        command_list.append(cmd)

    # submit jobs
    generate_run_commands(command_list,
                          num_cpus=1,
                          num_gpus=1,
                          mode='euler',
                          duration='23:59:00',
                          prompt=True,
                          mem=16000)


if __name__ == '__main__':
    main()
