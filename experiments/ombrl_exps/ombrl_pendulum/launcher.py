import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

ENTITY = 'trevenl'

general_configs = {
    'project_name': ['OMBRL_Pendulum_Nov13'],
    'num_offline_samples': [0, ],
    'icem_horizon': [20, ],
    'num_particles': [10, ],
    'num_samples': [500, ],
    'num_elites': [50, ],
    'init_std': [1.0],
    'num_steps': [5, ],
    'exponent': [1.0],
    'seed': list(range(5)),
    'num_episodes': [50, ],
    'min_bnn_steps': [5_000],
    'max_bnn_steps': [50_000],
    'linear_scheduler_steps': [20_000],
    'exploration': ['optimistic', 'pets', 'hucrl'],
    'reset_statistical_model': [0, 1],
    'regression_model': ['FSVGD'],
    'exploration_factor': [1.0, 2.0],
    'deterministic_policy_for_data_collection': [0, 1],
    'horizon': [200],
    'log_wandb': [1],
    'entity': [ENTITY, ],
    'int_reward_weight': [1.0, 2.0, 5.0]
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
