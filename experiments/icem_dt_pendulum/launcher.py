import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

general_configs = {
    'project_name': ['ICEM_Pendulum_Debug'],
    'num_offline_samples': [1_000],
    'noise_level': ['0.05_0.1'],
    'seed': [42],
    'num_episodes': [1],
    'exploration': ['pets'],
    'reset_statistical_model': [1],
    'regression_model': ['probabilistic_ensemble'],
    'state_data_source': ['discrete'],
    'measurement_dt_ratio': [1, 2, 3, 4, 6, 8, 12],
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
