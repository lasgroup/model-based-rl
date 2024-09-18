import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

general_configs = {
    'project_name': ['ICEM_RCCar_Perf_Analysis4'],
    'num_offline_samples': [800],
    'icem_colored_noise_exponent': [1.0],
    'seed': [17, 42, 69],
    'num_episodes': [30],
    'bnn_steps': [32_000],
    'bnn_features': ['64_64'],
    'bnn_use_schedule': [False],
    'smoother_steps': [64_000],
    'smoother_features': ['64_64_64'],
    'exploration': ['pets', 'optimistic', 'mean'],
    'reset_statistical_model': [0],
    'state_data_source': ['true'],
    'regression_model': ['probabilistic_ensemble'],
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
