import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

general_configs = {
    'project_name': ['ICEM_Cartpole_Sweep_0920'],
    'num_offline_samples': [0],
    'optimizer_horizon': [20],
    'icem_num_steps': [10],
    'icem_num_particles': [1, 5, 10],
    'noise_level': ['0.1_0.1'],
    'seed': [17, 42, 69],
    'num_episodes': [20],
    'bnn_steps': [48_000],
    'bnn_features': ['64_64'],
    'bnn_weight_decay': [1e-4],
    'smoother_steps': [64_000],
    'smoother_features': ['64_64_64'],
    'exploration': ['pets', 'optimistic', 'mean'],
    'reset_statistical_model': [1],
    'regression_model': ['probabilistic_ensemble'],
    'beta': [2.0],
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
