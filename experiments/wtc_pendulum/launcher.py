import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

general_configs = {
    'project_name': ['MBWTC_Apr25_16_30'],
    'num_offline_samples': [0, 100, 200, 500, 1000, 5000],
    'sac_horizon': [32, 64, 100, 128, ],
    'deterministic_policy_for_data_collection': [0, 1],
    'seed': list(range(5)),
    'num_episodes': [20, ],
    'sac_steps': [100_000, 1_000_000, ]
}


# general_configs = {
#     'project_name': ['test-project'],
#     'num_offline_samples': [500,],
#     'sac_horizon': [100, ],
#     'deterministic_policy_for_data_collection': [0,],
#     'seed': list(range(1)),
#     'num_episodes': [5, ]
# }


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
