import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations, available_gpus

PROJECT_NAME = 'Model_based_pets_March13_10_20'

general_configs = {
    'project_name': [PROJECT_NAME],
    'num_offline_samples': [0, 100, 200, 1000],
    'sac_horizon': [32, 64, 128, ],
    'deterministic_policy_for_data_collection': [0, 1],
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
                          duration='3:59:00',
                          prompt=True,
                          mem=16000)


if __name__ == '__main__':
    main()
