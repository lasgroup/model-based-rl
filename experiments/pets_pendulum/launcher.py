import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations, available_gpus

PROJECT_NAME = 'StandardPetsNov15-v2'

general_configs = {
    'project_name': [PROJECT_NAME],
    'num_offline_samples': [0, ],
    'deterministic_policy_for_data_collection': [1],
    'seed': list(range(5)),
    'icem_horizon': [20, ],
    'reset_statistical_model': [0],
    'bnn_train_steps': [20_000, 50_000],
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
