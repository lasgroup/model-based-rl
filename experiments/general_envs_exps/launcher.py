import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

general_configs = {
    'project_name': ['General_May01_10_50'],
    'num_offline_samples': [0],
    'sac_horizon': [100],
    'deterministic_policy_for_data_collection': [1],
    'seed': list(range(5)),
    # 'num_episodes': [20],
    'sac_steps': [250_000, 500_000, 1_000_000],
    'bnn_steps': [50_000],
    'first_episode_for_policy_training': [0],
    'exploration': ['optimistic', 'pets'],
    'reset_statistical_model': [1],
    'regression_model': ['FSVGD'],
    # 'env_name': ['Pendulum', 'RCCar', 'Greenhouse', 'Reacher']
}
general_configs_reacher = {'env_name': ['Reacher'], 'num_episodes': [50]} | general_configs
general_configs_rccar = {'RCCar': ['Reacher'], 'num_episodes': [50]} | general_configs
general_configs_pendulum = {'env_name': ['Pendulum'], 'num_episodes': [20]} | general_configs
general_configs_greenhouse = {'env_name': ['Greenhouse'], 'num_episodes': [20]} | general_configs

flags_combinations = dict_permutations(general_configs_reacher) + dict_permutations(general_configs_rccar) \
                     + dict_permutations(general_configs_pendulum) + dict_permutations(general_configs_greenhouse)


def main():
    command_list = []
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
