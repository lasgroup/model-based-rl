import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

general_configs = {
    'project_name': ['GenearalMay10_10_20'],
    'num_offline_samples': [0],
    'sac_horizon': [16, 32, 64],
    'deterministic_policy_for_data_collection': [1],
    'seed': list(range(5)),
    'sac_steps': [500_000, ],
    'bnn_steps': [40_000],
    'first_episode_for_policy_training': [5, ],
    'exploration': ['optimistic', 'pets'],
    'reset_statistical_model': [1],
    'regression_model': ['FSVGD', 'probabilistic_ensemble'],
}
general_configs_reacher = {'env_name': ['Reacher'], 'num_episodes': [50]} | general_configs
general_configs_rccar = {'env_name': ['RCCar'], 'num_episodes': [50]} | general_configs
# general_configs_pendulum = {'env_name': ['Pendulum'], 'num_episodes': [20]} | general_configs
# general_configs_greenhouse = {'env_name': ['Greenhouse'], 'num_episodes': [20]} | general_configs

# flags_combinations = dict_permutations(general_configs_reacher) + dict_permutations(general_configs_rccar) \
#                      + dict_permutations(general_configs_pendulum) + dict_permutations(general_configs_greenhouse)

flags_combinations = dict_permutations(general_configs_reacher)
flags_combinations += dict_permutations(general_configs_rccar)


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
                          mem=32000)


if __name__ == '__main__':
    main()
