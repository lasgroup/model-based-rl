import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

base_config = {
    'project_name': ['WTC_RCCar_Sep_08_16_15'],
    'num_offline_samples': [0],
    'sac_horizon': [100],
    'deterministic_policy_for_data_collection': [1],
    'seed': list(range(5)),
    'num_episodes': [100],
    'sac_steps': [500_000],
    'min_bnn_steps': [5_000],
    'max_bnn_steps': [50_000],
    'linear_scheduler_steps': [20_000],
    'first_episode_for_policy_training': [2],
    'reset_statistical_model': [1],
    'regression_model': ['FSVGD'],
    'max_time_factor': [1, 5],
    'horizon': [100],
    'transition_cost': [0.2],
    'use_log': [0],
    'scale_with_aleatoric_std': [0],
}

optimistic_config = {
    'exploration': ['optimistic'],
    'beta_factor': [2.0],
} | base_config

pets_config = {
    'exploration': ['pets'],
    'beta_factor': [2.0],
} | base_config

mean_config = {
    'exploration': ['mean'],
} | base_config

combrl_config = {
    'exploration': ['combrl'],
    'use_square': [0, 1],
    'int_rew_weight_init': [0.1, 1.0, 10, 100],
    'int_rew_weight_end': [0.0],
    'rew_decrease_steps': [100],
} | base_config


def main():
    command_list = []
    flags_combinations = dict_permutations(combrl_config) \
        + dict_permutations(optimistic_config) \
        + dict_permutations(mean_config) \
        + dict_permutations(pets_config)

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
