import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

general_configs = {
    'project_name': ['WTC_RCCar_May19_11_43'],
    'num_offline_samples': [0],
    'sac_horizon': [100],
    'deterministic_policy_for_data_collection': [0],
    'seed': list(range(5)),
    'num_episodes': [100],
    'sac_steps': [500_000],
    'min_bnn_steps': [5_000],
    'max_bnn_steps': [50_000],
    'linear_scheduler_steps': [20_000],
    'first_episode_for_policy_training': [0],
    'exploration': ['optimistic', 'pets', 'mean'],
    'reset_statistical_model': [1],
    'regression_model': ['FSVGD'],
    'max_time_factor': [5],
    'beta_factor': [2.0],
    'horizon': [100],
    'transition_cost': [0.2]
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
