import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

general_configs = {
    'project_name': ['WTC_RTCCar_May08_14_20'],
    'num_offline_samples': [0, ],
    'sac_horizon': [64, 100,],
    'deterministic_policy_for_data_collection': [1, ],
    'seed': list(range(5)),
    'num_episodes': [50],
    'sac_steps': [1_000_000, ],
    'bnn_steps': [50_000],
    'first_episode_for_policy_training': [0, ],
    'exploration': ['optimistic', 'pets'],
    'reset_statistical_model': [1],
    'regression_model': ['FSVGD'],
    'max_time_factor': [5, 10, ],
    'beta_factor': [1.0, 2.0],
    'horizon': [100],
    'transition_cost': [0.01, 0.5]
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
                          mem=64000)


if __name__ == '__main__':
    main()
