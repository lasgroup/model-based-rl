import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

general_configs = {
    'project_name': ['PendulumGeneral_SAC_Nov27_22_30'],
    'num_offline_samples': [0],
    'sac_horizon': [100],
    'deterministic_policy_for_data_collection': [0],
    'reward_source': ['dm-control','gym'],
    'seed': list(range(5)),
    'num_episodes': [10],
    'sac_steps': [100_000],
    'bnn_steps': [15_000],
    'first_episode_for_policy_training': [0],
    'exploration': ['optimistic', 'pets'],
    'reset_statistical_model': [0],
    'regression_model': ['probabilistic_ensemble','deterministic_ensemble'],
    'beta': [2.0],
    'weight_decay': [0.0],
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
