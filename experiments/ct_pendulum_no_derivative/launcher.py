import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

general_configs = {
    'project_name': ['CT_Pendulum_Debug'],
    'num_offline_samples': [0],
    'sac_horizon': [100],
    'deterministic_policy_for_data_collection': [1],
    'seed': [42],
    'num_episodes': [30],
    'sac_steps': [800_000],
    'bnn_steps': [48_000],
    'bnn_features': ['256, 256','64, 64, 64', '128, 128, 128', '256, 256, 256'],
    'first_episode_for_policy_training': [1],
    'exploration': ['pets'],
    'reset_statistical_model': [0],
    'regression_model': ['probabilistic_ensemble'],
    'beta': [2.0],
    'bnn_weight_decay': [0.0],
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
