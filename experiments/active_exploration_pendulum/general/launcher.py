import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

PROJECT_NAME = 'Active_DT_Pendulum_Dec12_15_00_TestOptimisticMeanPets'
ENTITY = 'kiten'

general_configs = {
    'seed': list(range(1)),
    'project_name': [PROJECT_NAME],
    'entity': [ENTITY],
    'num_offline_samples': [0],
    'num_online_samples': [100],
    'deterministic_policy_for_data_collection': [0],
    'reward_source': ['gym'],
    'num_episodes': [10],
    'bnn_steps': [50_000],
    'predict_difference': [1],
    'first_episode_for_policy_training': [0],
    'exploration': ['optimistic','mean','pets'],
    'reset_statistical_model': [0],
    'regression_model': ['probabilistic_ensemble'],
    'beta': [2.0],
    'env': ['balance'],
}

sac_configs = {
    'optimizer': ['sac'],
    'train_steps_sac': [100_000],
} | general_configs

icem_configs = {
    'optimizer': ['icem'],
    'optimizer_horizon': [20],
    'icem_num_steps': [10],
    'icem_colored_noise_exponent': [1.0],
} | general_configs


def main():
    command_list = []
    flags_combinations = dict_permutations(sac_configs) + dict_permutations(icem_configs)
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
