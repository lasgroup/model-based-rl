import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

PROJECT_NAME = 'Active_DT_Pendulum_Dec31_19_55_Test_Mean_GP'
ENTITY = 'kiten'

general_configs = {
    'seed': list(range(3)),
    'project_name': [PROJECT_NAME],
    'entity': [ENTITY],
    'optimizer': ['icem'],
    'num_offline_samples': [0],
    'num_online_samples': [200],
    'deterministic_policy_for_data_collection': [0],
    'reward_source': ['gym'],
    'num_episodes': [10],
    'bnn_steps': [50_000],
    'predict_difference': [1],
    'first_episode_for_policy_training': [0],
    'exploration': ['optimistic','mean','pets'],
    'reset_statistical_model': [0],
    'regression_model': ['GP'],
    'beta': [2.0],
    'env': ['balance'],
    # 'eval_envs': [['swing-up','balance']], # bugged on EULER
}

sac_configs = (
    {
        **general_configs,
        'optimizer': ['sac'],
        'train_steps_sac': [100_000],
    }
    if 'sac' in general_configs['optimizer']
    else None
)

icem_configs = (
    {
        **general_configs,
        'optimizer': ['icem'],
        'optimizer_horizon': [20],
        'icem_num_steps': [10],
        'icem_colored_noise_exponent': [1.0],
    }
    if 'icem' in general_configs['optimizer']
    else None
)


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
                          duration='23:59:00',
                          prompt=True,
                          mem=16000)


if __name__ == '__main__':
    main()
