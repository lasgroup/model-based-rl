import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

PROJECT_NAME = 'DT_Mountaincar_Feb24_17_50_GP_Anpassen'
ENTITY = 'kiten'

general_configs = {
    'seed': list(range(5)),
    'project_name': [PROJECT_NAME],
    'entity': [ENTITY],
    'optimizer': ['icem'],
    'num_offline_samples': [0],
    'num_online_samples': [200],
    'action_repeat': [4],
    'deterministic_policy_for_data_collection': [0],
    'reward_source': ['gym'],
    'num_episodes': [20],
    'bnn_steps': [15_000],
    'predict_difference': [1],
    'first_episode_for_policy_training': [0],
    'exploration': ['optimistic'],
    'reset_statistical_model': [0],
    'regression_model': ['GP'],
    'beta': [0.],
    'weight_decay': [0.0],
    'int_rew_weight_init': [0, 1.0, 10.0],
    'int_rew_weight_end': [0.0],
    'rew_decrease_steps': [-1],
    'save_trajectory_transitions': [1],
}

print("WARNING: Random initialization of env state disabled")
print("WARNING: Optimism in iCEM disabled (DEBUGGING)")

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
        'optimizer_horizon': [25,50],
        'icem_num_steps': [5],
        'icem_colored_noise_exponent': [1.0],
        'icem_num_particles': [1],
        'icem_num_samples': [500],
        'icem_num_elites': [100],
        'icem_alpha': [0.2],
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
                          duration='3:59:00',
                          prompt=True,
                          mem=16000)


if __name__ == '__main__':
    main()
