import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

PROJECT_NAME = 'PendulumGeneral_iCEM_Dec03_15_00_PermuteEvalEnvs'
ENTITY = 'kiten'

general_configs = {
    'seed': list(range(1)),
    'project_name': [PROJECT_NAME],
    'entity': [ENTITY],
    'optimizer': ['icem'],
    'num_offline_samples': [0],
    'num_online_samples': [200],
    'deterministic_policy_for_data_collection': [0],
    'reward_source': ['gym'],
    'num_episodes': [10],
    'bnn_steps': [15_000],
    'predict_difference': [1],
    'first_episode_for_policy_training': [0],
    'exploration': ['optimistic','mean','pets'],
    'reset_statistical_model': [0],
    'regression_model': ['probabilistic_ensemble'],
    'beta': [2.0],
    'weight_decay': [0.0],
    'env': ['swing-up','balance'],
    'eval_env': ['swing-up','balance'],
}

sac_configs = {
    'train_steps_sac': [100_000],
} | general_configs if 'sac' in general_configs['optimizer'] else None

icem_configs = {
    'optimizer_horizon': [20],
    'icem_num_steps': [10],
    'icem_colored_noise_exponent': [1.0],
} | general_configs if 'icem' in general_configs['optimizer'] else None


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
