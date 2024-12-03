import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

general_configs = {
    'project_name': ['CT_PendulumGeneral_iCEM_Dec03_16_45_PermuteEvalEnvs'],
    'num_offline_samples': [0],
    'optimizer_horizon': [20],
    'num_online_samples': [200],
    'deterministic_policy_for_data_collection': [0],
    'icem_num_steps': [10],
    'icem_colored_noise_exponent': [1.0],
    'reward_source': ['dm-control'],
    'control_cost': [0.02],
    'seed': list(range(5)),
    'num_episodes': [10],
    'bnn_steps': [15_000],
    'first_episode_for_policy_training': [0],
    'exploration': ['optimistic', 'pets'],
    'reset_statistical_model': [0],
    'regression_model': ['probabilistic_ensemble'],
    'beta': [2.0],
    'weight_decay': [0.0],
    'env': ['swing-up','balance'],
    'eval_env': ['swing-up','balance'],
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
