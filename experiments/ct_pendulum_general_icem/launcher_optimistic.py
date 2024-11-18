import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

general_configs = {
    'project_name': ['iCEM_CT_PendulumGeneral_Oct23_16_00'],
    'num_offline_samples': [0],
    'optimizer_horizon': [20],
    'num_online_samples': [200],
    'deterministic_policy_for_data_collection': [0,1],
    'icem_num_steps': [20],
    'icem_colored_noise_exponent': [1.0],
    'reward_source': ['dm-control'],
    'seed': list(range(5)),
    'num_episodes': [20],
    'bnn_steps': [15_000],
    'first_episode_for_policy_training': [0],
    'exploration': ['optimistic'],
    'reset_statistical_model': [0,1],
    'regression_model': ['probabilistic_ensemble', 'deterministic_ensemble'],
    'beta': [0.5, 1.0, 2.0, 3.0],
    'weight_decay': [0.0, 1e-4],
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