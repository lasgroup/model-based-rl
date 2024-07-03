import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

general_configs = {
    'project_name': ['CT_PendulumGeneral_June17_18_00'],
    'num_offline_samples': [0],
    'sac_horizon': [64,100,200],
    'deterministic_policy_for_data_collection': [0, 1],
    'seed': list(range(5)),
    'num_episodes': [30],
    'sac_steps': [500_000, 1_000_000],
    'bnn_steps': [50_000],
    'first_episode_for_policy_training': [0],
    'exploration': ['pets'],
    'reset_statistical_model': [1],
    'regression_model': ['deterministic_ensemble'],
    'beta': [2.0],
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
                          duration='11:59:00',
                          prompt=True,
                          mem=16000)


if __name__ == '__main__':
    main()
