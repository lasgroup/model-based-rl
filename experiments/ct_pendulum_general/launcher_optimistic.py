import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

general_configs = {
    'project_name': ['CT_PendulumGeneral_June13_15_00'],
    'num_offline_samples': [0],
    'sac_horizon': [64,100,200],
    'deterministic_policy_for_data_collection': [0, 1],
    'seed': list(range(5)),
    'num_episodes': [30],
    'sac_steps': [1_000_000],
    'bnn_steps': [50_000],
    'first_episode_for_policy_training': [0],
    'exploration': ['optimistic'],
    'reset_statistical_model': [1],
    'regression_model': ['FSVGD'],
    'beta': [0.5, 1.0, 2.0, 5.0],
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
