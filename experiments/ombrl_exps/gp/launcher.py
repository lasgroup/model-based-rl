import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

ENTITY = 'sukhijab'

general_configs = {
    'project_name': ['OMBRL_Nov26'],
    'seed': list(range(5)),
    'num_episodes': [20, ],
    'reset_statistical_model': [0, 1],
    'horizon': [200],
    'log_wandb': [1],
    'entity': [ENTITY, ],
    'calibration': [1],
    'reward_source': ['dm-control'],
    'action_repeat': [5, ],
    'icem_horizon': [5, ]
}

pets_config = {
                  'exploration': ['pets'],
                  'int_reward_weight': [0.0],
                  'exploration_factor': [1.0, 2.0],
                  'sample_with_eps_std': [0, 1],
              } | general_configs

hucrl_config = {
                   'exploration': ['hucrl'],
                   'int_reward_weight': [0.0],
                   'exploration_factor': [1.0, 2.0, ],
                   'sample_with_eps_std': [0, 1],
               } | general_configs

ombrl_config = {
                   'exploration': ['optimistic'],
                   'int_reward_weight': [1.0, 2.0, 3.0],
                   'sample_with_eps_std': [0, 1],
                   'exploration_factor': [1.0, ],
               } | general_configs

flags_combinations = dict_permutations(ombrl_config) + dict_permutations(hucrl_config) \
                     + dict_permutations(pets_config)


def main():
    command_list = []
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
