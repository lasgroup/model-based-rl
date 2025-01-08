import exp
from experiments.util import generate_run_commands, generate_base_command, dict_permutations

ENTITY = 'kiten'

general_configs = {
    'project_name': ['DT_OMBRL_Jan08_23_55_Test1'],
    'num_offline_samples': [0, ],
    'icem_horizon': [40, ],
    'num_particles': [10, ],
    'num_samples': [500, ],
    'num_elites': [50, ],
    'init_std': [1.0],
    'num_steps': [10, ],
    'exponent': [0.0],
    'seed': list(range(5)),
    'num_episodes': [20, ],
    'min_bnn_steps': [5_000],
    'max_bnn_steps': [50_000],
    'linear_scheduler_steps': [20_000],
    'reset_statistical_model': [0],
    'regression_model': ['probabilistic_ensemble', ],
    # 'regression_model': ['FSVGD', 'probabilistic_ensemble'],
    'horizon': [200],
    'log_wandb': [1],
    'entity': [ENTITY, ],
    'calibration': [1],
    'reward_source': ['dm-control'],
}

pets_config = {
                  'exploration': ['pets'],
                  'int_rew_weight_init': [0.0],
                  'int_rew_weight_end': [0.0],
                  'rew_decrease_steps': [15],
                  'exploration_factor': [1.0, ],
                  'sample_with_eps_std': [0],
              } | general_configs

hucrl_config = {
                   'exploration': ['hucrl'],
                   'int_rew_weight_init': [0.0],
                   'int_rew_weight_end': [0.0],
                   'rew_decrease_steps': [15],
                   'exploration_factor': [2.0, ],
                   'sample_with_eps_std': [0],
               } | general_configs

ombrl_config = {
                   'exploration': ['optimistic'],
                   'int_rew_weight_init': [1.0],
                   'int_rew_weight_end': [0.0],
                   'rew_decrease_steps': [15],
                   'sample_with_eps_std': [0, 1],
                   'exploration_factor': [1.0, 2.0],
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
                          duration='23:59:00',
                          prompt=True,
                          mem=16000)


if __name__ == '__main__':
    main()