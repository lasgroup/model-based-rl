import pandas as pd
import numpy as np
from typing import NamedTuple
import matplotlib.pyplot as plt
import matplotlib as mpl

beta_factor = 2
max_time_factor = 5
transition_cost = 0.2
deterministic_policy_for_data_collection = False
first_episode_for_policy_training = 0
NUM_SAMPLES = 5
MODEL_FREE_REWARD = 64.48

LEGEND_FONT_SIZE = 22
TITLE_FONT_SIZE = 30
TITLE_FONT_SIZE = 30
TABLE_FONT_SIZE = 20
LABEL_FONT_SIZE = 26
TICKS_SIZE = 24
OBSERVATION_SIZE = 300
sac_steps = 500_000

NUM_SAMPLES_PER_SEED = 5
LINE_WIDTH = 5

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=
r'\usepackage{amsmath}'
r'\usepackage{bm}'
r'\def\vx{{\bm{x}}}'
r'\def\vf{{\bm{f}}}')

mpl.rcParams['xtick.labelsize'] = TICKS_SIZE
mpl.rcParams['ytick.labelsize'] = TICKS_SIZE


class Statistics(NamedTuple):
    xs: np.ndarray
    ys_mean: np.ndarray
    ys_median: np.ndarray
    ys_std: np.ndarray
    name: str


BASELINE_NAMES = {
    'basline0': 'Optimistic',
    'basline1': 'Pets',
    'basline2': 'Mean',
    'basline3': 'Standard RL',
}

LINESTYLES = {
    'basline0': 'solid',
    'basline1': 'dashed',
    'basline2': 'dotted',
    'basline3': 'dashdot',
}

COLORS = {
    'basline0': 'C0',
    'basline1': 'C1',
    'basline2': 'C2',
    'basline3': 'C3',
}

LINESTYLES_FROM_NAMES = {BASELINE_NAMES[name]: style for name, style in LINESTYLES.items()}
COLORS_FROM_NAMES = {BASELINE_NAMES[name]: color for name, color in COLORS.items()}

################## Number of Measurements ##################
############################################################

data = pd.read_csv('wtc_model_based_rccar_num_measurement.csv')
filtered_df = data[(data['max_time_factor'] == max_time_factor) &
                   (data['beta_factor'] == beta_factor) &
                   (data['transition_cost'] == transition_cost) &
                   (data['deterministic_policy_for_data_collection'] == deterministic_policy_for_data_collection) &
                   (data['first_episode_for_policy_training'] == first_episode_for_policy_training) &
                   (data['sac_steps'] == sac_steps)]

optimistic_data = filtered_df[filtered_df['exploration'] == 'optimistic']['plot_tuple']
pets_data = filtered_df[filtered_df['exploration'] == 'pets']['plot_tuple']
mean_data = filtered_df[filtered_df['exploration'] == 'mean']['plot_tuple']


def prepare_statistics(data: pd.Series, name: str) -> Statistics:
    all_values = []
    for tuple in data:
        tuple = eval(tuple)
        indices, values = tuple
        all_values.append(np.array(values))

    all_values = np.stack(all_values)
    all_values = np.cumsum(all_values, axis=1)

    return Statistics(xs=np.arange(100),
                      ys_median=np.median(all_values, axis=0),
                      ys_std=np.std(all_values, axis=0),
                      ys_mean=np.mean(all_values, axis=0),
                      name=name)


optimistic_stats = prepare_statistics(optimistic_data, 'Optimistic')
pets_stats = prepare_statistics(pets_data, 'Pets')
mean_stats = prepare_statistics(mean_data, 'Mean')
standard_xs = np.arange(100)
standard_rl_stats = Statistics(xs=standard_xs,
                               ys_median=standard_xs * 200 + 200,
                               ys_std=standard_xs * 0,
                               ys_mean=standard_xs * 200 + 200,
                               name='Standard RL')

num_measurement_stats = [optimistic_stats, pets_stats, mean_stats, standard_rl_stats]

################## Reward ##################################
############################################################

data = pd.read_csv('wtc_model_based_rccar_episode_reward.csv')

filtered_df = data[(data['max_time_factor'] == max_time_factor) &
                   (data['beta_factor'] == beta_factor) &
                   (data['transition_cost'] == transition_cost) &
                   (data['deterministic_policy_for_data_collection'] == deterministic_policy_for_data_collection) &
                   (data['first_episode_for_policy_training'] == first_episode_for_policy_training) &
                   (data['sac_steps'] == sac_steps)]

optimistic_data = filtered_df[filtered_df['exploration'] == 'optimistic']['plot_tuple']
pets_data = filtered_df[filtered_df['exploration'] == 'pets']['plot_tuple']
mean_data = filtered_df[filtered_df['exploration'] == 'mean']['plot_tuple']


def prepare_statistics(data: pd.Series, name: str) -> Statistics:
    all_values = []
    for tuple in data:
        tuple = eval(tuple)
        indices, values = tuple
        all_values.append(np.array(values))

    all_values = np.stack(all_values)
    return Statistics(xs=np.arange(100),
                      ys_median=np.median(all_values, axis=0),
                      ys_std=np.std(all_values, axis=0),
                      ys_mean=np.mean(all_values, axis=0),
                      name=name)


optimistic_stats = prepare_statistics(optimistic_data, 'Optimistic')
pets_stats = prepare_statistics(pets_data, 'Pets')
mean_stats = prepare_statistics(mean_data, 'Mean')

episode_reward_stats = [optimistic_stats, pets_stats, mean_stats]

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

for stats in episode_reward_stats:
    axs[0].plot(stats.xs, stats.ys_median, label=stats.name,
                linewidth=LINE_WIDTH,
                linestyle=LINESTYLES_FROM_NAMES[stats.name], )
    axs[0].fill_between(stats.xs,
                        stats.ys_median - stats.ys_std / np.sqrt(NUM_SAMPLES),
                        stats.ys_median + stats.ys_std / np.sqrt(NUM_SAMPLES),
                        alpha=0.4, )

    axs[0].axhline(y=MODEL_FREE_REWARD, color='black', linestyle='-', label='Best Model-Free', linewidth=LINE_WIDTH)
    axs[0].set_xlabel('Episodes', fontsize=LABEL_FONT_SIZE)
    axs[0].set_ylabel('Episode Reward', fontsize=LABEL_FONT_SIZE)

for stats in num_measurement_stats:
    axs[1].plot(stats.xs, stats.ys_median,
                label=stats.name,
                linewidth=LINE_WIDTH,
                linestyle=LINESTYLES_FROM_NAMES[stats.name], )
    axs[1].fill_between(stats.xs,
                        stats.ys_median - stats.ys_std / np.sqrt(NUM_SAMPLES),
                        stats.ys_median + stats.ys_std / np.sqrt(NUM_SAMPLES),
                        alpha=0.4, )

    # plt.axhline(y=MODEL_FREE_REWARD, color='r', linestyle='--', label='Model free final reward')
    axs[1].set_xlabel('Episodes', fontsize=LABEL_FONT_SIZE)
    axs[1].set_ylabel('\# Measurements', fontsize=LABEL_FONT_SIZE)

handles, labels = [], []
for ax in axs:
    for handle, label in zip(*ax.get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)
by_label = dict(zip(labels, handles))

fig.legend(by_label.values(), by_label.keys(),
           ncols=5,
           loc='upper center',
           bbox_to_anchor=(0.5, 0.9),
           fontsize=LEGEND_FONT_SIZE,
           frameon=False)

fig.suptitle('RC Car [Duration=4s]', fontsize=TITLE_FONT_SIZE, y=0.95)
fig.tight_layout(rect=[0.0, 0.0, 1, 0.9])

plt.savefig('wtc_model_based_rccar.pdf')
plt.show()
