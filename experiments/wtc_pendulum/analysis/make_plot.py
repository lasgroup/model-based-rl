import pandas as pd
import numpy as np
from typing import NamedTuple
import matplotlib.pyplot as plt

data = pd.read_csv('data/wtc_model_based_pendulum.csv')

beta_factor = 1
max_time_factor = 5
NUM_SAMPLES = 5
MODEL_FREE_REWARD = 151.2183380126953

filtered_df = data[(data['max_time_factor'] == max_time_factor) &
                   (data['beta_factor'] == beta_factor)]


class Statistics(NamedTuple):
    xs: np.ndarray
    ys_mean: np.ndarray
    ys_median: np.ndarray
    ys_std: np.ndarray


optimistic_data = filtered_df[filtered_df['exploration'] == 'optimistic']['plot_tuple']
pets_data = filtered_df[filtered_df['exploration'] == 'pets']['plot_tuple']
mean_data = filtered_df[filtered_df['exploration'] == 'mean']['plot_tuple']


def prepare_statistics(data: pd.Series) -> Statistics:
    all_values = []
    for tuple in data:
        tuple = eval(tuple)
        indices, values = tuple
        all_values.append(np.array(values))

    all_values = np.stack(all_values)
    all_values = np.cumsum(all_values, axis=1)

    return Statistics(xs=np.arange(20),
                      ys_median=np.median(all_values, axis=0),
                      ys_std=np.std(all_values, axis=0),
                      ys_mean=np.mean(all_values, axis=0))


optimistic_stats = prepare_statistics(optimistic_data)
pets_stats = prepare_statistics(pets_data)
mean_stats = prepare_statistics(mean_data)

plt.plot(optimistic_stats.xs, optimistic_stats.ys_median, label='Optimistic')
plt.fill_between(optimistic_stats.xs,
                 optimistic_stats.ys_median - optimistic_stats.ys_std / np.sqrt(NUM_SAMPLES),
                 optimistic_stats.ys_median + optimistic_stats.ys_std / np.sqrt(NUM_SAMPLES),
                 alpha=0.4, )

plt.plot(pets_stats.xs, pets_stats.ys_median, label='Pets')
plt.fill_between(pets_stats.xs,
                 pets_stats.ys_median - pets_stats.ys_std / np.sqrt(NUM_SAMPLES),
                 pets_stats.ys_median + pets_stats.ys_std / np.sqrt(NUM_SAMPLES),
                 alpha=0.4, )

plt.plot(mean_stats.xs, mean_stats.ys_median, label='Mean')
plt.fill_between(mean_stats.xs,
                 mean_stats.ys_median - mean_stats.ys_std / np.sqrt(NUM_SAMPLES),
                 mean_stats.ys_median + mean_stats.ys_std / np.sqrt(NUM_SAMPLES),
                 alpha=0.4, )

# plt.axhline(y=MODEL_FREE_REWARD, color='r', linestyle='--', label='Model free final reward')
plt.xlabel('Episodes')
plt.ylabel('Number of measurements')
plt.legend()
plt.savefig('wtc_pendulum_num_measurements.pdf')
plt.show()
