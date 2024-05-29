import wandb
import pandas as pd

# Initialize wandb API
api = wandb.Api()

# Set your entity ant project name
entity = "sukhijab"
project = "WTC_RCCar_May19_11_43"

# project = "MBWTC_May03_11_00"

# Fetch all runs from the project
runs = api.runs(f"{entity}/{project}")

# Create an empty list to hold data
data = []

# logging_data = 'eval_true_env/episode_reward'
logging_data = 'eval_true_env/avg_episode_length'

# Loop through runs ant collect data
for idx, run in enumerate(runs):
    print(f"Run {idx + 1}")
    # Example of fetching run name, config, summary metrics
    history = run.scan_history(keys=['episode_idx', logging_data])

    # Save the data of each run, depending on your plotting needs, may need further processing
    reward_performance = [(item['episode_idx'], item[logging_data]) for item in history if
                          logging_data in item and 'episode_idx' in item]
    plot_tuple = list(zip(*reward_performance))

    run_data = {
        "name": run.name,
        "config": run.config,
        "summary": run.summary._json_dict,
        "plot_tuple": plot_tuple,
    }
    data.append(run_data)

# Convert list into pandas DataFrame
df = pd.DataFrame(data)

# Optional: Expand the config ant summary dicts into separate columns
config_df = df['config'].apply(pd.Series)
summary_df = df['summary'].apply(pd.Series)

# Joining the expanded config ant summary data with the original DataFrame
df = df.join(config_df).join(summary_df)

print(df.head())  # Display the first few rows of the DataFrame

# You can now save this DataFrame to a CSV file or perform further analysis
df.to_csv("wtc_model_based_rccar_num_measurement.csv", index=False)
