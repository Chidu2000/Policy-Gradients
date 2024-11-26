from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import numpy as np

save_paths = {
    "REINFORCE": Path("results/reinforce_logger_history.pickle"),
    "REINFORCE + baseline": Path("results/reinforce_baseline_logger_history.pickle"),
    "Actor-Critic (4)": Path("results/actor_critic_logger_history_4.pickle"),
    "Actor-Critic (200)": Path("results/actor_critic_logger_history_200.pickle"),
}

max_v = -np.inf
for algo, path in save_paths.items():
    with path.open("rb") as f:
        save_dict = pickle.load(f)
        episodes = save_dict["episodes"]
        returns = save_dict["history"]["evaluation return"]
        return_stds = save_dict["history"]["evaluation return_std"]

    episode_values = np.array([ep for ep, val in zip(episodes, returns) if val is not None])
    metric_values = np.array([val for val in returns if val is not None])
    metric_stds = np.array([val for val in return_stds if val is not None])

    plt.plot(episode_values, metric_values, label=algo)
    plt.fill_between(episode_values, metric_values + metric_stds, metric_values - metric_stds, alpha=0.1)
    max_v = max(max_v, metric_values.max())

plt.ylim(0, max_v * 1.5)
plt.hlines(1180, *plt.xlim(), label="baseline", colors="r") # Baseline average is 1180
plt.title("Comparison of all algorithm performances")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.legend()
plt.plot()
plt.savefig("results/figures/comparison_all_algos.png")
