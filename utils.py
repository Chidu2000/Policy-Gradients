from pathlib import Path
from typing import Iterable, Optional, Union
import jax
import jax.numpy as jnp
import chex
from flax import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


@struct.dataclass
class JaxState:
    pass

@struct.dataclass
class Transition(JaxState):
    """
    Class representing a transition (s, a, r, s') with additionnal informations such as the valid actions that can be taken from s
    and whether the next state is terminal.

    :param jax.Array observation: Observation of the state `s`, array of shape [4, 4, 31] and dtype jnp.float32
    :param jax.Array action: Action `a`, array of empty shape and dtype jnp.int32
    :param jax.Array next_observation: Observation of the next state `s'`, array of shape [4, 4, 31] and dtype jnp.float32
    :param jax.Array reward: Reward `r` for taking action `a` in state `s`, array of empty shape and dtype jnp.float32
    :param jax.Array done: Whether the transition (s, a, s') is terminal or not, array of empty shape and dtype jnp.bool
    :param jax.Array action_mask: Mask of the allowed actions to be taken in state `s`, array of shape [4] and dtype jnp.bool
    """
    observation: jax.Array
    action: jax.Array
    next_observation: jax.Array
    reward: jax.Array
    done: jax.Array
    action_mask: jax.Array


class Logger:
    def __init__(self, values_to_record: list[str]) -> None:
        self.values_to_record = values_to_record
        self.all_values_to_record = []
        for value_to_record in values_to_record:
            self.all_values_to_record.append(value_to_record)
            self.all_values_to_record.append(f"{value_to_record}_std")

        self.history = {k: [] for k in self.all_values_to_record}
        self.temp_record = {k: [] for k in self.values_to_record}
        self.episodes = []
        self.last_logged_episode = -1

    def record(self, values: dict[str, jax.Array]) -> None:
        """
        Records values of given metrics and add them to the list of values recorded for the current episode
        """
        for k, v in values.items():
            if len(v.squeeze().shape) == 0:
                self.temp_record[k].append(v.squeeze().item())
            else:
                self.temp_record[k].extend(np.array(v))

    def log(self, episode: int, **kwargs: float) -> None:
        """
        Computes and logs the average of each metric for the given episode
        """
        self.record(kwargs)
        if episode > self.last_logged_episode:
            self.last_logged_episode = episode
            self.episodes.append(episode)
            for metric_name, logged_values in self.temp_record.items():
                if len(logged_values):
                    mean_value = np.mean(logged_values)
                    std_value = np.std(logged_values)
                else:
                    mean_value = None
                    std_value = None

                self.history[metric_name].append(mean_value)
                self.history[f"{metric_name}_std"].append(std_value)

            self.temp_record = {k: [] for k in self.values_to_record}

    def _plot_ax(self, ax: Axes, metric_name: str, baselines: dict[str, float]) -> None:
        """Utility function to plot results to any matplotlib axes"""
        ax.set_title(f"{metric_name.capitalize()} per episode")
        episode_values = np.array([ep for ep, val in zip(self.episodes, self.history[metric_name]) if val is not None])
        metric_values = np.array([val for val in self.history[metric_name] if val is not None])
        stds = np.array([val for val in self.history[f"{metric_name}_std"] if val is not None])
        ax.plot(episode_values, metric_values)
        ax.fill_between(episode_values, metric_values + stds, metric_values - stds, alpha=0.1)

        max_v = max(baselines.get(metric_name, -np.inf), metric_values.max())
        min_v = min(baselines.get(metric_name, np.inf), metric_values.min())
        max_abs_v, min_abs_v = max(abs(max_v), abs(min_v)), min(abs(max_v), abs(min_v))

        if (max_abs_v / min_abs_v) < 10:
            bottom_margin = abs(min_v) * 0.2
            top_margin = abs(max_v) * 0.2

        else:
            bottom_margin = top_margin = max_abs_v * 0.2

        ax.set_ylim(min_v - bottom_margin, max_v + top_margin)

        if metric_name in baselines:
            ax.hlines(baselines[metric_name], *ax.get_xlim(), colors="r")

    def plot(self, plot_name: str, baselines: dict[str, float] = {}) -> None:
        """Makes nice figures"""
        fig, axes = plt.subplots(1, len(self.values_to_record), figsize=(7.5 * len(self.values_to_record), 8))
        fig.suptitle(plot_name)
        if len(self.values_to_record) == 1:
            axes = [axes]

        for ax, metric_name in zip(axes, self.values_to_record):
            self._plot_ax(ax, metric_name, baselines)

        plt.show()

    def save_figures(self, save_root: Path, baselines: dict[str, float] = {}, suffix: str = "") -> None:
        """Saves individual metric plots to files"""
        _suffix = "" if (suffix == "") else f"_{suffix}"
        for metric_name in self.values_to_record:
            fig = plt.figure(figsize=(15, 8))
            ax = fig.gca()
            self._plot_ax(ax, metric_name, baselines)
            fig.savefig(str(save_root / f"{metric_name.lower().replace(' ', '_')}{_suffix}.png"))
