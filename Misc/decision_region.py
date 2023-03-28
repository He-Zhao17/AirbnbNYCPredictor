from typing import List, Union

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.base import ClassifierMixin
from tqdm.notebook import trange

Number = Union[int, float, complex, np.number]


def pairplot_with_decision_regions(
    title: str,
    cls: ClassifierMixin,
    data: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_indices: List[int],
    feature_names: List[str],
    filler_values: List[Number],
    filler_ranges: List[Number],
) -> None:
    n = len(feature_indices)
    fig, axarr = plt.subplots(n, n, figsize=(10, 10))
    filler_values_dict = dict(enumerate(filler_values))
    filler_ranges_dict = dict(enumerate(filler_ranges))
    color_pallete = ["#2596be", "#ff7f0e", "#2ca02c"]  # sns.color_palette("tab10")
    contourf_kwargs = {"alpha": 0.2}
    scatter_kwargs = {
        "s": 12,
        "edgecolor": None,
        "alpha": 0.7,
        "marker": "o",
        "c": color_pallete,
    }
    scatter_highlight_kwargs = {
        "s": 12,
        "label": "Test data",
        "alpha": 0.7,
        "c": None,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in trange(n):
            for j in trange(n):
                ax = axarr[i, j]
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                index_x, index_y = feature_indices[i], feature_indices[j]
                if i == j:
                    sns.kdeplot(
                        ax=ax,
                        data=data,
                        x=feature_names[i],
                        hue="High Review Score",
                        palette="tab10",
                        alpha=0.5,
                        linewidth=0,
                        multiple="stack",
                        legend=False,
                    )
                    ax.set(
                        xlabel=None,
                        ylabel=None,
                    )
                else:
                    plot_decision_regions(
                        X.values,
                        y.values,
                        clf=cls,
                        ax=ax,
                        feature_index=[index_x, index_y],
                        filler_feature_values={
                            key: filler_values_dict[key]
                            for key in filler_values_dict
                            if key != index_x and key != index_y
                        },
                        filler_feature_ranges={
                            key: filler_ranges_dict[key]
                            for key in filler_ranges_dict
                            if key != index_x and key != index_y
                        },
                        legend=0,
                        X_highlight=X_test.values,
                        scatter_kwargs=scatter_kwargs,
                        contourf_kwargs=contourf_kwargs,
                        scatter_highlight_kwargs=scatter_highlight_kwargs,
                    )
                ax.tick_params(
                    axis="both",
                    which="both",
                    bottom=False,
                    left=False,
                    labelbottom=False,
                    labelleft=False,
                )
                if i == n - 1:
                    ax.set(xlabel=" ".join(feature_names[j].split("_")[:2]))
                    ax.tick_params(
                        axis="x",
                        which="both",
                        bottom=True,
                        labelbottom=True,
                    )
                if j == 0:
                    ax.set(ylabel=" ".join(feature_names[i].split("_")[:2]))
                    ax.tick_params(
                        axis="y",
                        which="both",
                        left=True,
                        labelleft=True,
                    )
    handles, labels = axarr[0, 1].get_legend_handles_labels()

    add_legend(fig, handles, ["0", "1", "Test data"])

    fig.suptitle(f"Decision Boundary for\n{title}", size="xx-large")
    plt.show()


def add_legend(fig, handles, labels):
    # https://github.com/mwaskom/seaborn/blob/bfbd6ad5b9717db42e302177d867b4a273df162b/seaborn/axisgrid.py#L89
    figlegend = fig.legend(handles, labels, loc="center right")

    # See https://github.com/matplotlib/matplotlib/issues/19197 for context
    fig.canvas.draw()
    if fig.stale:
        try:
            fig.draw(fig.canvas.get_renderer())
        except AttributeError:
            pass

    legend_width = figlegend.get_window_extent().width / fig.dpi
    fig_width, fig_height = fig.get_size_inches()
    fig.set_size_inches(fig_width + legend_width, fig_height)

    # Draw the plot again to get the new transformations
    fig.canvas.draw()
    if fig.stale:
        try:
            fig.draw(fig.canvas.get_renderer())
        except AttributeError:
            pass

    # Now calculate how much space we need on the right side
    legend_width = figlegend.get_window_extent().width / fig.dpi
    space_needed = legend_width / (fig_width + legend_width)
    margin = 0.01
    space_needed = margin + space_needed
    right = 1 - space_needed

    # Place the subplot axes to give space for the legend
    fig.subplots_adjust(right=right)
