from typing import List, Dict, Any, Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import seaborn as sns


def iqr_filter(df: pd.DataFrame, column: str, filter_val: float) -> Iterable[str]:
    # Computing IQR
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    # Filtering Values between Q1-1.5IQR and Q3+1.5IQR
    return df.query(f'(@q1 - {filter_val} * @iqr) <= {column} <= (@q3 + {filter_val} * @iqr)')


def density_plot(ax, x_label, vertical_line: Optional[float], df: pd.DataFrame, plot_title: str, score: str, bins: int):
    mu = np.mean(df[score])
    std = np.std(df[score])

    # Create a seaborn density histogram of the new data in each subplot
    sns.histplot(df[score], bins=30, kde=False, ax=ax, color='orange', stat="density", label='Data Density')

    # Plot the PDF of the new normal distribution
    xmin, xmax = ax.set_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2, label='Normal PDF')

    if vertical_line is not None:
        # Add vertical line at the specified Z-score for the new mean
        z_score = (vertical_line - mu) / std
        ax.axvline(x=vertical_line, color='b', linestyle='dashed', linewidth=2)

        # Add text label to the vertical line
        ax.text(vertical_line * .98, ax.get_ylim()[1] * 0.5, f'Z Score = {z_score:.1f}', rotation=90, color='b', ha='right')

        # Set the x-lim
        x_max = max(abs(mu + z_score * std), abs(mu - z_score * std)) * 1.05
        x_min = ax.get_xlim()[0]
        ax.set_xlim(x_min, x_max)

    # Set title and x-axis label for each subplot
    ax.set_title(f'{plot_title}')
    ax.set_xlabel(x_label)

    # Add legend (excluding Z-score line)
    ax.legend()


if __name__ == "__main__":
    import sys

    # Generate random data with a normal distribution
    docking_data = pd.read_csv("data/docking_scores_random_reagent.csv")
    # The IQR filter is needed to filter out bad values (-500 for when docking totally fails)
    docking_data = iqr_filter(df=docking_data, column='Docking', filter_val=10)
    output_fp = sys.argv[1]

    docking_kwargs = {
        'df': docking_data,
        'x_label': 'Docking Score (Free Energy x -1)',
        'vertical_line': None,
        'plot_title': 'Docking',
        'score': 'Docking',
        'bins': 30
    }

    rocs_kwargs = {
        'df': pd.read_csv('data/rocs_scores_random_reagent.csv'),
        'x_label': 'ROCS TanimotoCombo Score',
        'vertical_line': None,
        'plot_title': '3D Overlays',
        'score': 'TanimotoCombo',
        'bins': 30
    }

    tanimoto_kwargs = {
        'df': pd.read_csv("data/tanimoto_scores_random_reagent.csv"),
        'x_label': 'Tanimoto Similarity',
        'vertical_line': None,
        'plot_title': '2D Similarity',
        'score': 'Tanimoto',
        'bins': 30
    }
    kwarg_list = [docking_kwargs, rocs_kwargs, tanimoto_kwargs]
    # Creating a figure with 3 subplots
    fig, axes = plt.subplots(nrows=1, ncols=len(kwarg_list), figsize=(15, 5))

    for ax, kwargs in zip(axes, kwarg_list):
        density_plot(ax=ax, **kwargs)

    plt.tight_layout()
    plt.savefig(output_fp)
    plt.show()
