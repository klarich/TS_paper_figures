from typing import List, Dict, Any, Iterable

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

def density_plot(ax, x_label, vertical_line: float, df: pd.DataFrame, plot_title: str, score: str):
    mu = np.mean(df[score])
    std = np.std(df[score])

    # Create a seaborn density histogram of the new data in each subplot
    sns.histplot(df[score], bins=30, kde=False, ax=ax, color='orange', stat="density", label='Data Density')

    # Plot the PDF of the new normal distribution
    xmin, xmax = ax.set_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2, label='Normal PDF')

    # Add vertical line at the specified Z-score for the new mean
    z_score = (vertical_line - mu) / std
    ax.axvline(x=vertical_line, color='r', linestyle='dashed', linewidth=2)

    # Add text label to the vertical line
    ax.text(vertical_line * .95, ax.get_ylim()[1] * 0.5, f'Z = {z_score:.1f}', rotation=90, color='r', ha='right')

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
    # Generate random data with a normal distribution
    # docking_data = pd.read_csv("/Users/kathrynklarich1/dev/TS_paper_figures/data/docking_scores_random_10000.csv")
    # rocs_data = pd.read_csv("/Users/kathrynklarich1/dev/TS_paper_figures/data/rocs_scores_random_10000.csv")
    docking_data = pd.read_csv("/Users/kathrynklarich1/dev/TS_paper_figures/data/docking_scores_random_10000.csv")
    docking_data = iqr_filter(df=docking_data, column='Docking', filter_val=10)

    docking_kwargs = {
        'df': docking_data,
        'x_label': 'X-axis Label 1',
        'vertical_line': 13,
        'plot_title': 'Docking',
        'score': 'Docking'
    }

    rocs_kwargs = {
        'df': pd.read_csv('/Users/kathrynklarich1/dev/TS_paper_figures/data/rocs_scores_random_10000.csv'),
        'x_label': 'X-axis Label 2',
        'vertical_line': 1.8,
        'plot_title': 'ROCS',
        'score': 'TanimotoCombo'
    }

    kwarg_list = [docking_kwargs, rocs_kwargs]
    # Creating a figure with 3 subplots
    fig, axes = plt.subplots(nrows=1, ncols=len(kwarg_list), figsize=(15, 5))

    for ax, kwargs in zip(axes, kwarg_list):
        density_plot(ax=ax, **kwargs)

    plt.tight_layout()
    plt.show()
