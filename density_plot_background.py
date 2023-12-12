import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.stats import norm


def density_plot(data_fp: str, output_fp: str, img_path: str = None, score: str = 'TanimotoCombo', vertical_line: float = None, color: str = 'orange'):
    # Read in the data
    df = pd.read_csv(data_fp)
    tanimoto_combo = df[score]
    mean_tanimoto = np.mean(tanimoto_combo)
    std_tanimoto = np.std(tanimoto_combo)
    # Adjusting the font style and size for prettier visualization
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.2, rc={"font.family": "Serif", "axes.labelsize": 16})
    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(12, 7))
    # Plot the histogram of TanimotoCombo scores
    sns.histplot(tanimoto_combo, bins=30, kde=False, ax=ax, label='Histogram', stat='density', alpha=0.6,
                 color=color)
    # Overlay a normal distribution
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean_tanimoto, std_tanimoto)
    plt.plot(x, p, 'k', linewidth=4, label='Normal Distribution')
    # Add a verticle bar
    if vertical_line is not None:
        plt.axvline(x=vertical_line, color='black', linestyle='dashed', linewidth=2)
        z_score = (vertical_line - mean_tanimoto) / std_tanimoto
        plt.text(x=vertical_line-.03, y=1.0, s=f'{score} = {vertical_line}, Z score = {z_score:.1f}', rotation=90, fontdict={'fontsize': 14})
    # Add labels and title
    plt.xlabel('TanimotoCombo Score', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    plt.legend(frameon=False)
    plt.grid(axis='y')
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    # Overlay the figure with an image of the reagent
    # Read in the image
    if img_path:
        img = plt.imread(img_path)
        imagebox = OffsetImage(img, zoom=0.3)  # zoom parameter to scale the image
        ab = AnnotationBbox(imagebox, (xmax, p.max()), frameon=False, box_alignment=(1, 1),
                            bboxprops=dict(edgecolor='none'))
        ax.add_artist(ab)
    # Show the plot
    plt.tight_layout()
    plt.savefig(output_fp)
    plt.show()


if __name__ == "__main__":
    import fire
    fire.Fire(density_plot)