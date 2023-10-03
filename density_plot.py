import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.stats import norm

# Read in the data
data_filepath = "data/r57673.csv"
df = pd.read_csv(data_filepath)

# Read in the image
img_path = 'figures/r57673.png'
img = plt.imread(img_path)

tanimoto_combo = df['TanimotoCombo']
mean_tanimoto = np.mean(tanimoto_combo)
std_tanimoto = np.std(tanimoto_combo)

# Adjusting the font style and size for prettier visualization
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=1.2, rc={"font.family": "Serif", "axes.labelsize": 16})

# Create a figure and a set of subplots
fig, ax = plt.subplots(figsize=(12, 7))

# Plot the histogram of TanimotoCombo scores
sns.histplot(tanimoto_combo, bins=30, kde=False, ax=ax, label='Histogram', stat='density', alpha=0.6, color='orange')

# Overlay a normal distribution
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean_tanimoto, std_tanimoto)
plt.plot(x, p, 'k', linewidth=4, label='Normal Distribution')

# Add labels and title
plt.xlabel('TanimotoCombo Score', fontsize=20)
plt.ylabel('Density', fontsize=20)
plt.legend(frameon=False)
plt.grid(axis='y')
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

# Overlay the figure with an image of the reagent
imagebox = OffsetImage(img, zoom=0.3)  # zoom parameter to scale the image
ab = AnnotationBbox(imagebox, (xmax, p.max()), frameon=False, box_alignment=(1, 1), bboxprops=dict(edgecolor='none'))
ax.add_artist(ab)

# Show the plot
plt.tight_layout()
plt.savefig(f"figures/density_plot_reagent_57673.png")
plt.show()


