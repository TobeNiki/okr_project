import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import japanize_matplotlib
#plt.rcParams['font.family'] = "IPAGothic"

def draw_barcharts(value_table, x_labels, title):
    n_rows, n_dims = value_table.shape
    if n_rows != len(title):
        raise ValueError("value_table.shape[0] != len(title)")
    
    if n_dims != len(x_labels):
        raise ValueError("value_table.shape[1] != len(x_labels)")
    
    x_labels = [str(x_label) for x_label in x_labels]

    xmin = -1
    xmax = n_dims
    show_negative = np.min(value_table) < 0
    ymin = np.min(value_table) - 0.5
    ymax = np.max(value_table) + 0.5
    
    fig, axs = plt.subplots(n_rows, 1, figsize=(n_dims, n_rows * 1.5))

    for i, value in enumerate(value_table):
        ax = axs[i]
        ax.bar(x_labels, value,align='center')
        ax.set_title(title[i])
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.hlines([0], xmin, xmax, linestyle='-',linewidth=1)
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.tick_params(left='off', right='off', bottom='off')

        if i == n_rows - 1:
            ax.set_xticklabels(x_labels, rotation=45)
        else:
            ax.set_xticklabels([])
    plt.tight_layout()