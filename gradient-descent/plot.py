from typing import List, Tuple

# import torch
# from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from utils import Encoder


def plot_scores(words: List[str], classes: List[str], scores: np.ndarray):
    ax0 = plt.gca()
    # ax0.set_title('scores matrix')
    c = ax0.pcolor(scores)
    ax0.xaxis.tick_top(); ax0.set_ylim(ax0.get_ylim()[::-1])  
    xticks = tuple(classes)
    yticks = tuple(words)
    plt.xticks(np.arange(0.5, len(xticks) + .5), xticks)
    plt.yticks(np.arange(0.5, len(yticks) + .5), yticks)
    plt.show()
