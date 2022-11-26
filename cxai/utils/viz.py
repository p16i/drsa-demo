from matplotlib import pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap

from . import imagenet


class FigContext:
    def __init__(self, save_path=None, **kwargs):
        self.kwargs = kwargs
        self.save_path = save_path

    def __enter__(self):
        self.fig = plt.figure(**self.kwargs)

        return self.fig

    def __exit__(self, *args):
        if self.save_path is not None:
            self.fig.savefig(self.save_path)

        plt.close(self.fig)


def imshow(img):
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])


def heatmap(
    heatmap,
    title="",
    logit=None,
    reference_heatmap=None,
    total_score=None,
    grid_steps=-1,
    fontsize=None,
):
    """Plot heatmap; this is adapted from https://git.tu-berlin.de/gmontavon/lrp-tutorial/-/blob/main/utils.py

    Args:
        heatmap np.array(h, w):
        reference_heatmap (np.array(h, w), optional): used for calculating normalization values. Defaults to None.
        total_score (float, optional): used for normalizing scores. Defaults to None.
    """
    assert len(heatmap.shape) == 2

    if reference_heatmap is None:
        reference_heatmap = heatmap

    assert len(reference_heatmap.shape) == 2

    b = np.abs(reference_heatmap).max()

    my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    my_cmap[:, 0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)

    sum_Ri = np.sum(heatmap)
    if total_score is None:
        percent = 100
        txt = r"$\sum_i R_i=%.2f$" % (sum_Ri)
    else:
        percent = (sum_Ri / total_score) * 100
        txt = r"$\sum_i R_i=%.4f$ (%3.1f%%)" % (sum_Ri, percent)

    plt.axis("on")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(heatmap, cmap=my_cmap, vmin=-b, vmax=b)

    h = heatmap.shape[0]
    if grid_steps > 0:
        _grid(h, grid_steps)

    plt.title(f"{title}", fontsize=fontsize)
    if logit is not None:
        plt.xlabel(f"{txt}, logit={logit:.4f}", fontsize=fontsize)
    else:
        plt.xlabel(f"{txt}", fontsize=fontsize)


# this grid provides reference locations for further inspection.
def _grid(total_width, steps=4):
    step_size = total_width / steps
    for i in range(1, steps):
        if i % 2 == 0:
            ls = "-"
            alpha = 0.5
        else:
            ls = "--"
            alpha = 0.1
        lw = 1
        plt.axvline(i * step_size, lw=lw, color="black", ls=ls, alpha=alpha)
        plt.axhline(i * step_size, lw=lw, color="black", ls=ls, alpha=alpha)

        plt.axis("on")
        plt.xticks([])
        plt.yticks([])
