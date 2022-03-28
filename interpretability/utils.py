import torch
import numpy as np

from project_utils import to_numpy
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
params = {'text.usetex': True,
          'font.size': 16,
          'text.latex.preamble': [r"\usepackage{lmodern}"],
          'font.family': 'sans-serif',
          'font.serif': 'Computer Modern Sans serif',
          # 'text.latex.unicode': True,
          }
plt.rcParams.update(params)

sns.set_style("darkgrid")

explainers_color_map = {
    "Grad": (255, 255, 255),
    "Ours": (66, 202, 253),
    "Occlusion": (255, 32, 255),
    "GCam": (255, 255, 22),
    "Occ5": (128, 128, 128),
    "RISE": (255, 0, 43),
    "LIME": (112, 224, 65),
    "Occ9": (0, 0, 0),
    'DeepLIFT': (255, 165, 46),
    'IntGrad': (99, 38, 84),
    'IxG': (207, 174, 139),
    'GB': np.array([0.63008579, 1., 1.]) * 255,
}  # Adapted from Boyton's colors


def grad_to_img(img, linear_mapping, smooth=15, alpha_percentile=99.5):
    """
    Computing color image from dynamic linear mapping of B-cos models.
    Args:
        img: Original input image (encoded with 6 color channels)
        linear_mapping: linear mapping W_{1\rightarrow l} of the B-cos model
        smooth: kernel size for smoothing the alpha values
        alpha_percentile: cut-off percentile for the alpha value

    Returns:
        image explanation of the B-cos model
    """
    # shape of img and linmap is [C, H, W], summing over first dimension gives the contribution map per location
    contribs = (img * linear_mapping).sum(0, keepdim=True)
    contribs = contribs[0]
    # Normalise each pixel vector (r, g, b, 1-r, 1-g, 1-b) s.t. max entry is 1, maintaining direction
    rgb_grad = (linear_mapping / (linear_mapping.abs().max(0, keepdim=True)[0] + 1e-12))
    # clip off values below 0 (i.e., set negatively weighted channels to 0 weighting)
    rgb_grad = rgb_grad.clamp(0)
    # normalise s.t. each pair (e.g., r and 1-r) sums to 1 and only use resulting rgb values
    rgb_grad = to_numpy(rgb_grad[:3] / (rgb_grad[:3] + rgb_grad[3:]+1e-12))

    # Set alpha value to the strength (L2 norm) of each location's gradient
    alpha = (linear_mapping.norm(p=2, dim=0, keepdim=True))
    # Only show positive contributions
    alpha = torch.where(contribs[None] < 0, torch.zeros_like(alpha) + 1e-12, alpha)
    if smooth:
        alpha = F.avg_pool2d(alpha, smooth, stride=1, padding=(smooth-1)//2)
    alpha = to_numpy(alpha)
    alpha = (alpha / np.percentile(alpha, alpha_percentile)).clip(0, 1)

    rgb_grad = np.concatenate([rgb_grad, alpha], axis=0)
    # Reshaping to [H, W, C]
    grad_image = rgb_grad.transpose((1, 2, 0))
    return grad_image


def plot_contribution_map(contribution_map, ax=None, vrange=None, vmin=None, vmax=None, hide_ticks=True, cmap="bwr",
                          percentile=100):
    """
    Visualises a contribution map, i.e., a matrix assigning individual weights to each spatial location.
    As default, this shows a contribution map with the "bwr" colormap and chooses vmin and vmax so that the map
    ranges from (-max(abs(contribution_map), max(abs(contribution_map)).
    Args:
        contribution_map: (H, W) matrix to visualise as contributions.
        ax: axis on which to plot. If None, a new figure is created.
        vrange: If None, the colormap ranges from -v to v, with v being the maximum absolute value in the map.
            If provided, it will range from -vrange to vrange, as long as either one of the boundaries is not
            overwritten by vmin or vmax.
        vmin: Manually overwrite the minimum value for the colormap range instead of using -vrange.
        vmax: Manually overwrite the maximum value for the colormap range instead of using vrange.
        hide_ticks: Sets the axis ticks to []
        cmap: colormap to use for the contribution map plot.
        percentile: If percentile is given, this will be used as a cut-off for the attribution maps.

    Returns: The axis on which the contribution map was plotted.

    """
    assert len(contribution_map.shape) == 2, "Contribution map is supposed to only have spatial dimensions.."
    contribution_map = to_numpy(contribution_map)
    cutoff = np.percentile(np.abs(contribution_map), percentile)
    contribution_map = np.clip(contribution_map, -cutoff, cutoff)
    if ax is None:
        fig, ax = plt.subplots(1)
    if vrange is None or vrange == "auto":
        vrange = np.max(np.abs(contribution_map.flatten()))
    im = ax.imshow(contribution_map, cmap=cmap,
                   vmin=-vrange if vmin is None else vmin,
                   vmax=vrange if vmax is None else vmax)

    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    return ax, im


def explanation_mode(model, active=True):
    for mod in model.modules():
        if hasattr(mod, "explanation_mode"):
            mod.explanation_mode(active)
