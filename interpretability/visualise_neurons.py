import torch
from torch import nn
import numpy as np
from torch.autograd import Variable

from data.data_transforms import AddInverse
from data.imagenet_classnames import name_map
from interpretability.utils import grad_to_img, plot_contribution_map
from project_utils import to_numpy_img, to_numpy
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap


NAMES = {
    739: ("wheels", 100/100),
    797: ("faces", 100/100),
    843: ("snouts", 100/100),
    823: ("legs", 99/100),
    938: ("eyes", 100/100),
    473: ("grass", 98/100),
    341: ("watermarks", 100/100),
}

NUM_PATCHES = {
    739: [1, 2, 1, 2, 2, 2] + 5*[2] + 10*[1],
    797: [1, 2, 1, 2, 1, 1, 2] + 10*[1],
    843: [1]*7 + 10*[1],
    823: [2]*7 + 10*[1],
    938: [2]*7 + 10*[1],
    473: [2]*7 + 10*[1],
    341: [2, 2, 2, 2, 2, 2, 2] + 10*[1],
}


class AxHelper1:

    def __init__(self, ax, x0, y0, w, h, pad=0, imsize=224):
        self.ax = ax
        self.x0 = (x0 + pad * x0 // w) * int(bool(x0))
        self.y0 = y0
        self.w = w
        self.h = h
        self.imsize = imsize
        self.pad = pad
        if y0 < 0:
            ax.vlines([x0, x0 + w], y0, y0 + h, linewidth=.5)
            ax.hlines([y0, y0 + h], x0, x0 + w, linewidth=.5)

    def imshow(self, img, *args, lines=False, **kwargs):
        self.ax.imshow(img, *args, extent=(self.x0, self.x0 + self.w, self.y0,
                                      self.y0 + self.h), **kwargs)
        if lines:
            pad = 1.5
            self.ax.hlines([self.h - pad, self.y0 + pad], self.x0 + pad, self.x0 + self.w - pad, linestyle="dashed",
                      linewidth=2, zorder=10)
            self.ax.vlines([self.x0 + pad, self.x0 + self.w - pad], self.h - pad, self.y0 + pad, linestyle="dashed",
                      linewidth=2, zorder=10)

    def add_rect(self, x0, y0, w, h, color="crimson"):
        pad = 3
        lw = 5
        self.ax.hlines([min(self.y0 + y0, 112) - pad, max(self.y0 + y0 + h, 0) + pad], self.x0 + max(x0, 0) + pad,
                       self.x0 + min(x0 + w, self.imsize) - pad,
                       linestyle="dashed", linewidth=lw, zorder=10, color=color)
        self.ax.vlines([self.x0 + max(x0, 0) + pad, self.x0 + min(x0 + w, self.imsize) - pad], min(self.y0 + y0, 112) - pad,
                       max(self.y0 + y0 + h, 0) + pad,
                       linestyle="dashed", linewidth=lw, zorder=10, color=color)
        self.ax.hlines([min(self.y0 + y0, 112), max(self.y0 + y0 + h, 0)], self.x0 + max(x0, 0), self.x0 + min(x0 + w, self.imsize),
                       linewidth=lw, zorder=5, color="white")
        self.ax.vlines([self.x0 + max(x0, 0), self.x0 + min(x0 + w, self.imsize)], min(self.y0 + y0, 112), max(self.y0 + y0 + h, 0),
                       linewidth=lw, zorder=5, color="white")


def setup_fig(how_many=7, n_minis=30, model_stride=16, imsize=224):
    h_pad = 10
    pad2 = 5
    pad1 = 5
    extent = 24
    ratio = 1 / ((n_minis * 3 * extent + (n_minis - 1) * pad2) / ((how_many) * imsize + (how_many - 1) * pad1))
    scaling = 7

    fig, ax = plt.subplots(1, figsize=np.array((how_many + 10 / imsize,
                                                10 / imsize + 1 / 2 + 3 * extent / imsize * ratio)) * scaling)

    w1 = imsize
    w2 = 3 * extent * ratio
    x0s1 = [i * w1 + i * pad1 for i in range(how_many)]
    h1 = 112
    h2 = 3 * extent * ratio
    y0_1 = 0
    y0_2 = -h_pad - 3 * extent * ratio
    x0s2 = [i * w2 + i * pad2 * ratio for i in range(n_minis)]
    ax.set_xlim(-5, imsize * how_many + 5 * pad1 + 15)
    ax.set_ylim(-3 * extent * ratio - h_pad - 5, 112 + 5)

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    major_axes = [AxHelper1(ax, x0, y0_1, w1, h1, imsize=imsize) for x0 in x0s1]
    minor_axes = [AxHelper1(ax, x0, y0_2, w2, h2, imsize=imsize) for x0 in x0s2]
    return fig, ax, major_axes, minor_axes, (
        h_pad, pad2, pad1, extent, ratio, scaling, how_many, n_minis, model_stride)


def visualise_intermediate_neuron_large(neuron_idx, sorted_activations,
                                        model, data,
                                        fig_opts=None, with_text=True, num_patches=None):
    imsize = data.get_test_loader().dataset[0][0].shape[-1]
    fig_opts = fig_opts if isinstance(fig_opts, dict) else {}
    fig_opts.update({"imsize": imsize})
    fig, ax, major_axes, minor_axes, (
        h_pad, pad2, pad1, extent, ratio, scaling, how_many, n_minis, stride) = setup_fig(**fig_opts)
    proto_idx = 0
    sorted_array = sorted_activations[neuron_idx]
    if num_patches is None:
        num_patches = NUM_PATCHES[neuron_idx] if neuron_idx in NUM_PATCHES else 10 * [1]
    used_imgs = []
    for coord in sorted_array[:how_many + n_minis * 5]:
        if coord[0] in used_imgs:
            continue
        used_imgs.append(coord[0])
        top_coord = coord
        model.zero_grad()
        img = data.get_test_loader().dataset[coord[0]][0][None].cuda()
        _img = Variable(AddInverse()(img), requires_grad=True)
        out = model(_img)
        if proto_idx < how_many:

            vals, coords = torch.topk(out[0, neuron_idx].flatten(), 20)
            coords = np.array(np.unravel_index(to_numpy(coords), out.shape[-2:])).T

            coords = list(coords)
            the_img = to_numpy_img(img[0])
            the_img = np.concatenate((the_img, np.ones_like(the_img[..., :1])), axis=-1)
            reconstruction = np.zeros_like(the_img)
            alpha1 = None
            added_coords = []
            limitsx = []
            limitsy = []
            for i, (val, coord) in enumerate(zip(vals, coords)):
                if i != 0 and np.any([np.abs((coord - added_coords[i]).sum()) < 3 * extent / stride for i in
                                      range(len(added_coords))]):
                    continue

                if len(added_coords) >= num_patches[proto_idx]:
                    break
                added_coords.append(coord)
                val.backward(retain_graph=True)
                coord = None, *coord
                _reconstruction = grad_to_img(_img[0], torch.sign(val).detach() * _img.grad.data[0])
                _reconstruction = np.array(_reconstruction, dtype=float)
                _x0, _w = (coord[2] * 16 - int(1.5 * extent)), 3 * extent
                _y0, _h = (coord[1] * 16 - int(1.5 * extent)), 3 * extent
                limitsx.append(_x0)
                limitsy.append(_y0)
                slicerh = slice(max(_y0, 0), _y0 + _h)
                slicerw = slice(max(_x0, 0), _x0 + _w)

                if alpha1 is None:
                    alpha1 = np.max(_reconstruction[..., -1])
                reconstruction[slicerh, slicerw] = _reconstruction[slicerh, slicerw]
                reconstruction[slicerh, slicerw, -1] = (_reconstruction[slicerh, slicerw, -1] / alpha1).clip(0, 1)

                the_img[slicerh, slicerw, -1] = 0
                _img.grad = None
            ylim1 = top_coord[1] * 16 - 32 * 2
            ylim2 = top_coord[1] * 16 + 32 * 2 - min(ylim1, 0)
            ylim1 = max(0, ylim1 - max(0, ylim2 - imsize))
            ylim2 = min(imsize, ylim2)

            fracy, fracx = (np.array(top_coord[1:]) / out.shape[-1])
            ylim1, ylim2 = int(ylim1 + 16 * fracy), int(ylim2 - 16 * (1 - fracy))
            major_axes[proto_idx].imshow(the_img[ylim1:ylim2])
            major_axes[proto_idx].imshow(reconstruction[ylim1:ylim2])
            if len(added_coords) > 1:
                major_axes[proto_idx].add_rect(limitsx[1],
                                               112 - (limitsy[1] - ylim1),
                                               _w, - _h,
                                               color="orange")
            major_axes[proto_idx].add_rect(limitsx[0],
                                           112 - (limitsy[0] - ylim1),
                                           _w, - _h,
                                           color="deepskyblue")

            proto_idx += 1
            continue
        if proto_idx - how_many >= n_minis:
            break
        out = out[0, neuron_idx, coord[1], coord[2]]
        out.backward()
        reconstruction = grad_to_img(_img[0], torch.sign(out).detach() * _img.grad.data[0])

        reconstruction = np.array(reconstruction, dtype=float)
        reconstruction = reconstruction[max(coord[1] * 16 - int(1.5 * extent), 0):coord[1] * 16 + int(1.5 * extent),
                         max(coord[2] * 16 - int(1.5 * extent), 0):coord[2] * 16 + int(1.5 * extent)]

        minor_axes[proto_idx - how_many].imshow(reconstruction)
        proto_idx += 1

    if with_text and neuron_idx in NAMES and NAMES[neuron_idx][1] is not None:
        f1, f2 = np.array((19, 28)) * scaling / 3
        plt.figtext(-imsize / 3 - 10, 112 / 2, "{} strongest\nactivating  \nimages".format(how_many),
                    fontsize=f1,
                    ha="center", va="center", transform=ax.transData)
        plt.figtext(-imsize / 3 - 10, ax.get_ylim()[0] / 2, "Next strongest\nactivations".format(n_minis),
                    fontsize=f1,
                    ha="center", va="center", transform=ax.transData)
        plt.figtext(-imsize - 15, (ax.get_ylim()[1] + ax.get_ylim()[0]) / 2, "Neuron {}".format(
            neuron_idx, NAMES[neuron_idx][0], int(100 * float(NAMES[neuron_idx][1]))),
                    fontsize=f2, rotation=90,
                    ha="center", va="center", transform=ax.transData)
        plt.figtext(-imsize * 4 / 5 - 15, (ax.get_ylim()[1] + ax.get_ylim()[0]) / 2, "{:3d}/100 \n{}".format(
            int(100 * float(NAMES[neuron_idx][1])), NAMES[neuron_idx][0]),
                    fontsize=f1, rotation=90,
                    ha="center", va="center", transform=ax.transData)

    return fig, ax, major_axes, minor_axes


def visualise_intermediate_neuron_grid(model, data, neuron_idx, layer_idx, sorted_activations,
                          cols=7, rows=2, scaling=2):

    extent = 16 if layer_idx < 5 else 24 if layer_idx < 8 else 32 if layer_idx < 9 else 64

    how_many = cols * rows
    fig, ax = plt.subplots(1, figsize=np.array((cols * 2, rows * 2)) * scaling)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    used_imgs = []
    sorted_array = sorted_activations[neuron_idx]

    for coord in sorted_array:
        if coord[0] in used_imgs:
            continue
        used_imgs.append(coord[0])
        global_idx = len(used_imgs) - 1
        if global_idx == how_many:
            break
        model.zero_grad()
        img, tgt = data.get_test_loader().dataset[coord[0]]
        stride = img.shape[-1]
        img = img[None].cuda()
        _img = Variable(AddInverse()(img), requires_grad=True)
        out = model(_img)
        stride = stride // out.shape[-1]
        out = out[0, neuron_idx, coord[1], coord[2]].max()
        out.backward()
        reconstruction = grad_to_img(_img[0], torch.sign(out).detach() * _img.grad.data[0])

        the_img = to_numpy_img(img[0])
        the_img = np.concatenate((the_img, np.ones_like(the_img[..., :1])), axis=-1)
        y0, y1 = coord[2] * stride - int(1.5 * extent), coord[2] * stride + int(1.5 * extent)
        x0, x1 = coord[1] * stride - int(1.5 * extent), coord[1] * stride + int(1.5 * extent)

        the_img[max(x0, 0):x1, max(y0, 0):y1] = reconstruction[max(x0, 0):x1, max(y0, 0):y1]
        delta_x = x1 - x0
        delta_y = y1 - y0
        xlim0, xlim1 = x0 - .5 * delta_x, x1 + .5 * delta_x
        size = img.shape[-1]
        xlim0, xlim1 = np.array([xlim0 - np.clip(xlim1 - size, 0, None),
                                 xlim1 + np.clip(-xlim0, 0, None)], dtype=int).clip(0, size)
        ylim0, ylim1 = y0 - .5 * delta_y, y1 + .5 * delta_y
        ylim0, ylim1 = np.array([ylim0 - np.clip(ylim1 - size, 0, None),
                                 ylim1 + np.clip(-ylim0, 0, None)], dtype=int).clip(0, size)
        ax.imshow(the_img[xlim0:xlim1, ylim0:ylim1],
                  extent=(
                      2 * delta_x * (global_idx % cols), 2 * delta_x * (global_idx % cols + 1),
                      2 * delta_y * (rows) - 2 * delta_y * (global_idx // cols + 1),
                      2 * delta_y * (rows) - 2 * delta_y * (global_idx // cols),
                  ))
        ax.set_ylim(0 - 1.5, 2 * delta_y * rows + 1.5)
        ax.set_xlim(0 - 1.5, 2 * delta_x * cols + 1.5)
        fig.tight_layout(h_pad=0, w_pad=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.hlines([2 * delta_y * i for i in range(1, rows)], 0, 2 * delta_x * cols, color="white", linewidth=4,
                  zorder=10)
        ax.hlines([2 * delta_y * i for i in (0, rows)], 0 - 4, 2 * delta_x * cols + 4, color="black",
                  linewidth=4, zorder=20)
        ax.vlines([2 * delta_x * i for i in range(1, cols)], 0, 2 * delta_y * rows, color="white", linewidth=4,
                  zorder=10)
        ax.vlines([2 * delta_x * i for i in (0, cols)], 0 - 4, 2 * delta_y * rows + 4, color="black",
                  linewidth=4, zorder=20)

        layer_count = len([m for m in model.modules() if isinstance(m, nn.Conv2d)])
        plt.figtext(-.09 * ax.get_xlim()[1], ax.get_ylim()[1] / 2,
                    "Layer {}\nneuron {}".format(layer_count, neuron_idx),
                    fontsize=30 * scaling, ha="center",
                    va="center", transform=ax.transData, rotation=90)
        plt.figtext(-.025 * ax.get_xlim()[1], ax.get_ylim()[1] / 2, "{} highest actvts.".format(rows * cols),
                    fontsize=30 * scaling, ha="center",
                    va="center", transform=ax.transData, rotation=90)


class AxHelper2:

    def __init__(self, ax, x0, y0, w, h, minipad=0, n_imgs=3):
        self.ax = ax
        self.y0 = y0
        self.x0 = x0
        self.w = w
        self.h = h
        self.minipad = minipad
        self.n_imgs = n_imgs

    def imshow(self, img, *args, **kwargs):
        self.ax.imshow(img, *args, extent=(self.x0, self.x0 + self.w, self.y0, self.y0 + self.h), **kwargs)

    def set_title(self, *args, **kwargs):
        plt.figtext((self.n_imgs * 2 * self.w + self.n_imgs * self.minipad) / 2,
                    self.h * 1.2, *args, **kwargs, ha="center", va="center",
                    transform=self.ax.transData)


def visualise_class_explanations(model, class_idx, img_offsets, data_loader, pad=10,
                                 minipad=5, scaling=10):
    imgs_per_class = len(img_offsets)
    fig, ax = plt.subplots(1, figsize=np.array(
        (1, imgs_per_class * 2 + ((imgs_per_class - 1) * (pad + minipad)) / 224)) * 2 * scaling)
    axes = []

    for col_idx in range(imgs_per_class):
        axes += [AxHelper2(
            ax, (col_idx * 2) * 224 + col_idx * (pad + minipad) + idx * 224 + idx * minipad, 0,
            224, 224, minipad=minipad, n_imgs=imgs_per_class) for idx in range(2)]

    axes = np.array(axes)
    count = 0
    class_name = name_map[class_idx.item()].split(",")[0]

    for _off in img_offsets:
        im_idx = class_idx * 50 + _off

        img, tgt = data_loader.dataset[im_idx]
        _img = Variable(AddInverse()(img[None]).cuda(), requires_grad=True)
        out = model(_img).max()
        axes[count * 2].imshow(to_numpy_img(img))

        _img.grad = None
        model.zero_grad()
        out.backward()
        model.zero_grad()
        ax.set_xticks([])
        ax.set_yticks([])

        exp_c1 = grad_to_img(_img[0], _img.grad.data[0])
        axes[count * 2 + 1].imshow(exp_c1)
        count += 1

    axes[0].set_title(class_name, fontsize=12 * scaling / 3)

    ax.set_ylim(0, 224)
    ax.set_xlim(0, imgs_per_class*2 * 224 + (imgs_per_class*2 - 1) * (pad + minipad))
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.show()


def plot_top2_comparison(model, im_idx, data_loader):
    colors = ["royalblue", "white", "darkorange"]
    cmap = LinearSegmentedColormap.from_list("mycmap", colors)
    fig = plt.figure(figsize=(3 * 4, 1 * 4))
    gs = gridspec.GridSpec(figure=fig, nrows=1, ncols=4,
                           width_ratios=4 * [1], height_ratios=[1], wspace=0.25,
                           hspace=0)

    axes = [fig.add_subplot(gs[0, (i): ((i + 1))]) for i in range(4)]
    model.zero_grad()

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    img, tgt = data_loader.dataset[im_idx]
    axes[0].imshow(to_numpy_img(img))
    _img = Variable(AddInverse()(img[None]).cuda(), requires_grad=True)
    out = model(_img)[0]
    c1, c2 = (torch.topk(out, 2)[1])
    out[c1].backward(retain_graph=True)
    exp_c1 = grad_to_img(_img[0], _img.grad.data[0])
    contribs1 = (_img[0] * _img.grad.data[0]).sum(0)
    model.zero_grad()
    _img.grad = None
    out[c2].backward(retain_graph=True)
    exp_c2 = grad_to_img(_img[0], _img.grad.data[0])
    contribs2 = (_img[0] * _img.grad.data[0]).sum(0)

    out = out.sigmoid() / out.sigmoid().sum()
    axes[0].set_title(name_map[tgt.argmax().item()].split(",")[0],
                      fontsize=18, pad=15)
    axes[1].set_title(name_map[c1.item()].split(",")[0] + " ${:5.2f}\%$".format((out[c1] * 100).item()),
                      fontsize=18, pad=15)
    axes[2].set_title(name_map[c2.item()].split(",")[0] + " ${:5.2f}\%$".format((out[c2] * 100).item()),
                      fontsize=18, pad=15)
    axes[3].set_title("$\Delta$-Explanation", fontsize=18, pad=15)
    exp_c1[..., -1] = (exp_c1[..., -1] / np.percentile(exp_c1[..., -1], 99.5)).clip(0, 1)
    exp_c2[..., -1] = (exp_c2[..., -1] / np.percentile(exp_c2[..., -1], 99.5)).clip(0, 1)
    color_img = contribs1 - contribs2

    axes[1].imshow((exp_c1))
    axes[2].imshow(exp_c2)
    plot_contribution_map(color_img, axes[3], cmap=cmap, percentile=99)
    axes[1].patch.set_linewidth(5)
    axes[1].patch.set_edgecolor(colors[-1])
    axes[2].patch.set_linewidth(5)
    axes[2].patch.set_edgecolor(colors[0])
    plt.show()