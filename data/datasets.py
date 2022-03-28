from __future__ import print_function
import numpy as np
import os
from torchvision.datasets.folder import ImageFolder
from torchvision.datasets import CIFAR10, MNIST

from data.imagenet_classnames import name_map, folder_label_map


# Ignore imports, just importing here to consistently get all datasets from this file.
_ = CIFAR10, MNIST


class TinyImagenet(ImageFolder):

    base_folder = "tiny-imagenet-200"

    def __init__(self, root, train=True, transform=None, target_transform=None, **kwargs):
        _ = kwargs  # Just for consistency with other datasets.
        path = os.path.join(root, self.base_folder, "train" if train else "val")
        super().__init__(path, transform=transform, target_transform=target_transform)
        self.class_labels = {i: folder_label_map[folder] for i, folder in enumerate(self.classes)}


class Imagenet(ImageFolder):

    base_folder = "ILSVRC2012"
    classes = [name_map[i] for i in range(1000)]
    name_map = name_map

    def __init__(self, root, train=True, transform=None, target_transform=None, class_idcs=None,
                 **kwargs):
        _ = kwargs  # Just for consistency with other datasets.
        path = os.path.join(root, self.base_folder, "train" if train else "val")
        super().__init__(path, transform=transform, target_transform=target_transform)
        if class_idcs is not None:
            class_idcs = list(sorted(class_idcs))
            tgt_to_tgt_map = {c: i for i, c in enumerate(class_idcs)}
            self.classes = [self.classes[c] for c in class_idcs]
            self.samples = [(p, tgt_to_tgt_map[t]) for p, t in self.samples if t in tgt_to_tgt_map]
            self.class_to_idx = {k: tgt_to_tgt_map[v] for k, v in self.class_to_idx.items() if v in tgt_to_tgt_map}

        self.class_labels = {i: folder_label_map[folder] for i, folder in enumerate(self.classes)}
        self.targets = np.array(self.samples)[:, 1]
