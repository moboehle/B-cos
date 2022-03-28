import torch
from torchvision import transforms
from copy import copy

from data.datasets import TinyImagenet, Imagenet, MNIST, CIFAR10
from data.data_transforms import OneHot, NoTransform, MyToTensor
from torch.utils.data import DataLoader

from project_config import DATA_ROOT


class Data:

    datasets = {
        "MNIST": MNIST,
        "CIFAR10": CIFAR10,
        "TinyImagenet": TinyImagenet,
        "Imagenet": Imagenet,
    }

    default_params = {
        "num_workers": 8,
        "batch_size": 16,
        "test_batch_size": None,
        "num_classes": 10,
    }

    def __init__(self, dataset_name, data_path=DATA_ROOT,
                 pre_data_transforms=NoTransform(),
                 post_data_transforms=NoTransform(),
                 augmentation_transforms=NoTransform(),
                 test_time_transforms=NoTransform(),
                 world_size=None,
                 rank=None,
                 target_transform=None, only_test_loader=False, data_params=None, **kwargs):
        """

        Args:
            dataset_name: Name of the dataset as added in the datasets field of the Data class.
            data_path: Root folder for the dataset.
            pre_data_transforms: Transformations to be applied to all images, irrespective of train or test.
                E.g., this could be resizing of images to a standard size.
            post_data_transforms: Transformations to be applied to all images, irrespective of train or test, after
                augmentation_transforms or test_time_transforms have been applied.
            augmentation_transforms: Augmentation transforms to be used during training.
            test_time_transforms: Transforms specifically for inference time, e.g., CenterCrop.
            world_size: For multi-GPU usage, number of gpus.
            rank: For multi-GPU usage, rank of current gpu.
            target_transform: Transform to be applied to the target loaded from the dataset. E.g., from index to one-hot
                    if None, it defaults to OneHot
            only_test_loader: If this is True, the train loader is not initialised (to save time).
            data_params: Dataset specific parameters can be passed, such as e.g., the class indices for
                            Imagenet subsets.
            **kwargs: Either ignored or overwriting the default params,
                            such as num_workers, batch_size, test_batch_size, and num_classes, see 'default_params'.
        """
        assert dataset_name in Data.datasets, ("Dataset not recognised, available datasets are "
                                               + ", ".join([s for s in Data.datasets.keys()]))

        # Overwrite default parameters
        self.params = copy(Data.default_params)
        for key in Data.default_params.keys():
            if key in kwargs:
                self.params[key] = kwargs[key]
        if self.params["test_batch_size"] is None:
            self.params["test_batch_size"] = self.params["batch_size"]

        self.data_path = data_path
        self.dataset_name = dataset_name
        self.pre_data_transforms = pre_data_transforms
        self.post_data_transforms = post_data_transforms

        if target_transform is None:
            self.target_transform = OneHot(self.params["num_classes"])
        else:
            self.target_transform = target_transform

        # Initialising data loaders
        data_loader_params = dict() if data_params is None else data_params
        self.train_data_loader = None
        if not only_test_loader:
            self.train_data_loader = self.init_data_loader(train=True,
                                                           augmentation_transforms=augmentation_transforms,
                                                           data_params=data_loader_params, world_size=world_size,
                                                           rank=rank)
        self.test_data_loader = self.init_data_loader(train=False,
                                                      augmentation_transforms=test_time_transforms,
                                                      data_params=data_loader_params)

    def get_labels_dict(self):
        return self.train_data_loader.dataset.classes

    def init_data_loader(self, train=True, augmentation_transforms=NoTransform(), data_params=None,
                         world_size=None, rank=None, shuffle=None):
        if data_params is None:
            data_params = dict()

        # combine all transforms
        data_transform = transforms.Compose(
            [self.pre_data_transforms, augmentation_transforms, MyToTensor(),
             self.post_data_transforms]
        )

        # Initialise dataset
        data = Data.datasets[self.dataset_name](self.data_path, train=train, download=False,
                                                transform=data_transform,
                                                target_transform=self.target_transform, **data_params)

        # Multi-GPU data loading
        if rank is not None:
            data_sampler = torch.utils.data.distributed.DistributedSampler(data, num_replicas=world_size,
                                                                           rank=rank)
        else:
            data_sampler = None

        # Return dataloader
        shuffle = shuffle if shuffle is not None else train
        bs = self.params["test_batch_size"] if not train else self.params["batch_size"]
        return DataLoader(data, batch_size=bs,
                          shuffle=shuffle and data_sampler is None, num_workers=self.params["num_workers"],
                          sampler=data_sampler)

    def get_train_loader(self):
        if self.train_data_loader is None:
            return None
        return self.train_data_loader

    def get_test_loader(self):
        return self.test_data_loader
