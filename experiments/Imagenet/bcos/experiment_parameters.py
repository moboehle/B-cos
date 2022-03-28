from data.augmentations.rand_augment import RandAugment, Lighting
import numpy as np
from torchvision import transforms

from data.data_transforms import AddInverse, NoTransform
from experiment_utils import SimpleSchedule
from modules.losses import CombinedLosses, LogitsBCE, AuxLoss
from copy import deepcopy as copy
from torch import nn
from training.utils import TopKAcc

_IMAGENET_PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ]
}


def cosine_scheduler(base_value, final_value, epochs, warmup_epochs=5, start_warmup_value=1e-6):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs
    return schedule


def get_aug_trans(n=2, m=9, s=160, min_scale=0.08):
    return transforms.Compose([
                *([RandAugment(n=n, m=m)] if n > 1 else []),
                transforms.RandomResizedCrop(s, scale=(min_scale, 1.0)),
                transforms.RandomHorizontalFlip(),
                *([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)]
                  if n > 0 else []
                  ),
                transforms.ToTensor(),
                *([Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec'])]
                  if n > 0 else []
                  )
            ])


default_config = {
    "virtual_batch_size": 256,
    "batch_size": 256,
    "num_classes": 1000,
    "loss": CombinedLosses(LogitsBCE()),
    "training": True,
    "load_pretrained": True,
    "logit_bias": np.log(.01 / .99),
    "keep_n_checkpoints": 2,
    "opti": "Adam",
    "opti_opts": dict(),
    "eval_batch_f": TopKAcc((1, 5)),
    "logit_temperature": 10**(-1),
    "stopped": False,
    "base_lr": 2.5e-4,
    "augmentation_transforms": get_aug_trans(m=9, s=224),
    "test_time_transforms": transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224)]),
    "add_inverse": True,
    "pre_process_length": 0,
    "lr_steps": 60,
    "num_epochs": 100,
    "pre_process_img": AddInverse(),
    "fraction_of_batch": 1,
    "deterministic": False,
    "decay_factor": 10,
}


def update_default(params):
    exp = copy(default_config)
    exp.update(params)
    return exp


pretrained = {
    "pretrained-{}".format(net): update_default({
        "num_epochs": 0,
        "pretrained": True,
        "stopped": True,
        "network": "{d}".format(d=net),
        "logit_temperature": 1,
        "logit_bias": 0,
        "test_time_transforms": transforms.Compose([transforms.Resize(256),
                                                    transforms.CenterCrop(224)]),
        "to_probabilities": nn.Softmax(dim=1),
        "pre_process_img": NoTransform(),
    })
    for net in ["densenet121", "resnet34", "vgg11", "inception"]
}


densenets = {
    "densenet_{}".format(d): update_default({
        "model_url": "https://nextcloud.mpi-klsb.mpg.de/index.php/s/LW2HsNDzAzaaaDH/download",
        "network": "densenet_{d}".format(d=d),
        "logit_temperature": 10 ** (-3),
        "network_opts": {"memory_efficient": False},
    })
    for d in [121]
}


densenets_cossched = {
    "densenet_{}_cossched".format(d): update_default({
        "model_url": "https://nextcloud.mpi-klsb.mpg.de/index.php/s/G8kdm73qWn4L8Lg/download",
        "network": "densenet_{d}".format(d=d),
        "logit_temperature": 10 ** (-3),
        "batch_size": 128,
        "virtual_batch_size": 128,
        "num_epochs": 200,
        "sched_opts": {"sched": cosine_scheduler(1e-3, 1e-4, 200, warmup_epochs=10, start_warmup_value=1e-5)},
        "schedule": SimpleSchedule,
        # Memory efficient version is much slower and on slurm there is space even if using just two GPUs
        "network_opts": {"memory_efficient": False},
    })
    for d in [121]
}

resnet = {
    "resnet_{d}".format(d=d): update_default({
        "model_url": "https://nextcloud.mpi-klsb.mpg.de/index.php/s/mYe6TggQsDymZbk/download",
        "logit_temperature": 1e1,
        "network": "resnet_{}".format(d),
        "network_opts": dict(),
    })
    for d in [34]
}

vgg = {
    "vgg_{d}".format(d=d): update_default({
        "model_url": "https://nextcloud.mpi-klsb.mpg.de/index.php/s/kEHgFSSKMEdeA5N/download",
        "network": "vgg_{}".format(d),
        "logit_temperature": 1e-1,
        "network_opts": {"emb_ch": 6},
        "batch_size": 128
    })
    for d in [11]
}

inception_v3 = {
    "inception_v3": update_default({
        "model_url": "https://nextcloud.mpi-klsb.mpg.de/index.php/s/QSEo6QSNgPZqNfi/download",
        "network": "inception_v3",
        "logit_temperature": 1,
        "base_lr": 5e-4,
        "network_opts": {"aux_logits": True},
        "test_time_transforms": transforms.Compose([transforms.Resize(320), transforms.CenterCrop(299)]),
        "loss": CombinedLosses(LogitsBCE(), AuxLoss(1)),
        "augmentation_transforms": get_aug_trans(m=9, s=299),
    })
}


exps = dict()
exps.update(densenets)
exps.update(densenets_cossched)
exps.update(inception_v3)
exps.update(vgg)
exps.update(resnet)

for exp in exps.keys():
    exps[exp]["exp_name"] = exp  # for saving checkpoints in respective folders


def get_exp_params(exp_name):
    if exp_name not in exps:
        raise NotImplementedError("The configuration for {} is not specified yet.".format(exp_name))
    return exps[exp_name]
