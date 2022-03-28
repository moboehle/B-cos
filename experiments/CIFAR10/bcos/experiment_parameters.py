import numpy as np
from torchvision import transforms

from data.data_transforms import AddInverse
from experiment_utils import SimpleSchedule
from modules.losses import CombinedLosses, LogitsBCE
from copy import deepcopy as copy

standard_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4)])


default_config = {
    "kernel_sizes": [3] * 8 + [1],
    "stride": [1, 1, 2, 1, 1, 2, 1, 1, 1],
    "padding": [1] * 8 + [0],
    "out_c": [16 * 4] * 2 + [32 * 4] * 3 + [64 * 4] * 3 + [10],
    "augmentation_transforms": standard_aug,
    "virtual_batch_size": None,
    "num_classes": 10,
    "load_pretrained": True,
    "num_epochs": 100,
    "lr_steps": 30,
    "logit_bias": np.log(.1/.9),
    "logit_temperature": 1,
    "decay_factor": 2,
    "stopped": True,
    "base_lr": 5e-4,
    "batch_size": 64,
    "sched_opts": {"sched": [1e-3 * np.cos(np.pi * i / 200) + 1e-5 for i in range(100)]},
    "pre_process_length": 0,
    "schedule": SimpleSchedule,
    "embedding_channels": 6,
    "loss": CombinedLosses(LogitsBCE()), 
    "opti": "Adam",
    "opti_opts": dict(),
    "pre_process_img": AddInverse(),
}


def update_default(params):
    exp = copy(default_config)
    exp.update(params)
    return exp


model_urls = {
    1: "https://nextcloud.mpi-klsb.mpg.de/index.php/s/q8cRRojXZ2WNWbS/download",
    1.25: "https://nextcloud.mpi-klsb.mpg.de/index.php/s/oX6H7jQiSN2dfWJ/download",
    1.5: "https://nextcloud.mpi-klsb.mpg.de/index.php/s/3LxnBCarS8Lpq5b/download",
    1.75: "https://nextcloud.mpi-klsb.mpg.de/index.php/s/XLwEnaKr9bYQJaN/download",
    2: "https://nextcloud.mpi-klsb.mpg.de/index.php/s/mksX2CJMCDQbMYb/download",
    2.25: "https://nextcloud.mpi-klsb.mpg.de/index.php/s/qFsqLPdHLQyNHwH/download",
    2.5: "https://nextcloud.mpi-klsb.mpg.de/index.php/s/XwEAFrPSEYJ2C79/download",
}

ablation_exps = {
    f"9L-M2-B{b_exp}": update_default(
        {
            "model_url": model_urls[b_exp],
            "max_out": 2,
            "b_exp": b_exp,
            "stopped": False,
            "logit_temperature": 10 ** T,
            "scale": 10 ** (-2 + 1.5 * (2.5 - b_exp)),
        }
    )
    for b_exp, T in [(1, -3), (1.25, -3), (1.5, -2), (1.75, 1), (2, 2), (2.25, 2), (2.5, 3)]
}


exps = dict()
exps.update(ablation_exps)
for exp in exps.keys():
    exps[exp]["exp_name"] = exp  # for saving checkpoints in respective folders


def get_exp_params(exp_name):
    if exp_name not in exps:
        raise NotImplementedError("The configuration for {} is not specified yet.".format(exp_name))
    return exps[exp_name]
