import torch

from torch import nn
from torch.hub import download_url_to_file

from data.data_transforms import AddInverse
from experiment_utils import get_arguments
from modules.bcosconv2d import BcosConv2d
from modules.utils import FinalLayer, MyAdaptiveAvgPool2d
from training.training_utils import start_training
import os


def load_pretrained(exp_params, network):
    model_path = os.path.join("bcos_pretrained_C10", exp_params["exp_name"])
    model_file = os.path.join(model_path, "state_dict.pkl")

    if not os.path.exists(model_file):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        download_url_to_file(exp_params["model_url"], model_file, progress=False)
    loaded_state_dict = torch.load(model_file, map_location="cpu")
    network.load_state_dict(loaded_state_dict)


def get_model(exp_params):
    embedding_channels = 6 if isinstance(exp_params["pre_process_img"], AddInverse) else 3
    out_c = exp_params["out_c"]
    ks = exp_params["kernel_sizes"]
    stride = exp_params["stride"]
    padding = exp_params["padding"]
    logit_bias = exp_params["logit_bias"]
    logit_temperature = exp_params["logit_temperature"]
    max_out = exp_params.get("max_out", 1)
    b_exp = exp_params.get("b_exp", 2)
    scale = exp_params.get("scale", 1)
    network_list = []
    emb = embedding_channels

    for i in range(len(out_c)):
        _stride = stride[i]
        mod = BcosConv2d(emb, out_c[i], kernel_size=ks[i], stride=_stride, padding=padding[i], max_out=max_out, b=b_exp,
                         scale=scale)
        network_list += [mod]
        emb = out_c[i]
    network_list += [
        MyAdaptiveAvgPool2d((1, 1)),
        FinalLayer(bias=logit_bias, norm=logit_temperature)
    ]
    network = nn.Sequential(*network_list)
    if exp_params["load_pretrained"]:
        load_pretrained(exp_params, network)

    network.opti = exp_params["opti"]
    network.opti_opts = exp_params["opti_opts"]
    return network


def get_optimizer(model, base_lr):
    optimisers = {"Adam": torch.optim.Adam,
                  "AdamW": torch.optim.AdamW,
                  "SGD": torch.optim.SGD}
    the_model = model if not isinstance(model, (nn.DataParallel, DistributedDataParallel)) else model.module
    opt = optimisers[the_model.opti]
    opti_opts = the_model.opti_opts
    opti_opts.update({"lr": base_lr})
    opt = opt(the_model.parameters(), **opti_opts)
    opt.base_lr = base_lr
    return opt


if __name__ == "__main__":
    cmd_args = get_arguments()
    start_training(cmd_args, get_model, get_optimizer)
