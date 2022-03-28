import os
import argparse
from importlib import import_module

from project_utils import str_to_bool


class SimpleSchedule:

    def __init__(self, trainer):
        self.trainer = trainer
        self.sched = trainer.options["sched_opts"]["sched"]

    def __call__(self, epoch):
        return self.sched[epoch]

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)


def get_exp_params(save_path):
    """
    Retrieve the experiments specs by parsing the path. The experiments all follow the same naming
    and save_path convention, so that reloading a trainer from a path is easy.
    The experiments are imported with importlib.
    Args:
        save_path: Path to the experiment results folder. Should be of the form
            '...experiments/dataset/base_net/experiment_name'.

    Returns: Experiment specifications as dict.

    """
    base_dir, dataset, base_net, exp_name = save_path.split("/")[-4:]
    exp_params_module = ".".join([base_dir, dataset, base_net, "experiment_parameters"])
    exp_params_module = import_module(exp_params_module)
    exp_params = exp_params_module.get_exp_params(exp_name)
    return exp_params


def get_all_exps(base_dir, dataset, base_net):
    """
    Obtains the dictionary of all possible experiments for the given dataset and basenet specification.
    Returns: exps (dict): Dictionary with the experiments.

    """
    exp_params_module = ".".join([os.path.basename(base_dir), dataset, base_net, "experiment_parameters"])
    exp_params_module = import_module(exp_params_module)
    exps = exp_params_module.exps
    return exps


def argument_parser():
    """
    Create a parser with the standard parameters for an experiment. A bit redundant in its arguments, but hey.
    Overall a bit outdated.
    Returns: The parser.
    """
    parser = argparse.ArgumentParser(description="Dynamic Linear Network Training")
    parser.add_argument("--base_path", default="",
                        help="Base path for saving experiment results.")
    parser.add_argument("--exp_folder", default="experiments",
                        help="Relative path to root folder for the desired experiment configurations.")
    parser.add_argument("--distributed", default=False, type=str_to_bool,
                        help="Path for saving experiment results.")
    parser.add_argument("--dataset_name", default="CIFAR10",
                        type=str, help="Dataset name for data handler.")
    parser.add_argument("--experiment_name", default="9L-S-CoDA-SQ-1000",
                        type=str, help="Experiment name to load.")
    parser.add_argument("--model_config", default="final",
                        type=str, help="Name of the model config folder.")
    parser.add_argument("--continue_exp", default=False, type=str_to_bool,
                        help="Whether or not to continue the experiment from the last checkpoint.")
    parser.add_argument("--single_epoch", default=False, type=str_to_bool,
                        help="Whether or not to run only a single epoch.")
    parser.add_argument("--clear_folder", default=False, type=str_to_bool,
                        help="Whether or not to clear the folder from old content and reset the experiment.")
    return parser


def get_arguments():
    parser = argument_parser()
    opts = parser.parse_args()
    opts.save_path = os.path.join(opts.base_path, opts.exp_folder, opts.dataset_name, opts.model_config,
                                  opts.experiment_name)
    os.makedirs(opts.save_path, exist_ok=True)
    return opts
