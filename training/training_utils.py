import os
from importlib import import_module

import numpy as np
import torch
from torch import nn, distributed as dist, multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from data.data_handler import Data
from experiment_utils import get_exp_params
from training.trainer_base import Trainer
from training.utils import ExtendedOptimiser, check_for_exit_file


def start_training(params, get_model, get_optimizer):
    if not params.distributed:
        training_loop(params, get_model, get_optimizer)
    else:
        run_training_distributed(torch.cuda.device_count(),
                                 params, get_model, get_optimizer)


def initialise(get_model, params, ddp_rank, world_size):
    """
    Initialise the model for training, i.e., copy to GPU (possibly multiple GPUs).
    Also, if params.clear_folder is set to True, it wipes the save path clean of files before starting the training.
    Args:
        get_model: Function to load model
        params: Argparse object with comman line arguments
        ddp_rank: if multi-gpu training is used, rank of this job
        world_size: if multi-gpu training is used, number of gpus.

    Returns:

    """
    if ddp_rank is not None:
        setup(rank=ddp_rank, world_size=world_size)
    save_path = params.save_path
    if params.clear_folder:
        [os.system("rm -r {file_or_dir}".format(file_or_dir=os.path.join(save_path, f)))
         for f in os.listdir(save_path) if not f.startswith("screen")]
    exp_params = get_exp_params(save_path)
    network = get_model(exp_params)

    if ddp_rank is not None:
        network = network.cuda()
        network = DDP(network, device_ids=[torch.cuda.current_device()])
        if exp_params["virtual_batch_size"] is not None:
            exp_params["virtual_batch_size"] = exp_params["virtual_batch_size"] / world_size

    elif torch.cuda.device_count() > 1:
        network = nn.DataParallel(network)
        network = network.cuda()
    else:
        network = network.cuda()

    return network, exp_params, save_path


def training_loop(params, get_model, get_optimizer, ddp_rank=None, world_size=None):
    """
    Training loop which loads the network, the experiment parameters and so on and
    runs the training to the last epoch specified in the exp_params file.
    Can be interrupted by a keyboard interrupt signal to the process or by placing a file named 'please_exit'
    in the experiment folder.
    Args:
        params: Arguments as parsed from an argument parser.
        get_model: Function for loading the model.
        get_optimizer: Function for loading the optimiser.
        ddp_rank: In case DistributedDataParallel is used, this would be the rank.
        world_size: In case DistributedDataParallel is used, this is the number of processes.

    Returns:
        None

    """
    torch.cuda.empty_cache()
    with torch.cuda.device(0 + (ddp_rank if ddp_rank is not None else 0)):

        # Initialise network and experiment parameters
        network, exp_params, save_path = initialise(get_model, params, ddp_rank, world_size)
        # Get Data object for loading the training and test data
        data_handler = Data(params.dataset_name, rank=ddp_rank,
                            world_size=None if ddp_rank is None else torch.cuda.device_count(),
                            **exp_params)

        # Create the optimiser to use for the trainer.
        optimiser = ExtendedOptimiser(get_optimizer(network, exp_params["base_lr"]), save_path,
                                      **exp_params)
        # Initialise trainer object
        trainer = Trainer(network, data_handler, save_path, ddp_rank=ddp_rank, **exp_params)
        trainer.set_optimiser(optimiser)

        # Might continue from old checkpoint
        if "continue" in exp_params and exp_params["continue"]:
            trainer.reload()
            if "accuracy" not in trainer.measurements or \
                    float(trainer.epoch) not in np.array(trainer.measurements["accuracy"])[:, 0]:
                trainer.after_epoch()

        # Might not run all epochs at the same time but stop after each epoch
        single_epoch = hasattr(params, "single_epoch") and params.single_epoch

        print("Initialised Trainer. Starting {} epochs of training.".format(exp_params["num_epochs"]))
        optimiser.make_schedule(trainer)
        if ddp_rank is not None:
            dist.barrier()
        # Run single epoch
        while trainer.epoch < exp_params["num_epochs"] and not check_for_exit_file(trainer.save_path):
            trainer.run_epoch()
            if single_epoch:
                break
        if ddp_rank is None or ddp_rank == 0:
            trainer.save(False)

        if ddp_rank is not None:
            cleanup()

        print("Finished training. Woohoo.")


def load_trainer_from_path(path, **kwargs):
    """
    Retrieve the experiment file by parsing the path. The experiments all follow the same naming
    and save_path convention, so that reloading a trainer from a path is easy.
    The network is imported with importlib.
    Args:
        path: Path to the experiment results folder. Should be of the form
            '...experiments/dataset/base_net/experiment_name'.
        **kwargs: Allows to overwrite any parameters in the loaded exp_params.

    Returns:
        The trainer module.

    """
    base_dir, dataset, base_net, exp_name = path.split("/")[-4:]
    base_net_module = ".".join([base_dir, dataset, base_net, "model"])
    base_net_module = import_module(base_net_module)
    exp_params = get_exp_params(path)
    for param, val in kwargs.items():
        exp_params[param] = val
    network = base_net_module.get_model(exp_params)
    data_handler = Data(dataset, **exp_params)
    optimiser = ExtendedOptimiser(base_net_module.get_optimizer(network, exp_params["base_lr"]), path,
                                  **exp_params)
    trainer = Trainer(network, data_handler, path, **exp_params)
    trainer.set_optimiser(optimiser)
    trainer.before_epoch()
    return trainer


#
# Distributed Data Parallel
#

def rank_loop(rank, *args):
    training_loop(*args, ddp_rank=rank, world_size=torch.cuda.device_count())


def run_training_distributed(world_size, *train_loop_args):

    mp.spawn(rank_loop,
             args=train_loop_args,
             nprocs=world_size,
             join=True)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            rank=rank,
                            world_size=world_size,
                            )


def cleanup():
    dist.destroy_process_group()
