import os
import traceback
from os.path import join
import numpy as np
import torch
import matplotlib.pyplot as plt

from project_utils import to_numpy
from torch import nn


class TopKAcc:

    def __init__(self, topk=(1,)):
        self.topk = topk

    def __call__(self, model_out, model_in, target):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            target = target.argmax(1)
            maxk = max(self.topk)
            batch_size = target.size(0)

            _, pred = model_out.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = dict()
            for k in self.topk:
                correct_k = correct[:k].view(-1).float().sum(0)
                _res = float(to_numpy((correct_k/batch_size)))
                if k == 1:
                    res["accuracy"] = _res
                else:
                    res["Acc@{k}".format(k=k)] = _res
            return res


def eval_batch_cross_entropy(model_out, model_in, tgt):
    _ = model_in
    return {"accuracy": to_numpy(tgt.argmax(1) == model_out.argmax(1)).mean()}


class ExtendedOptimiser:

    def __init__(self, optimiser, save_path, decay_factor=10, lr_steps=10, virtual_batch_size=None,
                 min_lr=-1, warm_up=0, gradient_clipping=None, **opts):
        """
        This class is meant to give more flexibility for designing learning rate schedules and so on
        whilst leaving the trainer class as general as possible. Whenever the epoch of the optimiser is updated,
        the learning rate is updated according to the ExtendedOptimiser.simple_decay method or the 'schedule' function
        provided in the opts.
        Args:
            optimiser: torch optimiser as a backbone to this extension.
            base_lr: Base learning rate for scheduling.
            decay_factor: By how much to decrease the learning rate all 'lr_steps'.
            lr_steps: At which epochs to decrease the learning rate.
            virtual_batch_size: If not None, the network parameters are only updated whenever the virtual batch size
                is met. Requires virtual_batch_size to be a multiple of batch_size.
            min_lr: Minimum learning rate at which not to decay further.
            warm_up (bool, float+): If False, no warm up epoch is used. Otherwise, divide the base_lr
                by warm_up. Should be a value > 1.
            **opts: Additional options for the optimiser. For example, it could include a scheduling function
                (named 'schedule') to use instead of simple_decay.
        """
        if virtual_batch_size is not None:
            assert virtual_batch_size % opts["batch_size"] == 0, "If you use virtual batches, make sure it is " \
                                                                 "a multiple of the individual batch sizes."
        self.optimiser = optimiser
        self.virtual_batch_size = virtual_batch_size
        self.count = 0
        self.epoch = 0
        self.save_path = save_path
        assert hasattr(self.optimiser, "base_lr"), "The optimiser should have an attribute defining its base_lr."
        self.base_lr = self.optimiser.base_lr
        self.decay_factor = decay_factor
        self.lr_steps = lr_steps
        self.min_lr = min_lr
        self.warm_up = warm_up
        self.intermediate_loss = None
        # For compatibility with old version
        self.param_groups = self.optimiser.param_groups
        self.options = opts
        self.sched = []
        self.gradient_clipping = gradient_clipping

    def make_schedule(self, trainer):
        sched = []
        if "schedule" not in trainer.options:
            lr_func = self.simple_decay
        else:
            assert "sched_opts" in trainer.options, "If you provide a scheduler, you also need to specify its options."
            lr_func = trainer.options["schedule"](trainer)
        for epoch in range(trainer.options["num_epochs"]):
            # When using the LR Finder, it computes all lrs in the first step and then returns the array elements.
            sched.append(lr_func(epoch))
        self.sched = np.array(sched)

    def update_learning_rate(self, lr):
        """Sets a new learning rate for the optimiser."""
        for param_group in self.optimiser.param_groups:
            param_group['lr'] = lr

    def maybe_step(self, loss, batch_size):
        """
        This function calls .backward() on the loss and depending on whether virtual batch sizes are used,
        it updates the parameters directly or first accumulates the gradient.

        Args:
            loss: Total loss of a single batch.
            batch_size: Size of the batch.

        Returns: True if updated, else False
        """
        self.count += batch_size
        if self.virtual_batch_size is not None:
            (loss * batch_size / self.virtual_batch_size).backward()
        else:
            loss.backward()

        if (self.virtual_batch_size is None  # If virtual batch size is None, update always
                or self.count % batch_size != 0  # This can only happen for the final batch
                # Lastly, if the new batch_size is as large as the virtual batch size
                or (self.virtual_batch_size is not None and (self.count + batch_size) % self.virtual_batch_size == 0)):
            if self.gradient_clipping is not None:
                for param_group in self.optimiser.param_groups:
                    nn.utils.clip_grad_norm_(param_group["params"], self.gradient_clipping)

            self.optimiser.step()
            self.optimiser.zero_grad()
            self.intermediate_loss = None
            return True
        return False

    def init_epoch(self):
        self.optimiser.zero_grad()
        self.count = 0

    def update_epoch(self, epoch, model):
        self.epoch = epoch
        lr = self.get_current_lr()
        self.update_learning_rate(lr)

    def simple_decay(self, epoch):
        # If warm up is a positive value and epoch is 0, we start with a lower learning rate.
        # Return the maximum of the minimum learning rate and the lr as computed according to schedule.
        if self.warm_up and not self.epoch:
            return self.base_lr / self.warm_up
        return max(self.min_lr, self.base_lr * self.decay_factor ** (-int(epoch // self.lr_steps)))

    def get_current_lr(self):
        try:
            if len(self.sched):
                lr = self.sched[self.epoch]
                print("Setting Learning Rate To:", lr)
            else:
                lr = self.base_lr
        except Exception as e:
            print(self.epoch, len(self.sched))
            lr = self.base_lr
            print(traceback.print_tb(e.__traceback__))
            print(e)
        return lr


def trim_n_checkpoints(path, keep_n_checkpoints=2, **kwargs):
    """
    For the given path, deletes the oldest checkpoints such that only 'keep_n_checkpoints' remain.
    Args:
        path: Path in which to look for checkpoints.
        keep_n_checkpoints: Maximum number of checkpoints to keep.
        **kwargs: Just here for interface reasons, is not used.

    Returns:
        None

    """
    _ = kwargs
    # Maximally keep n checkpoints, i.e. the final, and n-1 best checkpoints.
    pre = "model_epoch_"
    post = ".pkl"
    ckpts = [f for f in os.listdir(path) if f.startswith(pre) and f.endswith(post) and "itrpt" not in f]
    if not ckpts:
        return None, None
    f_names = sorted(ckpts, key=lambda x: int(x[len(pre):-len(post)]))
    if len(f_names) <= keep_n_checkpoints:
        return
    for f_name in f_names[:-keep_n_checkpoints]:
        print("Removing old best checkpoint", f_name)
        try:
            os.remove(join(path, f_name))
        except FileNotFoundError:
            pass


class Measurements(dict):
    """
    The tracker collects all the measurements collected during training, i.e., all the different losses,
    the learning rate, the classification accuracies etc.
    """
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.best_accuracy_idx = -1

    def add_measurement(self, name, xvalue, yvalue):
        """
        Appends new (x, y)-pair to the measurements named 'name'. If not tracked yet, a new list for this item is
        started.
        Args:
            name: Name of the measurement.
            xvalue: x-value of the measurement (typically the epoch).
            yvalue: Corresponding y-value of the measurement.

        Returns:
            None

        """
        if name == "accuracy":
            if self.best_accuracy_idx == -1:
                self.best_accuracy_idx = 0
            else:
                if self.is_best_acc(yvalue):
                    self.best_accuracy_idx = len(self["accuracy"])  # Index to last entry will be best.
        if name not in self:
            self[name] = [[xvalue, yvalue]]
            return
        self[name] += [[xvalue, yvalue]]

    def save_measurements(self):
        """
        Write measurements to file in .gz format.
        Returns: None
        """
        for name, data in self.items():
            np.savetxt(join(self.save_path, name+".gz"), np.array(data))

    def batch_update(self, epoch, measurement_dict, prefix=""):
        """
        For a given single x-value (typically the epoch), add all measurements contained in the given dict
        to the respective lists.
        Args:
            epoch: Epoch for which these measurements were recorded.
            measurement_dict: Dictionary of (name, y-value)-pairs for this epoch.
            prefix: The measurements are prefixed in the object's dictionary with this prefix.
                Intended use: all train measurements are prefixed by 'train-'

        Returns:
            None
        """
        for key, value in measurement_dict.items():
            self.add_measurement(prefix + key, epoch, value)

    def load_measurements(self):
        """
        Reloads measurements from the save path.
        Returns: None
        """
        saved_files = [f for f in os.listdir(self.save_path) if f.endswith(".gz")]
        for file in saved_files:
            m = np.loadtxt(join(self.save_path, file))
            if len(m.shape) == 1:
                m = m[None]
            self[file[:-3]] = m.tolist()

        if "accuracy" in self:
            self.best_accuracy_idx = np.array(self["accuracy"])[:, 1].argmax()

    def get_best_accuracy(self):
        """ Returns epoch and accuracy for the epoch with the highest 'accuracy' measurement."""
        if self.best_accuracy_idx == -1:
            return None, None
        epoch, accuracy = self["accuracy"][self.best_accuracy_idx]
        return int(epoch), accuracy

    def is_best_acc(self, yvalue):
        return self["accuracy"][self.best_accuracy_idx][1] < yvalue

    def plot(self, metric, ax=None, xlabel="Epochs"):
        if ax is None:
            fig, ax = plt.subplots(1)
        ax.plot(*np.array(self[metric]).T, "o--")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric)
        ax.grid(which="major", alpha=.5, linestyle="--")


class PrintItem:
    """
    Simple class to allow for formatting values more easily when printing during training.
    """

    def __init__(self, name, fmt, func=None):
        """
        Args:
            name: Name of the item to be formatted.
            fmt: Formatting string, e.g. '>5.2f'
        """
        self.template = "{" + name + ":" + fmt + "}"
        self.name = name
        self.func = func if func is not None else lambda x: x

    def to_string(self, value):
        """

        Args:
            value: Value to format.

        Returns:
            Formatted value according to the fmt passed in init.
        """
        return ": ".join([self.name, self.template.format(**{self.name: self.func(value)})])


class CumulResultDict(dict):

    def __init__(self, total_samples, verbose=True, print_every=500, loss_fmt=">10.8f", **opts):
        """
        This dictionary-type object accumulates measurements over an epoch to allow for flexibly printing and
        collecting different measurements.
        Args:
            total_samples: Number of total samples in the data loader.
            verbose: Whether or not to print stuff in maybe_print.
            **opts: The 'opts' could include 'print_items' which should be of class PrintItem.
                By passing this as additional options when initialising the Trainer, it is possible to specify
                which losses shall be printed in the logs. As default, only tries to print the classification loss.
        """
        super().__init__()
        self.item_count = 0.
        self.last_print = -1
        self.print_every = print_every
        self.total_samples = total_samples
        self.verbose = verbose
        if "print_items" in opts:
            self.print_items = opts["print_items"]
        else:
            self.print_items = [
                PrintItem("accuracy", ">5.2f", lambda x: 100*x),
                PrintItem("Acc@5", ">5.2f", lambda x: 100*x),
                                ]
        self.print_names = [p.name for p in self.print_items]
        self.loss_fmt = ">10.8f"

    def add_results(self, results, batch_size):
        """
        Add the values in the results dictionary to the already stored items, multiplied by the batch-size.
        This implicitly assumes that all results are given as the mean over the mini-batch.
        """
        for key, value in results.items():
            value = self.to_primitive(value) * batch_size
            if key not in self.keys():
                self.update({key: value})
                continue
            self[key] += value
        self.item_count += batch_size

    @staticmethod
    def to_primitive(value):
        """Calls .item() on tensor objects, otherwise returns value again."""
        if isinstance(value, torch.Tensor):
            value = value.item()
        if isinstance(value, np.float):
            value = float(value)
        assert type(value) in (float, int, bool, str), str(type(value))
        return value

    def maybe_print(self):
        """If verbose, prints the status of the current epoch in terms of (%) of batch processed and
            cumulative losses.
        """
        print_idx = self.item_count // self.print_every
        if print_idx > self.last_print:
            self.last_print = print_idx
        else:
            return

        if not self.verbose:
            return
        status = "{0:>6.2f}% processed.".format(100*float(self.item_count) / self.total_samples)
        means = self.get_means()
        for mean in means.keys():
            if "Loss" not in mean or mean in self.print_names:
                continue
            self.print_names.append(mean)
            self.print_items.append(PrintItem(mean, self.loss_fmt))

        status = " | ".join([status, *[printer.to_string(means[printer.name]) for printer in self.print_items
                                       if printer.name in means]])
        print(status, flush=True)

    def get_means(self):
        """
        For all entries in the dictionary get the means.
        """
        return {k: v / self.item_count for k, v in self.items()
                if v is not None}


def check_for_exit_file(path):
    exists = "please_exit" in os.listdir(path)
    if exists:
        print("Found exit file. Will exit training now.")
        os.remove(join(path, "please_exit"))
    return exists


def default_eval_batch(model_out, model_in, tgt):
    _ = model_in
    return {"accuracy": to_numpy(tgt.argmax(1) == model_out.argmax(1)).mean()}


def default_batch_pre_f(trainer, batch):
    _ = trainer
    return batch
