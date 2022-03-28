import os
from collections import Iterable

import torch
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel
from torch import nn
from os.path import join

from data.data_transforms import NoTransform
from interpretability.explanation_methods.explainers.captum import IxG
from modules.utils import FinalLayer
from training.utils import Measurements, trim_n_checkpoints, CumulResultDict, default_eval_batch, default_batch_pre_f


class Trainer:

    def __init__(self, model, data_handler, save_path, loss, pre_process_img=NoTransform(),
                 pre_process_batch_f=default_batch_pre_f, eval_batch_f=tuple([default_eval_batch]),
                 ddp_rank=None, verbose=True, to_probabilities=torch.sigmoid, **options):
        """
        General Trainer with functions like run_epoch, evaluations, single step etc.
        Args:
            model: PyTorch neural network model
            data_handler: Data object from data.utils.datasets to store test and train loaders etc.
            save_path: Path for saving and loading models and all related info.
            loss: CombinedLosses module which evaluates all losses.
            pre_process_img: Pre-processing function for images before calling the neural network model.
                This is outsourced so that models with and without different pre-processings can use the same
                dataloader.
            pre_process_batch_f: Do something with the batch before feeding it to the model. This could include
                making a multi-image or an adversarial attack on the batch. Default is just returning same as
                    the dataloader.
            eval_batch_f (Iterable): Metrics to evaluate a mini-batch on. Each should return a dictionary
                with the metric name and the result.
            ddp_rank: Rank in distributed training.
            verbose: Verbosity.
            **options: Any additional parameter for the training process can be added here. This
                includes stuff for printing, number of checkpoints to keep, etc.
        """

        self.options = options
        self.pre_process_img = pre_process_img
        self.pre_process_batch_f = pre_process_batch_f
        self.eval_batch_fs = eval_batch_f if isinstance(eval_batch_f, Iterable) else [eval_batch_f]
        self.model = model
        self.ddp_rank = ddp_rank
        self.to_probabilities = to_probabilities
        self.coda_layers = []
        self.pool_layer = None
        self.final_layer = None
        for m in self.model.modules():
            if hasattr(m, "explanation_mode"):
                self.coda_layers.append(m)
            if isinstance(m, nn.AdaptiveAvgPool2d):
                self.pool_layer = m
            if isinstance(m, FinalLayer):
                self.final_layer = m
        self.loss = loss
        self.data = data_handler
        self.epoch = 0
        self.verbose = verbose
        # Before training, set_optimiser needs to be set!
        self.optimiser = None
        self.base_lr = -1
        self.measurements = Measurements(save_path)

        self.train_loader = self.data.get_train_loader()
        self.test_loader = self.data.get_test_loader()
        self.save_path = save_path

    def set_optimiser(self, optimiser):
        """
        Setting the optimiser for training the model.
        Args:
            optimiser: ExtendedOptimiser to handle learning rate updates and learning rate schedules.

        Returns: None

        """
        self.optimiser = optimiser
        self.base_lr = optimiser.base_lr

    def preprocess_batch(self, batch):
        """
        Function for preprocessing the next mini-batch coming from the
        data loader. In its base functionality it merely expects batch to be a tuple of image and target and transfers
        them to gpu if available.
        This function can be used to make the image a Variable, to make a multi-image, etc.
        Args:
            batch: Batch as coming from the data loader.

        Returns:
            "Pre-processed" batch (tuple). If it returns (None, None), the corresponding step will be skipped.

        """
        img, tgt = self.pre_process_batch_f(self, batch)
        if torch.cuda.is_available():
            return img.cuda(), tgt.cuda()
        return img, tgt

    def eval_batch(self, model_out, model_in, tgt):
        """
        Running evaluation for a mini-batch, such as the classification accuracy.
        Evaluates the self.eval_batch_fs and writes the results in the outgoing dict.
        Args:
            model_out: Output of the model.
            model_in: Input to the model.
            tgt: Target for the model.

        Returns: Dictionary with the evaluation result.

        """
        _ = model_in  # Ignore unused input.
        eval_dict = dict()
        for eval_f in self.eval_batch_fs:
            eval_dict.update(eval_f(model_out, model_in, tgt))
        return eval_dict

    def step(self, batch):
        """
        Processes a given mini-batch of the data loader for one training step.
        Args:
            batch: Next item in the data loader.

        Returns:
            A result dictionary with the different losses and the evaluation of the mini-batch.
            If it returns None, this 'step' will be skipped (e.g. if the batch size does not allow for creating
                a multi-image of the correct size).
            Additionally, return the batch size.

        """
        img, tgt = self.preprocess_batch(batch)
        batch_size = len(img)

        class_scores = self(img)

        batch_results = self.loss(self, class_scores, img, tgt)
        total_loss = batch_results.collect()

        stepped = self.optimiser.maybe_step(total_loss, batch_size=batch_size)
        if stepped and "after_step" in self.options:
            self.options["after_step"](self.model)
        eval_result = self.eval_batch(class_scores, img, tgt)
        batch_results.update(eval_result)
        return batch_results, batch_size

    def predict(self, img, to_probabilities=True):
        """
        Returns the sigmoid of the model output.
        """
        if to_probabilities:
            return self.to_probabilities(self(img))
        return self(img)

    def modules(self):
        """
        Calls and returns self.model.modules()
        """
        return self.model.modules()

    def run_epoch(self):
        """
        Runs the trainer.step() function for all mini-batches in the training set and
        afterwards runs 'after_epoch'.
        Returns: None

        """
        assert self.optimiser is not None, "Please first set an optimiser for training."
        self.before_epoch()
        self.model.train()
        total_samples = len(self.train_loader.dataset)
        self.optimiser.init_epoch()
        lr = self.optimiser.get_current_lr()
        self.measurements.add_measurement("learning_rate", self.epoch, lr)
        results = CumulResultDict(total_samples, self.verbose, **self.options)
        vb = self.options["virtual_batch_size"]
        if vb is None:
            vb = self.options["batch_size"]
        loader_iter = iter(self.train_loader)
        count = 0.

        while True:
            for _ in range(int(vb//self.options["batch_size"]-1)):
                batch = next(loader_iter, None)
                if batch is None:
                    break
                count += len(batch[0])
                result, batch_size = self.step(batch)
                results.add_results(result, batch_size)
                results.maybe_print()
            batch = next(loader_iter, None)
            if batch is None:
                break
            count += len(batch[0])
            if self.ddp_rank is not None:
                dist.barrier()
            result, batch_size = self.step(batch)
            results.add_results(result, batch_size)
            results.maybe_print()

        self.epoch += 1
        self.optimiser.update_epoch(self.epoch, self.model)
        self.measurements.batch_update(self.epoch, results.get_means(), prefix="train-")
        self.after_epoch(**self.options)

    def before_epoch(self):
        if "before_epoch" in self.options and self.options["before_epoch"] is not None:
            self.options["before_epoch"](self)

    def after_epoch(self, eval_every=1, **kwargs):
        """
        Evaluates the model and saves the results.
        Returns: None

        """
        _ = kwargs
        if self.ddp_rank == 0 or self.ddp_rank is None:
            self.save(best_model=False)
            if self.epoch % eval_every == 0:
                eval_results = self.evaluate()
                self.measurements.batch_update(self.epoch, eval_results)
                self.maybe_save_best()
                trim_n_checkpoints(self.save_path, **self.options)

        if self.ddp_rank is not None:
            dist.barrier()

    @torch.no_grad()
    def evaluate(self):
        """
        Evaluates the evaluation metrics and all non-filtered losses (see CombinedLosses) on the
        test set.
        Returns:
            eval_results (dict): Dictionary with all evaluated metrics.

        """
        self.model.eval()
        total_samples = len(self.test_loader.dataset)
        eval_results = CumulResultDict(total_samples, verbose=True)  # Always print the evaluation results.

        for batch in self.test_loader:
            img, tgt = self.preprocess_batch(batch)
            if img is None:
                continue
            class_scores = self(img)
            # Filter losses to only include specified 'eval_losses' to speed up evaluation
            batch_results = self.loss(self, class_scores, img, tgt, filtered=True)
            eval_result = self.eval_batch(class_scores, img, tgt)
            batch_results.update(eval_result)
            eval_results.add_results(batch_results, len(img))
        print("{dashes} Evaluated epoch {epoch:3d} {dashes}".format(epoch=self.epoch, dashes="-"*24))
        eval_results.maybe_print()
        print("\n{dashes}\n".format(dashes="-" * (48 + len(" Evaluated epoch 000 "))))
        return eval_results.get_means()

    def __call__(self, input_img):
        """Evaluates the model on the pre-processed image."""
        return self.model(self.pre_process_img(input_img))

    def save(self, best_model=True):
        """
        Saves model and measurements to save path.
        Args:
            best_model: Whether to save as one of the best checkpoints.

        Returns:
            None

        """
        if best_model:
            torch.save(self.model.state_dict(),
                       join(self.save_path, "model_epoch_{}.pkl".format(self.epoch)))

        last_prefix = "last_model_epoch_"
        # First make sure the model is saved before deleting the old files.
        torch.save(self.model.state_dict(),
                   join(self.save_path, "tmp_{0}{1}.pkl".format(last_prefix, self.epoch)))
        # Remove old (potentially multiple) 'last_model_epoch' files (in case of error there might be more than one).
        last_epochs = [join(self.save_path, f) for f in os.listdir(self.save_path) if f.startswith(last_prefix)]
        for file in last_epochs:
            os.remove(file)
        os.rename(join(self.save_path, "tmp_{0}{1}.pkl".format(last_prefix, self.epoch)),
                  join(self.save_path, "{0}{1}.pkl".format(last_prefix, self.epoch)))
        self.measurements.save_measurements()

    def reload(self, with_model=True, verbose=True):
        """
        Tries to reload the latest checkpoint and measurements in the save_path.
        Args:
            with_model: If False, only load the measurements. If True, load measurements and model.
            verbose: Whether or not to print stuff.

        Returns:
            None

        """
        f_name, epoch = self.get_last_checkpoint()
        if epoch is None:
            if verbose:
                print("No checkpoint found. Not loading any model.")
            return
        if verbose:
            print("Loading epoch {0}.".format(epoch))
        self.epoch = epoch
        if with_model:
            loaded_state_dict = torch.load(f_name, map_location="cpu")
            if "optimiser" in loaded_state_dict:
                # opt_state_dict = loaded_state_dict["optimiser"]
                # self.optimiser.optimiser.load_state_dict(opt_state_dict)
                loaded_state_dict = loaded_state_dict["the_model"]

            state_dict_keys = self.model.state_dict().keys()
            for k in state_dict_keys:
                if "count" in k:
                    print("count was found")
                    print(k in loaded_state_dict.keys() or ("module." + k) in loaded_state_dict.keys())
                if "count" in k and not (k in loaded_state_dict.keys() or ("module." + k) in loaded_state_dict.keys()):
                    print("Setting tensor")
                    loaded_state_dict[k] = torch.tensor([0.])
            loaded_state_dict = loaded_state_dict.items()
            if isinstance(self.model, DistributedDataParallel):

                to_load = {}
                for k, v in loaded_state_dict:
                    options = k, "module."+k, k[len("module."):]
                    for option in options:
                        if option in state_dict_keys:
                            to_load[option] = v
                            continue

                self.model.load_state_dict(to_load)
            else:
                self.model.load_state_dict({k[len("module."):] if k.startswith("module.") else k: v for
                                            k, v in loaded_state_dict
                                            if k in self.model.state_dict().keys() or "count" in k
                                            or k[len("module."):] in self.model.state_dict().keys()
                                            })
        self.measurements.load_measurements()

    def load_best_epoch(self, verbose=True):
        """
        Obtains the epoch with the highest accuracy from the measurements and loads respective file.
        Args:
            verbose: Verbosity.

        Returns:
            None

        """
        self.reload(with_model=False)
        epoch, accuracy = self.measurements.get_best_accuracy()
        if epoch is None:
            if verbose:
                print("No measurements recorded yet. Not loading any model.")
            return
        if verbose:
            print("Loading epoch {0} with accuracy {1:.2f}".format(epoch, accuracy*100))
        f_name = join(self.save_path, "model_epoch_{}.pkl".format(epoch))
        state_dict = torch.load(f_name, map_location="cpu")
        # Load multi-gpu models to single gpu
        state_dict = ({k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()})
        self.model.load_state_dict(state_dict)

    def maybe_save_best(self):
        """
        Save model if it is the best one so far. Save measurements in any case.
        Also save if there is a please_save file in the save_path
        Returns: None
        """
        epoch, accuracy = self.measurements.get_best_accuracy()
        is_best_model = epoch == self.epoch or accuracy == self.measurements["accuracy"][-1][1]
        if is_best_model:  # If best epoch is the current one, save. Otherwise, a previous epoch was better.
            print("Best epoch so far. Saving additional checkpoint.")
            self.save(is_best_model)

    def load_epoch(self, epoch):
        """
        Load epoch by name.
        Args:
            epoch: Name of model checkpoint file or integer value of epoch.

        Returns:
            None
        """
        if isinstance(epoch, int):
            epoch = "model_epoch_{epoch}.pkl".format(epoch=epoch)
        f_name = join(self.save_path, epoch)
        state_dict = torch.load(f_name, map_location="cpu")
        # Load multi-gpu models to single gpu
        state_dict = ({k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()})
        self.model.load_state_dict(state_dict)

    def list_epochs(self):
        """
        Print list of available epochs in self.save_path.
        Returns:
            None
        """
        ckpt_files = [f for f in os.listdir(self.save_path) if
                      f.endswith(".pkl") and "epoch" in f]
        epochs = dict()
        for i, ckpt_file in enumerate(ckpt_files):
            print(i, ckpt_file)
            epochs[i] = ckpt_file
        return epochs

    def get_last_checkpoint(self):
        """

        Returns:
            (f_name, epoch): filename of latest checkpoint and corresponding epoch.

        """
        if not os.path.exists(self.save_path):
            return None, None
        files = os.listdir(self.save_path)

        def get_last(pre):
            post = ".pkl"
            ckpts = [f for f in files if f.startswith(pre) and f.endswith(post)]
            if len(ckpts) == 0:
                return None, None
            _f_name = sorted(ckpts, key=lambda x: int(x[len(pre):-len(post)]))[-1]
            _epoch = int(_f_name[len(pre):-len(post)])
            return join(self.save_path, _f_name), _epoch

        # First try last epoch version
        f_name, epoch = get_last("last_model_epoch_")
        if f_name is not None:
            return f_name, epoch
        # Then try old method of selecting the last model_epoch file.
        return get_last("model_epoch_")

    def explanation_mode(self, active=True):
        """
        (De-)activates the explanation mode for all dynamic linear layers.
        If detaching is active, computing the gradient of the output w.r.t. the image yields the linear matrix.
        Args:
            active (bool): If True, the weight computations are detached from the graph for dynamic linear layers.

        Returns: None

        """
        [m.explanation_mode(active) for m in self.coda_layers]

    def attribute(self, image, target, **kwargs):
        """
        This method returns the contribution map according to Input x Gradient.
        Specifically, if the prediction model is a dynamic linear network, it returns the contribution map according
        to the linear mapping (IxG with detached dynamic weights).
        Args:
            image: Input image.
            target: Target class to check contributions for.
            kwargs: just for compatibility...
        Returns: Contributions for desired level.

        """
        _ = kwargs
        self.explanation_mode(True)
        attribution_f = IxG(self.model)
        att = attribution_f.attribute(self.pre_process_img(image), target)
        self.explanation_mode(False)
        return att

    @torch.no_grad()
    def attribute_selection(self, image, targets, **kwargs):
        """
        Runs trainer.attribute for the list of targets.


        Args:
            image: Input image.
            targets: Target classes to check contributions for.
            kwargs: just for compatibility...

        Returns: Contributions for desired level.

        """
        _ = kwargs
        return torch.cat([self.attribute(image, t) for t in targets], dim=0)
