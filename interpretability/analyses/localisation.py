import torch
import argparse
import os
import pickle
from os.path import join
import numpy as np
import torch.nn.functional as F

from interpretability.analyses.utils import load_trainer, Analyser
from interpretability.explanation_methods import get_explainer
from project_utils import to_numpy
from interpretability.analyses.localisation_configs import configs


class LocalisationAnalyser(Analyser):

    default_config = {
        "explainer_name": "Ours",
        "explainer_config": None
    }
    conf_fn = "conf_results.pkl"

    def __init__(self, trainer, config_name, plotting_only=False, verbose=True, **config):
        """
        This analyser evaluates the localisation metric (see CoDA-Net paper).
        Args:
            trainer: Trainer object.
            plotting_only: Whether or not to load previous results. These can then be used for plotting.
            **config:
                explainer_config: Config key for the explanation configurations.
                explainer_name: Which explanation method to load. Default is Ours.
                verbose: Warn when overwriting passed parameters with the analysis config parameters.

        """
        self.config_name = config_name
        analysis_config = configs[config_name]
        if verbose:
            for k in analysis_config:
                if k in config:
                    print("CAVE: Overwriting parameter:", k, analysis_config[k], config[k], flush=True)
        analysis_config.update(config)
        super().__init__(trainer, **analysis_config)
        if plotting_only:
            self.load_results()
            return
        self.explainer = get_explainer(trainer, self.config["explainer_name"], self.config["explainer_config"])
        self.sorted_confs = None
        save_folder = join(trainer.save_path, self.get_save_folder())
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        self.compute_sorted_confs()

    def compute_sorted_confs(self):
        """
        Sort image indices by the confidence of the classifier and store in sorted_confs.
        Returns: None

        """
        save_path = join(self.trainer.save_path, "localisation_analysis", "epoch_{}".format(self.trainer.epoch))
        fp = join(save_path, self.conf_fn)

        if os.path.exists(fp):
            print("Loading stored confidences", flush=True)
            with open(fp, "rb") as file:
                self.sorted_confs = pickle.load(file)
            return
        print("No confidences file found, calculating now.", flush=True)
        trainer = self.trainer
        confidences = {i: [] for i in range(trainer.options["num_classes"])}

        loader = trainer.data.get_test_loader()
        img_idx = -1
        with torch.no_grad():
            for img, tgt in loader:
                img, tgt = img.cuda(), tgt.cuda()
                logits, classes = trainer.predict(img, to_probabilities=False).max(1)
                for logit, pd_class, gt_class in zip(logits, classes, tgt.argmax(1)):
                    img_idx += 1
                    if pd_class != gt_class:
                        continue
                    confidences[int(gt_class.item())].append((img_idx, logit.item()))

        for k, vlist in confidences.items():
            confidences[k] = sorted(vlist, key=lambda x: x[1], reverse=True)

        with open(fp, "wb") as file:
            pickle.dump(confidences, file)

        self.sorted_confs = confidences

    def get_sorted_indices(self):
        """
        This method generates a list of indices to be used for sampling from the dataset and evaluating the
            multi images.
        In particular, the images per class are sorted by their confidence.
        Then, a random set of n classes (for the multi image) is sampled and for each class the next
            most confident image index that was not used yet is added to the list.
        Thus, when using this list for creating multi images, the list contains blocks of size n with
        image indices such that (1) each class occurs at most once per block and (2) the class confidences
            decrease per block for each class individually.

        Returns: list of indices

        """
        idcs = []
        classes = np.array([k for k in self.sorted_confs.keys()])
        class_indexer = {k: 0 for k in classes}

        # Only use images with a minimum confidence of 50%
        # This is, of course, the same for every attribution method
        def get_conf_mask_v(_c_idx):
            return torch.tensor(self.sorted_confs[_c_idx][class_indexer[_c_idx]][1]).sigmoid().item() > .5
        # Only use classes that are still confidently classified
        mask = np.array([get_conf_mask_v(k) for k in classes])
        n_imgs = self.config["n_imgs"]
        # Always use the same set of classes for a particular model
        np.random.seed(42)
        while mask.sum() > n_imgs:
            # Of the still available classes, sample a set of classes randomly
            sample = np.random.choice(classes[mask], size=n_imgs, replace=False)

            for c_idx in sample:
                # Store the corresponding index of the next class image for each of the randomly sampled classes
                img_idx, conf = self.sorted_confs[c_idx][class_indexer[c_idx]]
                class_indexer[c_idx] += 1
                mask[c_idx] = get_conf_mask_v(c_idx) if class_indexer[c_idx] < len(self.sorted_confs[c_idx]) else False
                idcs.append(img_idx)
        return idcs

    def get_save_folder(self, epoch=None):
        """
        'Computes' the folder in which to store the results.
        Args:
            epoch: currently evaluated epoch.

        Returns: Path to save folder.

        """
        if epoch is None:
            epoch = self.trainer.epoch
        return join("localisation_analysis", "epoch_{}".format(epoch),
                    self.config_name,
                    self.config["explainer_name"],
                    "smooth-{}".format(int(self.config["smooth"])),
                    self.config["explainer_config"])

    def analysis(self):
        sample_size, n_imgs = self.config["sample_size"], self.config["n_imgs"]
        trainer = self.trainer
        loader = trainer.data.get_test_loader()
        fixed_indices = self.get_sorted_indices()
        metric = []
        explainer = self.explainer
        offset = 0
        single_shape = loader.dataset[0][0].shape[-1]
        for count in range(sample_size):
            multi_img, tgts, offset = self.make_multi_image(n_imgs, loader, offset=offset,
                                                            fixed_indices=fixed_indices)
            # calculate the attributions for all classes that are participating
            attributions = explainer.attribute_selection(multi_img, tgts).sum(1, keepdim=True)
            if smooth:
                attributions = F.avg_pool2d(attributions, smooth, stride=1, padding=(smooth - 1) // 2)
            # Only compare positive attributions
            attributions = attributions.clamp(0)
            # Calculate the relative amount of attributions per region. Use avg_pool for simplicity.
            with torch.no_grad():
                contribs = F.avg_pool2d(attributions, single_shape, stride=single_shape).permute(0, 1, 3, 2).reshape(
                    attributions.shape[0], -1)
                total = contribs.sum(1, keepdim=True)
            contribs = to_numpy(torch.where(total * contribs > 0, contribs/total, torch.zeros_like(contribs)))
            metric.append([contrib[idx] for idx, contrib in enumerate(contribs)])
            print("{:>6.2f}% of processing complete".format(100*(count+1.)/sample_size), flush=True)
        result = np.array(metric).flatten()
        print("Percentiles of localisation accuracy (25, 50, 75, 100): ", np.percentile(result, [25, 50, 75, 100]))
        return {"localisation_metric": result}

    @staticmethod
    def make_multi_image(n_imgs, loader, offset=0, fixed_indices=None):
        """
        From the offset position takes the next n_imgs that are of different classes according to the order in the
        dataset or fixed_indices .
        Args:
            n_imgs: how many images should be combined for a multi images
            loader: data loader
            offset: current offset
            fixed_indices: whether or not to use pre-defined indices (e.g., first ordering images by confidence).

        Returns: the multi_image, the targets in the multi_image and the new offset


        """
        assert n_imgs in [4, 9]
        tgts = []
        imgs = []
        count = 0
        i = 0
        if fixed_indices is not None:
            mapper = fixed_indices
        else:
            mapper = list(range(len(loader.dataset)))

        # Going through the dataset to sample images
        while count < n_imgs:
            img, tgt = loader.dataset[mapper[i + offset]]
            i += 1
            tgt_idx = tgt.argmax().item()
            # if the class of the new image is already added to the list of images for the multi-image, skip this image
            # This should actually not happen since the indices are sorted in blocks of 9 unique labels
            if tgt_idx in tgts:
                continue
            imgs.append(img[None])
            tgts.append(tgt_idx)
            count += 1
        img = torch.cat(imgs, dim=0)
        img = img.view(-1, int(np.sqrt(n_imgs)), int(np.sqrt(n_imgs)), *img.shape[-3:]).permute(0, 3, 2, 4, 1, 5).reshape(
            -1, img.shape[1], img.shape[2] * int(np.sqrt(n_imgs)), img.shape[3] * int(np.sqrt(n_imgs)))
        return img.cuda(), tgts, i + offset + 1


def argument_parser():
    """
    Create a parser with run_experiments arguments.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="Localisation metric analyser.")
    parser.add_argument("--save_path", default=None, help="Path for model checkpoints.")
    parser.add_argument("--reload", default="last",
                        type=str, help="Which epoch to load. Options are 'last', 'best' and 'epoch_X',"
                                       "as long as epoch_X exists.")
    parser.add_argument("--explainer_name", default="Ours",
                        type=str, help="Which explainer method to use. Ours uses trainer.attribute.")
    parser.add_argument("--analysis_config", default="default_3x3",
                        type=str, help="Which analysis configuration file to load.")
    parser.add_argument("--explainer_config", default="default",
                        type=str, help="Which explainer configuration file to load.")
    parser.add_argument("--batch_size", default=1,
                        type=int, help="Batch size for the data loader.")
    parser.add_argument("--smooth", default=15,
                        type=int, help="Determines by how much the attribution maps are smoothed (avg_pool).")
    return parser


def get_arguments():
    parser = argument_parser()
    opts = parser.parse_args()
    return opts


def main(config):

    trainer = load_trainer(config.save_path, config.reload, batch_size=config.batch_size)

    analyser = LocalisationAnalyser(trainer, config.analysis_config, explainer_name=config.explainer_name,
                                    explainer_config=config.explainer_config, smooth=config.smooth)
    analyser.run()


if __name__ == "__main__":

    params = get_arguments()
    main(params)

