import os
from os.path import join

import numpy as np

from training.training_utils import load_trainer_from_path


def load_trainer(save_path, reload="best", batch_size=1):
    trainer = load_trainer_from_path(save_path, only_test_loader=True, batch_size=batch_size)
    trainer.reload(verbose=True)
    if reload == "best":
        trainer.load_best_epoch()
    elif reload == "last":
        pass  # Already loaded by default
    elif reload.startswith("epoch_"):
        trainer.load_epoch(int(reload[len("epoch_"):]))
    else:
        raise NotImplementedError("This reload option is not defined.", reload)
    trainer.model.cuda()
    trainer.model.eval()
    return trainer


class Analyser:

    default_config = {}

    def __init__(self, trainer, **config):
        self.trainer = trainer
        for k, v in self.default_config.items():
            if k not in config:
                config[k] = v
        self.config = config
        self.results = None

    def analysis(self):
        raise NotImplementedError("Need to implement analysis function.")

    def run(self):
        results = self.analysis()
        self.save_results(results)

    def save_results(self, results):
        save_path = join(self.trainer.save_path,  self.get_save_folder())
        os.makedirs(save_path, exist_ok=True)
        for k, v in results.items():
            np.savetxt(join(save_path, "{}.np".format(k)), v)

        with open(join(save_path, "config.log"), "w") as file:
            for k, v in self.get_config().items():
                k_v_str = "{k}: {v}".format(k=k, v=v)
                print(k_v_str)
                file.writelines([k_v_str, "\n"])

    def get_save_folder(self, epoch=None):
        raise NotImplementedError("Need to implement get_save_folder function.")

    def get_config(self):
        config = self.config
        config.update({"epoch": self.trainer.epoch})
        return config

    def load_results(self, epoch=None):
        save_path = join(self.trainer.save_path,  self.get_save_folder(epoch))
        # print("Trying to load results from", save_path)
        if not os.path.exists(save_path):
            return
        results = dict()
        files = [f for f in os.listdir(save_path) if f.endswith(".np")]
        for file in files:
            results[file[:-3]] = np.loadtxt(join(save_path, file))
        self.results = results
