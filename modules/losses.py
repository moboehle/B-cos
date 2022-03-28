from torch import nn
from torch.nn.parallel import DistributedDataParallel

from collections import Iterable


class LossDict(dict):

    def __init__(self):
        """
        A dictionary with an additional 'collect' method to add up all the individual losses.
        This is used for tracking the losses easily by name.
        """
        super().__init__()

    def collect(self):
        loss = 0
        for loss_key, loss_value in self.items():
            if "loss" in loss_key.lower():
                loss += loss_value
        return loss


class CombinedLosses:

    def __init__(self, *losses, eval_losses=tuple(["CLS-Loss"])):
        """
        Evaluates all losses passed during initialisation when called by the trainer.
        Args:
            *losses: Losses to evaluate.
            eval_losses: During evaluation, we might not be interested in tracking all losses, especially
                if it makes the evaluation slower (e.g., matrix regularisation).
                eval_losses should be iterable.
        """
        self.losses = losses
        self.eval_losses = eval_losses
        assert isinstance(eval_losses, Iterable), "eval_losses needs to be an Iterable."

    def __call__(self, trainer, model_out, img, tgt, filtered=False):
        result_dict = LossDict()
        for loss in self.losses:
            if filtered and loss.get_alias() not in self.eval_losses:
                continue
            result_dict.update(loss.compute(trainer, model_out, img, tgt))
        return result_dict

    def __str__(self):
        obj_str = "CombinedLosses("
        obj_str += "\n\t".join([obj_str, *[str(loss) for loss in self.losses]])
        return "\n".join([obj_str, "{:>14s})".format(" ")])


class BaseLoss:

    def __str__(self):
        attributes = self.__dict__
        return str(self.__class__.__name__) + "(" + ", ".join(["{k}={v}".format(k=k, v=v) for k, v in attributes.items()
                                                               if not k.startswith("_")]) + ")"

    def get_alias(self):
        return str(self.__class__.__name__)

    def evaluate(self, trainer, model_out, img, tgt, **kwargs):
        raise NotImplementedError("This function needs to be implemented for all losses.")

    def compute(self, trainer, model_out, img, tgt):
        return {self.get_alias(): self.evaluate(trainer, model_out, img, tgt)}

    def __repr__(self):
        return self.__str__()


class LogitsBCE(nn.BCEWithLogitsLoss, BaseLoss):

    def __init__(self, reduction="mean"):
        super().__init__(reduction=reduction)

    def get_alias(self):
        return "CLS-Loss"

    def evaluate(self, trainer, model_out, img, tgt, **kwargs):
        return super().forward(model_out, tgt)


class LogitsCE(nn.CrossEntropyLoss, BaseLoss):

    def __init__(self):
        super().__init__()

    def get_alias(self):
        return "CLS-Loss"

    def evaluate(self, trainer, model_out, img, tgt, **kwargs):
        return super().forward(model_out, tgt.argmax(1))


class AuxLoss(nn.BCEWithLogitsLoss, BaseLoss):

    def __init__(self, w=.5):
        super().__init__()
        self.w = w

    def get_alias(self):
        return "AuxLoss"

    def evaluate(self, trainer, model_out, img, tgt, **kwargs):
        the_model = trainer.model if not isinstance(trainer.model, DistributedDataParallel) else trainer.model.module
        return self.w * super().forward(the_model[0].aux_out, tgt)
