import torch
from captum.attr import IntegratedGradients, GuidedBackprop, InputXGradient, Saliency, LayerGradCam, DeepLift
from torch import nn as nn
from torch.autograd import Variable

from data.data_transforms import AddInverse
from interpretability.explanation_methods.utils import CaptumDerivative
import torch.nn.functional as F

class IntGrad(CaptumDerivative, IntegratedGradients):

    def __init__(self, trainer, n_steps=20, internal_batch_size=1):
        CaptumDerivative.__init__(self, trainer, n_steps=n_steps, internal_batch_size=internal_batch_size)
        IntegratedGradients.__init__(self, self.trainer)


class GB(CaptumDerivative, GuidedBackprop):

    def __init__(self, trainer):
        CaptumDerivative.__init__(self, trainer)
        GuidedBackprop.__init__(self, self.trainer)


class IxG(CaptumDerivative, InputXGradient):

    def __init__(self, trainer):
        CaptumDerivative.__init__(self, trainer)
        InputXGradient.__init__(self, self.trainer)


class Grad(CaptumDerivative, Saliency):

    def __init__(self, trainer):
        CaptumDerivative.__init__(self, trainer)
        self.configs.update({"abs": False})
        Saliency.__init__(self, self.trainer)


class GradCam(CaptumDerivative):

    def __init__(self, trainer, add_inverse=True, interpolate_mode="nearest"):
        model = trainer.model
        CaptumDerivative.__init__(self, trainer)
        assert hasattr(trainer.model[0], "get_features")
        self.configs.update({
            "relu_attributions": True,
            "interpolate_mode": interpolate_mode,
            "add_inverse": add_inverse,
                             })  # As in original GradCam paper
        seq_model = model[0].get_sequential_model()
        self.features = seq_model[:-1]
        self.classifier = seq_model[-1]

    def attribute(self, img, target, **kwargs):
        if self.configs["add_inverse"]:
            img = AddInverse()(img)
        var_features = Variable(self.features(img), requires_grad=True)
        out = F.adaptive_avg_pool2d(self.classifier(var_features), 1)[..., 0, 0]
        out[0, target].backward()
        att = (var_features.grad.sum(dim=(-2, -1), keepdim=True) * var_features).sum(1, keepdim=True)
        if self.configs["relu_attributions"]:
            att.relu_()
        return LayerGradCam.interpolate(att, img.shape[-2:], interpolate_mode=self.configs["interpolate_mode"])


class DeepLIFT(CaptumDerivative, DeepLift):

    def __init__(self, trainer):
        CaptumDerivative.__init__(self, trainer)
        DeepLift.__init__(self, self.trainer)
