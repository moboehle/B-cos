import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class AdaptiveSumPool2d(nn.AdaptiveAvgPool2d):

    def __init__(self, output_size):
        """
        Same as AdaptiveAvgPool2d, only that the normalisation is undone.
        Args:
            output_size: Adaptive size of the output.
        """
        super().__init__(output_size)
        self.shape = None

    def forward(self, in_tensor):
        self.shape = in_tensor.shape[-2:]
        return super().forward(in_tensor) * np.prod(self.shape)


class MyAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):

    def __init__(self, output_size):
        """
        Same as AdaptiveAvgPool2d, with saving the shape for matrix upscaling.
        Args:
            output_size: Adaptive size of the output.
        """
        super().__init__(output_size)
        self.shape = None

    def forward(self, in_tensor):
        self.shape = in_tensor.shape[-2:]
        return super().forward(in_tensor)


class TrainerAsModule(nn.Module):

    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer
        self.eval()

    def forward(self, in_tensor):
        return self.trainer(in_tensor)


class Normalise(nn.Module):

    def __init__(self):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406])[None, :, None, None], requires_grad=False)
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225])[None, :, None, None], requires_grad=False)

    def forward(self, in_tensor):
        return (in_tensor-self.mean)/self.std


class FinalLayer(nn.Module):

    def __init__(self, norm=1, bias=-5):
        """
        Used to add a bias and a temperature scaling to the final output of a particular model.
        Args:
            norm: inverse temperature, i.e., T^{-1}
            bias: constant shift in logit space for all classes.
        """
        super().__init__()
        assert norm != 0, "Norm 0 means average pooling in the last layer of the old trainer. " \
                          "Please add size.prod() of final layer as img_size_norm to exp_params."
        self.norm = norm
        self.bias = bias

    def forward(self, in_tensor):
        out = (in_tensor.view(*in_tensor.shape[:2])/self.norm + self.bias)
        return out
