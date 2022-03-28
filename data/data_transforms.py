import torch
from torchvision import transforms
import torch.nn as nn


class OneHot:
    def __init__(self, num_classes):
        """
        This is a target transform function to convert a single index into a one-hot encoding.
        Args:
            num_classes: total number of classes
        """
        self.n = num_classes

    def __call__(self, target):
        one_hot = torch.zeros(self.n, dtype=torch.float)
        one_hot[target] = 1
        return one_hot


class BatchOneHot:
    def __init__(self, num_classes):
        """
        This implements one hot encoding for a list of target indices.
        Args:
            num_classes: total number of classes.
        """
        self.n = num_classes

    def __call__(self, target):
        one_hot = torch.zeros(target.shape[0], self.n, dtype=torch.float)
        for b_idx in range(len(target)):
            one_hot[b_idx, target[b_idx]] = 1
        return one_hot


class AddInverse(nn.Module):

    def __init__(self, dim=1):
        """
            Adds (1-in_tensor) as additional channels to its input via torch.cat().
            Can be used for images to give all spatial locations the same sum over the channels to reduce color bias.
        """
        super().__init__()
        self.dim = dim

    def forward(self, in_tensor):
        out = torch.cat([in_tensor, 1-in_tensor], self.dim)
        return out


class MyToTensor(transforms.ToTensor):

    def __init__(self):
        """
        Same as transforms.ToTensor, except that if input to __call__ is already tensor, the input is returned unchanged
        """
        super().__init__()

    def __call__(self, input_img):
        if not isinstance(input_img, torch.Tensor):
            return super().__call__(input_img)
        return input_img


class NoTransform(nn.Module):
    """A 'do nothing' transform to be used when no transform is needed."""
    def __call__(self, in_tensors):
        return in_tensors
