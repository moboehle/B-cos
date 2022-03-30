import torch.nn.functional as F
from torch import nn
import numpy as np


class NormedConv2d(nn.Conv2d):
    """
    Standard 2D convolution, but with unit norm weights.
    """

    def forward(self, in_tensor):
        shape = self.weight.shape
        w = self.weight.view(shape[0], -1)
        w = w/(w.norm(p=2, dim=1, keepdim=True))
        return F.conv2d(in_tensor, w.view(shape),
                        self.bias, self.stride, self.padding, self.dilation, self.groups)


class BcosConv2d(nn.Module):

    def __init__(self, inc, outc, kernel_size=1, stride=1, padding=0, max_out=2, b=2,
                 scale=None, scale_fact=100, **kwargs):
        super().__init__()
        
        ks = kernel_size
        self.stride = stride
        self.linear = NormedConv2d(inc, outc * max_out, ks, stride, padding, 1, 1, bias=False)
        self.outc = outc * max_out
        self.b = b
        self.max_out = max_out
        self.inc = inc
        self.kernel_size = ks
        self.kssq = ks**2 if not isinstance(ks, tuple) else np.prod(ks)
        self.padding = padding
        self.detach = False
        if scale is None:
            ks_scale = ks if not isinstance(ks, tuple) else np.sqrt(np.prod(ks))
            self.scale = (ks_scale * np.sqrt(self.inc)) / scale_fact
        else:
            self.scale = scale

    def forward(self, in_tensor):
        """
        In the case of B=2, we do not have to explicitly calculate the cosine term.
        Args:
            in_tensor: Input tensor. Expected shape: (B, C, H, W)

        Returns:
            BcosConv2d output on the input tensor.
        """
        if self.b == 2:
            return self.fwd_2(in_tensor)
        return self.fwd_b(in_tensor)

    def explanation_mode(self, detach=True):
        """
        Enter 'explanation mode' by setting self.explain and self.detach.
        Args:
            detach: Whether to 'detach' the weight matrix from the computational graph so that it is not
                            taken into account in the backward pass.

        Returns: None

        """
        self.detach = detach

    def fwd_b(self, in_tensor):
        # Simple linear layer
        out = self.linear(in_tensor)
        bs, _, h, w = out.shape

        # MaxOut computation
        if self.max_out > 1:
            bs, _, h, w = out.shape
            out = out.view(bs, -1, self.max_out, h, w)
            out = out.max(dim=2, keepdim=False)[0]

        # If B=1, no further calculation necessary.
        if self.b == 1:
            return out / self.scale

        # Calculating the norm of input patches. Use average pooling and upscale by kernel size.
        norm = (F.avg_pool2d((in_tensor ** 2).sum(1, keepdim=True), self.kernel_size, padding=self.padding,
                             stride=self.stride) * self.kssq + 1e-6  # stabilising term
                ).sqrt_()

        # get absolute value of cos
        # TODO: unnecessary doubling of stabilising term. Only affects CIFAR10 experiments in the paper.
        abs_cos = (out / norm).abs() + 1e-6

        # In order to compute the explanations, we detach the dynamically calculated scaling from the graph.
        if self.detach:
            abs_cos = abs_cos.detach()

        # additional factor of cos^(b-1) s.t. in total we have norm * cos^b with original sign
        out = out * abs_cos.pow(self.b-1)
        return out / self.scale

    def fwd_2(self, in_tensor):
        # Simple linear layer
        out = self.linear(in_tensor)

        # MaxOut computation
        if self.max_out > 1:
            bs, _, h, w = out.shape
            out = out.view(bs, -1, self.max_out, h, w)
            out = out.max(dim=2, keepdim=False)[0]

        # Calculating the norm of input patches. Use average pooling and upscale by kernel size.
        # TODO: implement directly as F.sum_pool2d...
        norm = (F.avg_pool2d((in_tensor ** 2).sum(1, keepdim=True), self.kernel_size, padding=self.padding,
                                    stride=self.stride) * self.kssq + 1e-6  # stabilising term
                ).sqrt_()

        # In order to compute the explanations, we detach the dynamically calculated scaling from the graph.
        if self.detach:
            out = (out * out.abs().detach())
            norm = norm.detach()
        else:
            out = (out * out.abs())

        return out / (norm * self.scale)
