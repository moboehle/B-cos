import numpy as np
import torch

from interpretability.explanation_methods.utils import ExplainerBase, limit_n_images


class Occlusion(ExplainerBase):

    def __init__(self, trainer, stride=32, ks=32, batch_size=8, only_positive=False):
        super().__init__(trainer)
        self.masks = None
        self.participated = None
        self.n_part = None
        self.max_imgs_bs = 1  # Process images individually.
        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride
        self.ks = ks
        self.batch_size = batch_size
        self.only_positive = only_positive

    def make_masks(self, img):
        """
        Uses the input image to compute the masks that should be applied to an image of that size.
        The masks are then saved and used for future images.
        Args:
            img: image to be explained

        Returns: None

        """
        stride = self.stride
        ks = self.ks
        total = img.shape[-1] // stride[-1] * img.shape[-2] // stride[-2]
        strided_shape = (np.array(img.shape[-2:]) / np.array(stride)).astype(int)
        if ks % 2 == 1:
            ks2 = (ks - 1) // 2
            off = 0
        else:
            ks2 = ks
            off = 1
        occlusion_masks = []
        for idx in range(total):
            mask = torch.ones(img.shape[-2:])
            wpos, hpos = np.unravel_index(idx, shape=strided_shape)
            mask[max(0, (hpos+off) * stride[0] - ks2): hpos * stride[0] + ks2,
            max(0, (wpos + off) * stride[1] - ks2): wpos * stride[1] + ks2] = 0
            occlusion_masks.append(mask)
        self.masks = torch.stack(occlusion_masks, dim=0)[:, None].cpu()
        # for each mask, compute which location participated in the occlusion
        # shape of participated is (n_masks, h, w)
        participated = (self.masks - 1).abs()[:, 0]
        # calculate how many positions participated in any given occlusion mask
        # Shape of n_part is n_masks, 1, 1
        n_part = (participated.view(self.masks.shape[0], -1).sum(1))[:, None, None, None]
        self.participated = participated[:, None]
        self.n_part = n_part

    @limit_n_images  # Processes images individually.
    @torch.no_grad()
    def attribute(self, img, target, return_all=False):
        self.trainer.model.zero_grad()
        batch_size = self.batch_size
        stride = self.stride
        assert ((np.array(img.shape[-2:]) % np.array(stride)) == 0).all()
        org_out = self.trainer(img).cpu()  # (1, n_classes). This is the prediction on the unperturbed input
        img = img.cpu()
        if self.masks is None or self.masks.shape[-1] != img.shape[-1]:
            self.make_masks(img)
        masks = self.masks.cpu()

        # Assuming img has single batch example len(img)==1
        # Compute masked images
        masked_input = img * masks
        # Evaluate masked images and store the prediction of this perturbed input
        pert_out = torch.cat([self.trainer(masked_input[idx * batch_size: (idx+1) * batch_size].cuda()).cpu()
                              for idx in range(int(np.ceil(len(masked_input)/batch_size)))], dim=0)

        # Compute the difference between the original input and the masked inputs
        diff = (org_out - pert_out).clamp(0) if self.only_positive else (org_out - pert_out)
        diff = diff[:, :, None, None]

        # Calculate the influence per location per mask. In particular, all those that participated get attributed
        # 1/n_part of the change in output
        influence = (self.participated * diff) / self.n_part

        # The sum over dimension zero sums the influence per position over all masks that a location was used in.
        if return_all:
            return influence.sum(0, keepdim=True).cuda()

        return influence.sum(0, keepdim=True)[:, int(target)][:, None].cuda()

    def attribute_selection(self, img, targets):
        return self.attribute(img, targets, return_all=True)[0, targets][:, None]
