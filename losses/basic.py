import torch
from torch.nn import Module
from torch.nn.functional import binary_cross_entropy
from torch import nn


class L1Loss(Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.eps = 1e-6

    def forward(self, pred: torch.Tensor, gt, train_mask):
        return (torch.abs(pred - gt) * train_mask).sum() / (train_mask.sum() + self.eps)


class SmoothL1Loss(Module):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()
        self.eps = 1e-6

    def forward(self, pred, gt, train_mask):
        diff = torch.abs(pred - gt)
        great_than_1 = diff > 1
        diff[great_than_1] = diff[great_than_1] - 0.5
        diff[~great_than_1] = 0.5 * torch.pow(diff[~great_than_1], 2)
        return (diff * train_mask).sum() / (train_mask.sum() + self.eps)


class DiceLoss(Module):
    """Dice = 1 - F1Score
    minimal dice equal to maximize F1Score"""

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.eps = 1e-4

    def forward(self, pred, gt, mask, weights=None):
        if weights:
            mask *= weights
        intersection = (pred * gt * mask).sum()

        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        return 1 - 2 * intersection / union


class OHEMLoss(Module):
    def __init__(self, ohem: bool = True):
        super(OHEMLoss, self).__init__()
        self.negative_ratio = 3.0
        self.eps = 1e-4
        self.ohem = ohem

    def forward(self,
                pred: torch.Tensor,
                gt: torch.Tensor,
                train_mask: torch.Tensor):
        """
        Args:
            pred: shape :math:`(N, 1, H, W)`, the prediction of network
            gt: shape :math:`(N, H, W)`, the target
            train_mask: shape :math:`(N, H, W)`, the mask indicates positive regions
        """
        loss = binary_cross_entropy(pred, gt, reduction="none")

        positive = (gt * train_mask).byte()
        positive_loss = loss * positive.float()
        positive_count = positive.float().sum()

        negative = ((1 - gt) * train_mask).byte()
        negative_count = int(negative.float().sum())
        if self.ohem:
            negative_count = min(negative_count, int(positive_count * self.negative_ratio))
        negative_loss = loss * negative.float()
        negative_loss, _ = negative_loss.view(-1).topk(negative_count)

        return (positive_loss.sum() + negative_loss.sum()) / (positive_count + negative_count + self.eps)


if __name__ == '__main__':
    import numpy as np

    np.random.seed(123456)
    loss = OHEMLoss()
    pred1 = torch.from_numpy(np.random.uniform(0, 1, (3, 320, 320)).astype(float))
    gt1 = torch.from_numpy(np.random.randint(0, 2, (3, 320, 320)).astype(float))
    mask1 = torch.ones((3, 320, 320))
    print(loss(pred1, gt1, mask1))
