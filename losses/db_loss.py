import torch
from torch import nn
from losses.basic import OHEMLoss, L1Loss, DiceLoss


class DBLoss(nn.Module):
    def __init__(self, k=50, alpha=1.0, beta=10):
        super(DBLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.score_loss = OHEMLoss()  # Ls
        self.f1_loss = DiceLoss()  # Lb
        # self.f1_loss = OHEMLoss(ohem=False)  # Lb
        self.threshold_loss = L1Loss()  # Lt

    def forward(self, pred, shrunk_label, threshold_label, train_mask):
        pred_scores = pred[:, 0, ...]
        pred_threshold = pred[:, 1, ...]
        pred_binary = torch.sigmoid(self.k * (pred_scores - pred_threshold))
        loss_score = self.score_loss(pred_scores, shrunk_label, train_mask)
        loss_f1 = self.f1_loss(pred_binary, shrunk_label, train_mask)
        loss_threshold = self.threshold_loss(pred_threshold, threshold_label, train_mask)
        loss_final = self.alpha * loss_score + self.alpha * loss_f1 + self.beta * loss_threshold

        return dict(
            synth=loss_final,
            score=loss_score,
            binary=loss_f1,
            threshold=loss_threshold
        )


if __name__ == '__main__':
    import torch
    import numpy as np

    np.random.seed(123456)
    loss = DBLoss()
    pred = torch.from_numpy(np.random.uniform(0, 1, (3, 3, 320, 320)).astype(float))
    gt = torch.from_numpy(np.random.randint(0, 2, (3, 320, 320)).astype(float))
    mask = torch.ones((3, 320, 320))
    loss_info = loss(pred, gt, gt, mask)
    for key in loss_info:
        print(key, ": ", loss_info[key])
