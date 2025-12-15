import torch
import pytorch_lightning as pl

class BinarySegmentationMetrics(pl.LightningModule):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

        self.register_buffer("TP", torch.tensor(0.))
        self.register_buffer("TN", torch.tensor(0.))
        self.register_buffer("FP", torch.tensor(0.))
        self.register_buffer("FN", torch.tensor(0.))

    def update(self, logits, y_true):
        """
        y_true: [B, H, W] (0/1)
        logits: [B, C, H, W]
        """
        preds = torch.argmax(logits, dim=1) # [B, H, W]

        y_true = y_true.bool()
        preds = preds.bool()

        self.TP += torch.sum((y_true == 1) & (preds == 1))
        self.TN += torch.sum((y_true == 0) & (preds == 0))
        self.FP += torch.sum((y_true == 0) & (preds == 1))
        self.FN += torch.sum((y_true == 1) & (preds == 0))

    def compute(self):
        TP, TN, FP, FN = self.TP, self.TN, self.FP, self.FN
        eps = self.eps

        accuracy  = (TP + TN) / (TP + TN + FP + FN + eps)
        precision = TP / (TP + FP + eps)
        recall    = TP / (TP + FN + eps)
        f1        = 2 * precision * recall / (precision + recall + eps)
        iou       = TP / (TP + FP + FN + eps)

        return {
            "acc": accuracy,
            "prec": precision,
            "rec": recall,
            "f1": f1,
            "iou": iou,
        }

    def reset(self):
        self.TP.zero_()
        self.TN.zero_()
        self.FP.zero_()
        self.FN.zero_()

