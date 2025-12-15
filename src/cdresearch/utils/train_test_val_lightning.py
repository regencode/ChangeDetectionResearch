import torch
from tqdm import tqdm
from .metrics import calculate_metrics
from .display import display_during_inference
from .metrics_lightning import BinarySegmentationMetrics
import pytorch_lightning as pl
from torch import optim
from torch.optim import lr_scheduler


class ChangeDetectionModel(pl.LightningModule):
    def __init__(self, model, loss_fn,
                 optim_kwargs={"lr" : 1e-2},
                 scheduler_kwargs=None):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optim_kwargs = optim_kwargs
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.val_metrics = BinarySegmentationMetrics()
        self.test_metrics = BinarySegmentationMetrics()
        self.save_hyperparameters(ignore=["model", "loss_fn"])

    def forward(self, X_batch):
        return self.model(X_batch)

    def on_train_epoch_start(self):
        opt = self.optimizers()
        self.log("lr", opt.param_groups[0]["lr"], on_epoch=True)

    def process_batch(self, batch):
        x, y = batch
        x = (( x / 255.0) * 2) - 1
        x = x.permute(1, 0, 2, 3, 4)
        return x, y

    def training_step(self, batch, batch_idx):
        x, y = self.process_batch(batch)
        logits = self(x[0], x[1])
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self.process_batch(batch)
        logits = self(x[0], x[1])
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.val_metrics.update(logits, y)

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        self.log_dict(
            {f"val_{k}": v for k, v in metrics.items()},
            prog_bar=True
        )
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        x, y = self.process_batch(batch)
        logits = self(x[0], x[1])
        loss = self.loss_fn(logits, y)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        self.test_metrics.update(logits, y)

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        self.log_dict(
            {f"val_{k}": v for k, v in metrics.items()},
            prog_bar=True
        )
        self.test_metrics.reset()


    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), **self.optim_kwargs)
        if self.scheduler_kwargs:
            scheduler = lr_scheduler.LinearLR(optimizer, **self.scheduler_kwargs)
            return {
                "optimizer" : optimizer,
                "lr_scheduler" : scheduler
            }
        return optimizer

