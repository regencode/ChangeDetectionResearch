import torch
from tqdm import tqdm
from .metrics import calculate_metrics
from .display import display_during_inference
from .metrics_lightning import BinarySegmentationMetrics
import pytorch_lightning as pl
from torch import optim
from torch.optim import lr_scheduler
import einops as ein


class ChangeDetectionModel(pl.LightningModule):
    def __init__(self, model, loss_fn,
                 optim_kwargs={"optim" : "sgd", "lr" : 1e-2},
                 scheduler_kwargs=None):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim_kwargs.pop("optim")
        self.optim_kwargs = optim_kwargs
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.val_metrics = BinarySegmentationMetrics()
        self.test_metrics = BinarySegmentationMetrics()
        self.save_hyperparameters(ignore=["model", "loss_fn"])

    def forward(self, X_batch):
        return self.model(X_batch[0], X_batch[1])

    def on_train_epoch_start(self):
        opt = self.optimizers()
        self.log("lr", opt.param_groups[0]["lr"], on_epoch=True)
        self._running_loss = 0.0
        self._num_steps = 0

    def process_batch(self, batch):
        x, y = batch
        x = (( x / 255.0) * 2) - 1
        x = x.permute(1, 0, 2, 3, 4)
        return x, y

    def training_step(self, batch, batch_idx):
        x, y = self.process_batch(batch)
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self._running_loss += loss.item()
        self._num_steps += 1
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log("running_train_loss", self._running_loss / self._num_steps, prog_bar=True, on_epoch=False, on_step=True)
        return loss

    def on_validation_epoch_start(self):
        self._running_loss = 0.0
        self._num_steps = 0


    def validation_step(self, batch, batch_idx):
        x, y = self.process_batch(batch)
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self._running_loss += loss.item()
        self._num_steps += 1
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log("running_val_loss", self._running_loss / self._num_steps, prog_bar=True, on_epoch=False, on_step=True)
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
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        self.test_metrics.update(logits, y)
        if batch_idx % 5 == 0:
            x = ein.rearrange(x, "p n c h w -> n p c h w")
            x = ((x + 1.0)*0.5)
            display_during_inference(x, y, logits)

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        self.log_dict(
            {f"test_{k}": v for k, v in metrics.items()},
            prog_bar=True
        )
        self.test_metrics.reset()


    def configure_optimizers(self):
        if self.optim.lower() == "sgd":
            optimizer = optim.SGD(self.parameters(), **self.optim_kwargs)
        elif self.optim.lower() == "adam":
            optimizer = optim.Adam(self.parameters(), **self.optim_kwargs)
        elif self.optim.lower() == "adamw":
            optimizer = optim.AdamW(self.parameters(), **self.optim_kwargs)
        else:
            assert AssertionError("self.optim_kwargs.optim must be either sgd, adam or adaw")
            return

        print(f"using optimizer: {optimizer}")
        if self.scheduler_kwargs:
            scheduler = lr_scheduler.LinearLR(optimizer, **self.scheduler_kwargs)
            return {
                "optimizer" : optimizer,
                "lr_scheduler" : scheduler
            }
        return optimizer

