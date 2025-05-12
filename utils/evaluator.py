# utils/evaluator.py

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

class Evaluator(pl.LightningModule):
    def __init__(self, model, strategy_name='unknown', lr=0.001):
        super().__init__()
        self.model = model
        self.strategy_name = strategy_name
        self.lr = lr

    def forward(self, x, edge_index, batch):
        return self.model(x, edge_index, batch)

    def training_step(self, batch, batch_idx):
        x, edge_index, y, b = batch.x, batch.edge_index, batch.y, batch.batch
        logits = self.forward(x, edge_index, b)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, edge_index, y, b = batch.x, batch.edge_index, batch.y, batch.batch
        logits = self.forward(x, edge_index, b)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return acc

    def test_step(self, batch, batch_idx):
        x, edge_index, y, b = batch.x, batch.edge_index, batch.y, batch.batch
        logits = self.forward(x, edge_index, b)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return acc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
