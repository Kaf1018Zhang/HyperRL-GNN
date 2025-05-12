import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from utils.evaluator import Evaluator
import torch.nn as nn
import torch.nn.functional as F
import os

from modules.composable_blocks import (
    GCNEncoder, GATEncoder, SAGEEncoder,
    pooling_mean, pooling_max, GlobalAttentionPooling,
    LinearReadout, MLPReadout, TransformerReadout
)
from datasets.loader_factory import load_dataset

# a constructor like train_rl_controller.py 
def build_model(action_dict, in_channels, out_channels):
    # Just copy the same; or import from a public module
    from modules.composable_blocks import GlobalAttentionPooling, TransformerReadout

    encoder_type = action_dict["encoder"]
    hidden_dim   = action_dict["hidden_dim"]
    bn_on        = action_dict["bn"]
    if encoder_type == "GCN":
        encoder = GCNEncoder(in_channels, hidden_dim, with_bn=bn_on)
    elif encoder_type == "GAT":
        encoder = GATEncoder(in_channels, hidden_dim, with_bn=bn_on)
    else:
        encoder = SAGEEncoder(in_channels, hidden_dim, with_bn=bn_on)

    pool_type = action_dict["pooling"]
    if pool_type == "mean":
        pooling_fn = pooling_mean
    elif pool_type == "max":
        pooling_fn = pooling_max
    else:
        pooling_fn = GlobalAttentionPooling(hidden_dim)

    readout_type = action_dict["readout"]
    if readout_type == "linear":
        readout = LinearReadout(hidden_dim, out_channels)
    elif readout_type == "mlp":
        readout = MLPReadout(hidden_dim, out_channels, hidden_dim=hidden_dim)
    else:
        readout = TransformerReadout(d_model=hidden_dim, out_dim=out_channels)

    dropout_rate = action_dict["dropout"]

    class ComposedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.pooling_fn = pooling_fn
            self.readout = readout
            self.dropout = dropout_rate

        def forward(self, x, edge_index, batch):
            x = self.encoder(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            if isinstance(self.readout, TransformerReadout):
                out = self.readout(x, batch)
            else:
                if isinstance(self.pooling_fn, nn.Module):
                    x_pooled = self.pooling_fn(x, batch)
                else:
                    x_pooled = self.pooling_fn(x, batch)
                out = self.readout(x_pooled)
            return out

    return ComposedModel()


def deploy_best_strategy():
    if not os.path.exists("best_strategy.json"):
        print("[Deploy] Error: best_strategy.json not found. Please run train_rl_controller.py first.")
        return

    with open("best_strategy.json", "r") as f:
        record = json.load(f)

    dataset_name = record["dataset_name"]
    best_actions = record["best_actions"]
    print(f"[Deploy] Loaded best strategy for {dataset_name}: {best_actions}")

    # load train/val/test
    train_data, val_data, test_data = load_dataset(dataset_name)

    # combine train+val => for final train
    from torch.utils.data import ConcatDataset
    train_val_dataset = ConcatDataset([train_data, val_data])

    # DataLoader
    train_loader = DataLoader(train_val_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)

    # in_channels/out_channels
    in_channels  = train_data.num_node_features
    out_channels = train_data.num_classes

    model = build_model(best_actions, in_channels, out_channels)
    module = Evaluator(model, strategy_name="Deployed", lr=best_actions["lr"])

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=5
    )

    print("[Deploy] Training on (train+val) with best synergy ...")
    trainer.fit(module, train_loader)

    print("[Deploy] Testing on test set ...")
    results = trainer.test(module, test_loader)
    print(f"[Deploy] Final test results => {results}")


if __name__ == "__main__":
    deploy_best_strategy()
