# experiments/train_rl_controller.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pytorch_lightning as pl
import numpy as np
import json
import matplotlib.pyplot as plt

from modules.rl_controller import MultiHeadRLController  # Use the controller that supports structure embedding + Gumbel‑Softmax
from modules.composable_blocks import (
    GCNEncoder, GATEncoder, SAGEEncoder,
    pooling_mean, pooling_max, GlobalAttentionPooling,
    LinearReadout, MLPReadout, TransformerReadout,
    raw_features, spectral_features, virtual_node_features
)
from datasets.loader_factory import load_dataset
from utils.evaluator import Evaluator
from torch_geometric.loader import DataLoader
from core.reward_utils import compute_reward
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning.callbacks import EarlyStopping

###################################
# 0) Visualization Collection
###################################
os.makedirs("vis", exist_ok=True)

###################################
# 1) Discrete Space (encoder/pooling/readout/augment/hidden_dims/lr/...)
###################################
encoder_opts = ["GCN", "GAT", "GraphSAGE"]
pooling_opts = ["mean", "max", "attention"]
readout_opts = ["linear", "mlp", "transformer"]
augment_opts = ["raw", "spectral", "virtual"]
hidden_dim_opts = [32, 64]
dropout_opts = [0.0, 0.5]
lr_opts = [1e-3, 1e-4]
bn_opts = [False, True]
temp_opts = [0.5, 1.0]

###################################
# 2) Record train/val
###################################
class PlotLossAccCallback(pl.Callback):
    def __init__(self, episode_idx):
        super().__init__()
        self.episode_idx = episode_idx
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get("train_loss")
        train_acc = trainer.callback_metrics.get("train_acc")
        if train_loss is not None:
            self.train_losses.append(train_loss.cpu().item())
        if train_acc is not None:
            self.train_accs.append(train_acc.cpu().item())

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")
        val_acc = trainer.callback_metrics.get("val_acc")
        if val_loss is not None:
            self.val_losses.append(val_loss.cpu().item())
        if val_acc is not None:
            self.val_accs.append(val_acc.cpu().item())

    def on_train_end(self, trainer, pl_module):
        epochs = range(len(self.train_losses))
        fig, ax1 = plt.subplots()
        ax1.set_title(f"Episode {self.episode_idx} Training/Validation")
        ax1.set_xlabel("Epoch")

        ax1.set_ylabel("Loss", color="red")
        ax1.plot(epochs, self.train_losses, label="Train Loss", color="red")
        if len(self.val_losses) == len(epochs):
            ax1.plot(epochs, self.val_losses, label="Val Loss", color="orange")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Accuracy", color="blue")
        ax2.plot(epochs, self.train_accs, label="Train Acc", color="blue")
        if len(self.val_accs) == len(epochs):
            ax2.plot(epochs, self.val_accs, label="Val Acc", color="green")

        ax2.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(f"vis/episode_{self.episode_idx}_curve.png")
        plt.close()

###################################
# 3) Build Composable Model
###################################

def build_model(action_dict, in_channels, out_channels):
    # 1) Encoder
    if action_dict["encoder"] == "GCN":
        encoder = GCNEncoder(in_channels, action_dict["hidden_dim"], with_bn=action_dict["bn"])
    elif action_dict["encoder"] == "GAT":
        encoder = GATEncoder(in_channels, action_dict["hidden_dim"], with_bn=action_dict["bn"])
    else:
        encoder = SAGEEncoder(in_channels, action_dict["hidden_dim"], with_bn=action_dict["bn"])

    # 2) Pooling
    if action_dict["pooling"] == "mean":
        pooling_fn = pooling_mean
    elif action_dict["pooling"] == "max":
        pooling_fn = pooling_max
    else:
        pooling_fn = GlobalAttentionPooling(action_dict["hidden_dim"])

    # 3) Readout
    if action_dict["readout"] == "linear":
        readout = LinearReadout(action_dict["hidden_dim"], out_channels)
    elif action_dict["readout"] == "mlp":
        readout = MLPReadout(action_dict["hidden_dim"], out_channels, hidden_dim=action_dict["hidden_dim"])
    else:
        readout = TransformerReadout(d_model=action_dict["hidden_dim"], out_dim=out_channels)

    class ComposedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.pooling_fn = pooling_fn
            self.readout = readout
            self.dropout = action_dict["dropout"]

        def forward(self, x, edge_index, batch):
            x = self.encoder(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Transformer vs Pool
            if isinstance(self.readout, TransformerReadout):
                out = self.readout(x, batch)  # CLS token Output graph by graph
            else:
                if isinstance(self.pooling_fn, nn.Module):
                    x_pooled = self.pooling_fn(x, batch)
                else:
                    x_pooled = self.pooling_fn(x, batch)
                out = self.readout(x_pooled)
            return out

    return ComposedModel()


def apply_feature_augment(data, augment_type):
    if augment_type == "raw":
        return raw_features(data)
    elif augment_type == "spectral":
        return spectral_features(data)
    else:
        return virtual_node_features(data)

###################################
# 4) Single Episode: train/val => val_acc => reward
###################################

def run_episode(dataset_name, controller, episode_idx, device="cuda"):
    # A) Sampling
    state = torch.tensor([[0.0]], dtype=torch.float, device=device)
    actions, log_prob = controller.sample_actions(state)
    action_dict = controller.parse_actions(actions)
    print(f"[RL] Episode {episode_idx} => Sampled actions: {action_dict}")

    # B) Data: only refer to train+val
    train_data, val_data, test_data = load_dataset(dataset_name)
    for dset in [train_data, val_data]:
        for data in dset:
            apply_feature_augment(data, action_dict["augment"])

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=0)

    in_channels = train_data.num_node_features
    out_channels = train_data.num_classes

    # C) Build model + EarlyStopping + Vis
    composed_model = build_model(action_dict, in_channels, out_channels)
    module = Evaluator(composed_model, strategy_name="RL-chosen", lr=action_dict["lr"])

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=3,
        verbose=True
    )
    plot_callback = PlotLossAccCallback(episode_idx)

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=5,
        callbacks=[early_stop_callback, plot_callback]
    )
    trainer.fit(module, train_loader, val_loader)

    val_acc = trainer.callback_metrics.get("val_acc", torch.tensor(0.0)).item()

    # D) RL reward = val_acc
    reward = compute_reward(accuracy=val_acc, model_complexity=None, alpha=0.0)
    print(f"[RL] Episode {episode_idx} => val_acc={val_acc:.4f}, reward={reward:.4f}")
    return log_prob, reward, action_dict, val_acc, module

###################################
# 5) Main Function
###################################

def train_rl_controller():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize RL controller (Supports structure embedding + Gumbel‑Softmax, and implements phased search internally)
    controller = MultiHeadRLController(
        encoder_opts, pooling_opts, readout_opts, augment_opts,
        hidden_dim_opts, dropout_opts, lr_opts, bn_opts, temp_opts
    ).to(device)

    optimizer = torch.optim.Adam(controller.parameters(), lr=1e-3)

    episodes = 40
    dataset_name = "ENZYMES"  # "PROTEINS" or "ENZYMES"
    rl_patience = 10
    no_improve_count = 0

    best_val_acc = -1.0
    best_actions = None

    all_episode_train_loss = []
    all_episode_val_loss = []
    all_episode_val_acc = []

    for epi in range(episodes):
        print(f"========== RL Episode {epi} ==========")
        log_prob, reward, action_dict, val_acc, module = run_episode(dataset_name, controller, epi, device=device)

        final_train_loss = module.trainer.callback_metrics.get("train_loss", torch.tensor(999.9)).item()
        final_val_loss = module.trainer.callback_metrics.get("val_loss", torch.tensor(999.9)).item()

        all_episode_train_loss.append(final_train_loss)
        all_episode_val_loss.append(final_val_loss)
        all_episode_val_acc.append(val_acc)

        loss = controller.reinforce_loss(log_prob, reward)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_actions = action_dict
            no_improve_count = 0
            print(f"!!! Found new best val_acc = {best_val_acc:.4f} at episode {epi}")
        else:
            no_improve_count += 1
            if no_improve_count >= rl_patience:
                print(f"[RL] Early stopping after {epi} episodes because no val_acc improvement in last {rl_patience} episodes.")
                break

    print("[RL] Training finished.")
    print(f"Best val_acc = {best_val_acc:.4f}, with actions = {best_actions}")

    best_record = {
        "dataset_name": dataset_name,
        "best_val_acc": best_val_acc,
        "best_actions": best_actions
    }
    with open("best_strategy.json", "w") as f:
        json.dump(best_record, f, indent=2)
    print("==> best_strategy.json has been saved. You can now run deploy_controller.py to do final test.")

    episodes_done = len(all_episode_val_acc)
    xs = range(episodes_done)

    plt.figure()
    plt.title("RL Search Overall: Train/Val Loss & Val Acc")
    plt.plot(xs, all_episode_train_loss, label="Final Train Loss", color="red")
    plt.plot(xs, all_episode_val_loss, label="Final Val Loss", color="orange")
    plt.ylabel("Loss")
    plt.xlabel("Episode")
    plt.legend(loc="upper left")

    ax2 = plt.gca().twinx()
    ax2.plot(xs, all_episode_val_acc, label="Val Acc", color="blue")
    ax2.set_ylabel("Val Acc")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("vis/rl_search_overall.png")
    plt.close()


if __name__ == "__main__":
    train_rl_controller()
