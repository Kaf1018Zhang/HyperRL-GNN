import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    SAGEConv,
    global_mean_pool,
    global_max_pool,
    GlobalAttention
)

######################################
# 1. Encoder (GCN / GAT / GraphSAGE)
######################################
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, with_bn=False):
        super().__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels) if with_bn else None
        self.bn2 = nn.BatchNorm1d(out_channels) if with_bn else None

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        if self.bn1: x = self.bn1(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        if self.bn2: x = self.bn2(x)

        return x


class GATEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, with_bn=False):
        super().__init__()
        # concat=False => 每层输出大小 = out_channels
        self.conv1 = GATConv(in_channels, out_channels, heads=heads, concat=False)
        self.conv2 = GATConv(out_channels, out_channels, heads=1, concat=False)
        self.bn1 = nn.BatchNorm1d(out_channels) if with_bn else None
        self.bn2 = nn.BatchNorm1d(out_channels) if with_bn else None

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)  # => shape [*, out_channels]
        x = F.elu(x)
        if self.bn1: x = self.bn1(x)

        x = self.conv2(x, edge_index)  # => shape [*, out_channels]
        x = F.elu(x)
        if self.bn2: x = self.bn2(x)

        return x


class SAGEEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, with_bn=False):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, out_channels)
        self.conv2 = SAGEConv(out_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels) if with_bn else None
        self.bn2 = nn.BatchNorm1d(out_channels) if with_bn else None

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        if self.bn1: x = self.bn1(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        if self.bn2: x = self.bn2(x)

        return x

######################################
# 2. Pooling (mean / max / attention)
######################################
def pooling_mean(x, batch):
    return global_mean_pool(x, batch)

def pooling_max(x, batch):
    return global_max_pool(x, batch)

class GlobalAttentionPooling(nn.Module):
    def __init__(self, gate_in_dim):
        super().__init__()
        self.gate_nn = nn.Sequential(
            nn.Linear(gate_in_dim, gate_in_dim),
            nn.ReLU(),
            nn.Linear(gate_in_dim, 1)
        )
        self.att_pool = GlobalAttention(self.gate_nn)

    def forward(self, x, batch):
        return self.att_pool(x, batch)

######################################
# 3. Readout: linear / MLP / transformer
######################################
class LinearReadout(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)

class MLPReadout(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class TransformerReadout(nn.Module):
    def __init__(self, d_model, out_dim, nhead=4, num_layers=1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1,1,d_model))
        self.out_fc = nn.Linear(d_model, out_dim)
        self.d_model = d_model

    def forward_one_graph(self, x_sub):
        # 加CLS token
        cls_tokens = self.cls_token  # [1,1,d_model]
        x_sub = x_sub.unsqueeze(0)   # => [1, n_i, d_model]
        x_cat = torch.cat([cls_tokens, x_sub], dim=1) # => [1, n_i+1, d_model]
        out = self.transformer_enc(x_cat)             # => [1, n_i+1, d_model]
        cls_out = out[:,0,:]                          # => [1, d_model]
        logits = self.out_fc(cls_out)                 # => [1, out_dim]
        return logits.squeeze(0)                      # => [out_dim]

    def forward(self, x, batch):
        device = x.device
        n_graphs = batch.max().item() + 1
        outs = []
        for g in range(n_graphs):
            mask = (batch == g)
            x_sub = x[mask]  # shape [n_i, d_model]
            if x_sub.size(0) == 0:
                outs.append(torch.zeros(self.out_fc.out_features, device=device))
                continue
            out_sub = self.forward_one_graph(x_sub)
            outs.append(out_sub)  # => shape [out_dim]

        return torch.stack(outs, dim=0)  # => [batch_size, out_dim]

######################################
# 4. Feature Augment: raw / spectral / virtual
######################################
def raw_features(data):
    return data

def spectral_features(data):
    # NNeed to implement Laplace features, etc. I will think about it in the future.
    return data

def virtual_node_features(data):
    # Need to add virtual nodes, edges, etc. May try if I have time.
    return data
