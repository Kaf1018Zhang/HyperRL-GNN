import torch
import torch.nn as nn
import torch.nn.functional as F

class ModuleExecutor(nn.Module):
    """
    Abstraction: GNN Block + pooling + Classifer
    """
    def __init__(self, gnn_block, pooling_fn, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.encoder = gnn_block(in_channels, hidden_channels)
        self.pooling_fn = pooling_fn
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = self.encoder(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.pooling_fn(x, batch)
        logits = self.classifier(x)
        return logits
