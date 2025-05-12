import torch
from torch_geometric.datasets import TUDataset

def get_proteins_datasets():
    dataset = TUDataset(root='data/PROTEINS', name='PROTEINS')
    dataset = dataset.shuffle()
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len

    train_dataset = dataset[:train_len]
    val_dataset = dataset[train_len:train_len + val_len]
    test_dataset = dataset[train_len + val_len:]

    return train_dataset, val_dataset, test_dataset
