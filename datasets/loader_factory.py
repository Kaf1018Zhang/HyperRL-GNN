from datasets.proteins_loader import get_proteins_datasets
from datasets.enzymes_loader import get_enzymes_datasets

def load_dataset(name):
    name = name.lower()
    if name == 'proteins':
        return get_proteins_datasets()
    elif name == 'enzymes':
        return get_enzymes_datasets()
    else:
        raise ValueError(f"Unknown dataset: {name}")
