"""
Loading the prepared data.

Every training script gets its data through these few functions, so all
experiments are guaranteed to read the same files.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from common import config


def _load_npz(path):
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def load_client(client_id, part="train", small=True):
    """One client's own data. part is "train" or "val"."""
    suffix = "_small" if small else ""
    return _load_npz(config.CLIENTS_DIR / f"client_{client_id}" / f"{part}{suffix}.npz")


def load_test(small=True):
    suffix = "_small" if small else ""
    return _load_npz(config.TEST_DIR / f"test{suffix}.npz")


def load_pooled(part="train", small=True):
    """All clients' data stacked together.

    Only used for the centralised reference models, which show what we could
    reach if privacy didn't matter. The federated code never calls this.
    """
    pieces = [load_client(client_id, part, small) for client_id in config.CLIENT_ATTACKS]
    return {name: np.concatenate([piece[name] for piece in pieces]) for name in pieces[0]}


def make_loader(arrays, batch_size, shuffle, seed=0):
    features = torch.tensor(arrays["X"], dtype=torch.float32)
    labels = torch.tensor(arrays["y"], dtype=torch.float32).unsqueeze(1)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(features, labels), batch_size=batch_size,
                      shuffle=shuffle, generator=generator)
