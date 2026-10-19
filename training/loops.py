"""
Small training helpers shared by centralised and federated training.

Keeping one copy means a client in federated learning trains in exactly the
same way as the centralised reference model.
"""

import torch


def train_one_epoch(model, loader, optimizer, loss_fn):
    """One pass over the data. Returns the average training loss."""
    model.train()
    total_loss = 0.0
    for features, labels in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(features), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_probability(model, X, batch_size=500):
    """Attack probability for every row of X (a numpy array)."""
    model.eval()
    outputs = []
    for start in range(0, len(X), batch_size):
        batch = torch.tensor(X[start:start + batch_size], dtype=torch.float32)
        outputs.append(torch.sigmoid(model(batch)).squeeze(1))
    return torch.cat(outputs).numpy()


@torch.no_grad()
def average_loss(model, arrays, loss_fn, batch_size=500):
    """Average loss on a whole dataset, without training."""
    model.eval()
    total = 0.0
    X, y = arrays["X"], arrays["y"]
    for start in range(0, len(X), batch_size):
        features = torch.tensor(X[start:start + batch_size], dtype=torch.float32)
        labels = torch.tensor(y[start:start + batch_size], dtype=torch.float32).unsqueeze(1)
        total += loss_fn(model(features), labels).item() * len(labels)
    return total / len(X)


def copy_weights(model):
    """A detached copy of all weights, safe to keep while the model keeps training."""
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def weights_size_in_bytes(weights):
    """How many bytes these weights take when sent over the network (float32)."""
    return sum(value.numel() * value.element_size() for value in weights.values())
