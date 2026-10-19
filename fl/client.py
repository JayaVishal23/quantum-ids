"""
A federated client.

Each client loads ONLY its own folder (data/clients/client_k). It never sees
the other clients' data. It does two jobs:

  local_train      start from the global model, train on own data, return new weights
  evaluate_global  report how well the global model does on its own validation data
"""

import time

import torch
from torch import nn

from attacks.poisoning import LABEL_FLIP, MODEL_POISON, NO_ATTACK, flip_labels, poison_update
from common import data
from models.factory import build_model
from training.loops import average_loss, copy_weights, train_one_epoch


class Client:
    def __init__(self, client_id, model_name, qnn_device="default.qubit", qnn_diff_method="backprop",
                 attack=NO_ATTACK):
        self.client_id = client_id
        self.train_data = data.load_client(client_id, "train")
        self.val_data = data.load_client(client_id, "val")
        self.model = build_model(model_name, qnn_device, qnn_diff_method)
        self.loss_fn = nn.BCEWithLogitsLoss()

        # A malicious client. Everything else about it stays normal: it holds
        # valid keys and its messages are correctly formed and encrypted.
        self.attack = attack
        if attack == LABEL_FLIP:
            self.train_data = flip_labels(self.train_data)

    @property
    def num_train_rows(self):
        return len(self.train_data["y"])

    def local_train(self, global_weights, epochs, lr, batch_size, seed):
        self.model.load_state_dict(global_weights)

        # A fresh optimiser each round: the client only keeps the weights it was sent.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loader = data.make_loader(self.train_data, batch_size, shuffle=True, seed=seed)

        start = time.perf_counter()
        for _ in range(epochs):
            train_loss = train_one_epoch(self.model, loader, optimizer, self.loss_fn)
        seconds = time.perf_counter() - start

        weights = copy_weights(self.model)
        if self.attack == MODEL_POISON:
            weights = poison_update(global_weights, weights)

        # This dictionary is the update sent to the server.
        return {
            "client_id": self.client_id,
            "num_rows": self.num_train_rows,
            "weights": weights,
            "train_loss": train_loss,
            "seconds": seconds,
        }

    def evaluate_global(self, global_weights):
        """Only a loss number and a row count leave the client, never the data."""
        self.model.load_state_dict(global_weights)
        return {
            "client_id": self.client_id,
            "num_rows": len(self.val_data["y"]),
            "val_loss": average_loss(self.model, self.val_data, self.loss_fn),
        }
