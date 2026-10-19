"""
The federated server.

It holds the global model and never sees any client data. Each round it:
  1. hands out the current global weights
  2. receives the clients' new weights and combines them (FedAvg for now)
  3. receives each client's validation loss, to decide which round was best
"""

from fl.aggregators import AGGREGATORS
from models.factory import build_model
from training.loops import copy_weights


class Server:
    def __init__(self, model_name, aggregator="fedavg", qnn_device="default.qubit", qnn_diff_method="backprop"):
        self.model = build_model(model_name, qnn_device, qnn_diff_method)
        self.aggregate_fn = AGGREGATORS[aggregator]

        self.best_val_loss = float("inf")
        self.best_round = 0
        self.best_weights = self.global_weights()

    def global_weights(self):
        return copy_weights(self.model)

    def aggregate(self, updates):
        new_weights = self.aggregate_fn(
            [update["weights"] for update in updates],
            [update["num_rows"] for update in updates],
        )
        self.model.load_state_dict(new_weights)

    def record_validation(self, round_number, reports):
        """Combine clients' validation losses (weighted by rows) and remember the best round.

        This is how we choose the final model WITHOUT looking at the test set.
        """
        total_rows = sum(report["num_rows"] for report in reports)
        val_loss = sum(report["val_loss"] * report["num_rows"] for report in reports) / total_rows

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_round = round_number
            self.best_weights = self.global_weights()

        return val_loss
