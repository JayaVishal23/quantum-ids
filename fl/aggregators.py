"""
How the server combines the clients' weights into one global model.

    fedavg   the normal rule: a weighted average
    median   a defence: the middle value, which one bad client cannot control
"""

import torch


def fedavg(client_weights, client_rows):
    """Federated averaging: a weighted average of every weight.

    A client with more training rows gets proportionally more say:
        global = sum over clients of (rows_k / total_rows) * weights_k

    This is the standard rule, but it has no protection: an average can be
    dragged anywhere by one extreme value.
    """
    total_rows = sum(client_rows)
    return {
        name: sum(weights[name] * (rows / total_rows) for weights, rows in zip(client_weights, client_rows))
        for name in client_weights[0]
    }


def coordinate_wise_median(client_weights, client_rows):
    """For every single weight, take the middle value of the clients' values.

    With 3 clients, one malicious client can never decide the result: however
    extreme its number is, the middle value still lies between the two honest
    ones. It tolerates one malicious client out of three, not two.

    `client_rows` is deliberately ignored. Weighting by dataset size would let a
    malicious client claim to be large and regain influence.
    """
    return {
        name: torch.median(torch.stack([weights[name] for weights in client_weights]), dim=0).values
        for name in client_weights[0]
    }


AGGREGATORS = {
    "fedavg": fedavg,
    "median": coordinate_wise_median,
}
