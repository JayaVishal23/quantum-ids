"""
Classical models used as baselines.

  MLP       8 -> 16 -> 8 -> 1   (289 trainable parameters)
  Tiny MLP  8 -> 8 -> 1         (81 trainable parameters)

The tiny MLP exists for one reason: the quantum model also has 81
parameters. Comparing two models of the same size is fairer than comparing
the QNN with a big classical network.

Both output a raw score (a "logit"). The loss function turns it into a
probability with a sigmoid, which is more numerically stable than putting the
sigmoid inside the model.
"""

from torch import nn

from common import config

INPUTS = config.NUM_SELECTED_FEATURES


def build_mlp():
    return nn.Sequential(
        nn.Linear(INPUTS, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )


def build_tiny_mlp():
    return nn.Sequential(
        nn.Linear(INPUTS, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )
