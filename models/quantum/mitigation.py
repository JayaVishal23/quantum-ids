"""
Readout error mitigation (REM) for the quantum detector.

Of the three error sources in models/quantum/noise.py, readout error is the
only one that can be undone after the fact, and that is what makes REM worth
doing: it needs no extra circuit runs on the data, only a one-off calibration.

The idea is simple enough to check by hand. Our noise model flips the reported
bit of each qubit with some probability, so what the device reports is a
shrunk, shifted version of the true expectation value:

    reported = gain * true + offset

Two states we already know the answer for are enough to find the two unknowns:

    all qubits in |0>   true <Z> = +1   device reports  reported_zero
    all qubits in |1>   true <Z> = -1   device reports  reported_one

    gain   = (reported_zero - reported_one) / 2
    offset = (reported_zero + reported_one) / 2

and then every later measurement is corrected by rearranging the same line:

    true = (reported - offset) / gain

This is the diagonal case of the confusion-matrix inversion that hardware
providers call measurement calibration. It is diagonal because our noise model
flips each qubit independently; a real device also has correlated readout, for
which the same method is used with a full 2^n x 2^n matrix.

Two honest caveats, both worth stating in the report:

  * The calibration circuits have to put the qubits into |1> with an X gate,
    and that gate is itself noisy, so a little gate error leaks into what we
    attribute to readout. Real calibration has exactly the same problem.
  * REM only removes readout error. Gate error and decoherence happened before
    the measurement and no amount of post-processing brings that information
    back.

    from models.quantum.mitigation import calibration_matrix, with_readout_correction
    gain, offset = calibration_matrix(8, scale=1.0)
    model = with_readout_correction(noisy_model, gain, offset)
"""

import numpy as np
import pennylane as qml
import torch
from torch import nn

from models.quantum.noise import ibm_noise_model


def calibration_matrix(n_qubits, scale=1.0, shots=None):
    """Run the two calibration circuits and return (gain, offset) per qubit.

    This is the only part of REM that costs circuit runs, and it is paid once
    per device rather than once per prediction.
    """
    device = qml.device("default.mixed", wires=n_qubits, shots=shots)

    @qml.qnode(device)
    def calibration(flip):
        if flip:
            for wire in range(n_qubits):
                qml.PauliX(wires=wire)
        return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

    noisy_calibration = calibration
    if scale > 0:
        noisy_calibration = qml.add_noise(calibration, ibm_noise_model(scale))

    reported_zero = np.asarray(noisy_calibration(False), dtype=float)
    reported_one = np.asarray(noisy_calibration(True), dtype=float)

    gain = (reported_zero - reported_one) / 2.0
    offset = (reported_zero + reported_one) / 2.0

    if np.any(np.abs(gain) < 1e-6):
        raise ValueError("Calibration found a qubit whose readout carries no "
                         "signal at all; it cannot be corrected.")
    return gain, offset


def correct(reported, gain, offset):
    """Undo the readout error in measured expectation values."""
    return (np.asarray(reported) - offset) / gain


class ReadoutCorrection(nn.Module):
    """The same correction as a layer, so it can sit inside the model."""

    def __init__(self, gain, offset):
        super().__init__()
        self.register_buffer("gain", torch.tensor(np.asarray(gain), dtype=torch.float32))
        self.register_buffer("offset", torch.tensor(np.asarray(offset), dtype=torch.float32))

    def forward(self, x):
        return (x - self.offset) / self.gain

    def extra_repr(self):
        return f"qubits={len(self.gain)}, mean gain={self.gain.mean():.4f}"


def quantum_layer_position(model):
    """Where the circuit sits in the Sequential, so we can get in behind it."""
    for position, layer in enumerate(model):
        if isinstance(layer, qml.qnn.TorchLayer):
            return position
    raise ValueError("This model has no quantum layer.")


def with_readout_correction(model, gain, offset):
    """A copy of the model with the correction inserted after the circuit.

    The existing submodules are reused rather than rebuilt, so the trained
    weights come along and no state_dict keys have to be renamed.
    """
    layers = list(model)
    layers.insert(quantum_layer_position(model) + 1, ReadoutCorrection(gain, offset))
    return nn.Sequential(*layers)


@torch.no_grad()
def quantum_features(model, X, batch_size=100):
    """The <Z> values the circuit reports, before the classical head.

    Pulling these out once means the expensive part (simulating the circuit)
    is paid once and the mitigation, which is only arithmetic, can be applied
    to the saved numbers.
    """
    front = nn.Sequential(*list(model)[:quantum_layer_position(model) + 1])
    front.eval()
    batches = []
    for start in range(0, len(X), batch_size):
        batch = torch.tensor(X[start:start + batch_size], dtype=torch.float32)
        batches.append(front(batch))
    return torch.cat(batches).numpy()


@torch.no_grad()
def head_probability(model, features):
    """Attack probability from already-measured <Z> values."""
    head = nn.Sequential(*list(model)[quantum_layer_position(model) + 1:])
    head.eval()
    scores = head(torch.tensor(np.asarray(features), dtype=torch.float32))
    return torch.sigmoid(scores).squeeze(1).numpy()
