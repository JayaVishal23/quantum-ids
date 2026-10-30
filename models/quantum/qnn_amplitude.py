"""
The amplitude-encoded QNN, with a trainable projection in front of it.

    39 scaled features
      -> Linear(39 -> 16): a trainable projection, learned WITH the circuit
      -> AmplitudeEmbedding on 8 qubits: the vector becomes the amplitudes of
         the quantum state, padded out to 2^8 = 256 and normalised
      -> StronglyEntanglingLayers x 3: trainable rotations + CNOTs
      -> measure <Z> on each qubit: 8 numbers between -1 and 1
      -> Linear(8 -> 1): one attack score

WHY THIS EXISTS
Angle encoding spends one qubit per feature, so 8 qubits meant 8 features out
of 39 and a mutual-information filter had to pick them in advance. Amplitude
encoding stores a whole vector in the amplitudes of the same 8 qubits, so the
feature budget stops being tied to the qubit count, and the projection that
compresses 39 features down to 16 is trained by gradient descent along with
everything else instead of being fixed before training starts.

THE NORM
Amplitude encoding needs a unit vector, so normalising throws away how large
the vector was. That matters here: sbytes and rate are features where the
magnitude IS the signal. So the norm is computed first and appended as one
extra component, and the normalisation that follows cannot destroy it.

WHAT IT COSTS
The projection is 39 x 16 + 16 = 640 parameters, and those parameters are part
of the model, so they travel in every federated round like any other weight.
The whole model is 721 parameters against the angle-encoded model's 81. That
is a real cost in communication and it is reported, not hidden.
"""

import pennylane as qml
import torch
from torch import nn

from common import config

N_QUBITS = 8
N_LAYERS = 3
N_PROJECTED = 16          # features handed to the circuit, before the norm column
N_AMPLITUDES = 2 ** N_QUBITS

DEFAULT_DEVICE = "default.qubit"
DEFAULT_DIFF_METHOD = "backprop"


class NormAwareProjection(nn.Module):
    """39 features -> N_PROJECTED trainable components, plus the original norm.

    The norm goes in as its own component so that the unit-length requirement
    of amplitude encoding cannot discard the overall scale of the record.
    """

    def __init__(self, in_features=len(config.NUMERIC_FEATURES), out_features=N_PROJECTED):
        super().__init__()
        self.project = nn.Linear(in_features, out_features)

    def forward(self, x):
        norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        projected = torch.tanh(self.project(x))     # keep the components bounded
        return torch.cat([projected, norm], dim=-1)


def make_circuit(device_name=DEFAULT_DEVICE, diff_method=DEFAULT_DIFF_METHOD):
    """`inputs` is the projected vector, `weights` the trainable gate angles."""
    device = qml.device(device_name, wires=N_QUBITS)

    @qml.qnode(device, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        # pad_with fills the unused amplitudes; normalize makes it a legal state.
        qml.AmplitudeEmbedding(inputs, wires=range(N_QUBITS), pad_with=0.0, normalize=True)
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        return [qml.expval(qml.PauliZ(wire)) for wire in range(N_QUBITS)]

    return circuit


def build_amplitude_qnn(device_name=DEFAULT_DEVICE, diff_method=DEFAULT_DIFF_METHOD):
    circuit = make_circuit(device_name, diff_method)
    weight_shapes = {"weights": (N_LAYERS, N_QUBITS, 3)}

    return nn.Sequential(
        NormAwareProjection(),
        qml.qnn.TorchLayer(circuit, weight_shapes),
        nn.Linear(N_QUBITS, 1),
    )
