"""
The hybrid quantum neural network (QNN).

    8 features in [0, pi]
      -> AngleEmbedding: each feature is the angle of an RY gate on its own qubit
      -> StronglyEntanglingLayers x 3: trainable rotations + CNOTs linking the qubits
      -> measure <Z> on each qubit: 8 numbers between -1 and 1
      -> Linear(8 -> 1): one attack score

Trainable parameters:
    quantum part   3 layers x 8 qubits x 3 angles = 72
    classical part 8 weights + 1 bias             =  9
    total                                          = 81   (same as the tiny MLP)

The circuit runs on a simulator. PennyLane's TorchLayer wraps it so PyTorch
can train it like any other layer.

Which simulator? We measured both on this laptop (8 qubits, batch of 32):
    default.qubit   + backprop  ~474 training rows/s   (~19 s per epoch)
    lightning.qubit + adjoint    ~66 training rows/s   (~172 s per epoch)
With only 8 qubits the quantum state is just 2^8 = 256 numbers, so PyTorch
can simulate a whole batch at once and get gradients by ordinary backprop.
lightning.qubit is built for bigger circuits and works one row at a time.
On real quantum hardware you would use the parameter-shift rule instead.
"""

import pennylane as qml
from torch import nn

from common import config

N_QUBITS = config.NUM_SELECTED_FEATURES
N_LAYERS = 3

DEFAULT_DEVICE = "default.qubit"
DEFAULT_DIFF_METHOD = "backprop"


def make_circuit(device_name=DEFAULT_DEVICE, diff_method=DEFAULT_DIFF_METHOD):
    """The quantum circuit. `inputs` are the 8 features, `weights` the trainable angles."""
    device = qml.device(device_name, wires=N_QUBITS)

    @qml.qnode(device, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        return [qml.expval(qml.PauliZ(wire)) for wire in range(N_QUBITS)]

    return circuit


def build_qnn(device_name=DEFAULT_DEVICE, diff_method=DEFAULT_DIFF_METHOD):
    circuit = make_circuit(device_name, diff_method)
    weight_shapes = {"weights": (N_LAYERS, N_QUBITS, 3)}

    return nn.Sequential(
        qml.qnn.TorchLayer(circuit, weight_shapes),
        nn.Linear(N_QUBITS, 1),
    )


def save_circuit_drawing(path):
    """Picture of the circuit for the report and slides."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    circuit = make_circuit()
    example_inputs = torch.zeros(N_QUBITS)
    example_weights = torch.zeros(N_LAYERS, N_QUBITS, 3)

    figure, _ = qml.draw_mpl(circuit, style="pennylane")(example_inputs, example_weights)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    # Text version, also handy for the report.
    return qml.draw(circuit)(example_inputs, example_weights)
