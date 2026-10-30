"""
A realistic hardware noise model for the quantum detector.

Everything else in this project runs on an ideal simulator, which answers
"does the model work?" but not "would it work on a real machine?". This file
answers the second question by putting the same circuit on a density-matrix
simulator with the three error sources a superconducting device actually has:

  gate errors        every gate is slightly wrong. Two-qubit gates are about
                     twenty times worse than single-qubit gates, which matters
                     here because our circuit is built from a CNOT ring.
  decoherence        a qubit forgets. T1 is how long it keeps its energy, T2
                     how long it keeps its phase, and both are compared against
                     how long the gate takes.
  readout error      the final measurement reports the wrong bit.

The numbers below are representative of an IBM Eagle-class processor. They are
device parameters, not fitted to our data, and they are read from one place so
the report can quote them and a reviewer can change them.

Noise is applied for EVALUATION only. Training stays on the ideal simulator,
which is standard practice: a model is trained in simulation and then tested
for how much accuracy survives on hardware.

    from models.quantum.noise import ibm_noise_model, noisy_model
    model = noisy_model("qnn", scale=1.0)
"""

import pennylane as qml

# ------------------------------------------------------------------
# Device parameters, IBM Eagle-class superconducting processor
# ------------------------------------------------------------------
# Times are in seconds. Error rates are probabilities per gate.

T1_SECONDS = 180e-6          # energy relaxation time
T2_SECONDS = 120e-6          # dephasing time (always <= 2 * T1)

ONE_QUBIT_GATE_ERROR = 3.0e-4
ONE_QUBIT_GATE_TIME = 35e-9

TWO_QUBIT_GATE_ERROR = 8.0e-3     # ~27x worse than a single-qubit gate
TWO_QUBIT_GATE_TIME = 450e-9

READOUT_ERROR = 1.3e-2

# Probability the qubit relaxes to |1> rather than |0>; on a cold device the
# qubit almost always settles into the ground state.
EXCITED_STATE_POPULATION = 0.01

TWO_QUBIT_GATES = ["CNOT", "CZ", "ECR", "CY", "CRX", "CRY", "CRZ"]


def ibm_noise_model(scale=1.0):
    """The noise model, with every rate multiplied by `scale`.

    scale=0 is a noiseless device, scale=1 is the device described above, and
    larger values show how the circuit behaves on worse hardware. Sweeping it
    is more informative than a single number, because it shows whether accuracy
    degrades gently or falls off a cliff.
    """
    if scale < 0:
        raise ValueError("scale must be zero or positive.")

    one_qubit_error = min(ONE_QUBIT_GATE_ERROR * scale, 1.0)
    two_qubit_error = min(TWO_QUBIT_GATE_ERROR * scale, 1.0)
    readout_error = min(READOUT_ERROR * scale, 0.5)

    def one_qubit_noise(op, **_):
        qml.DepolarizingChannel(one_qubit_error, wires=op.wires)
        qml.ThermalRelaxationError(EXCITED_STATE_POPULATION, T1_SECONDS, T2_SECONDS,
                                   ONE_QUBIT_GATE_TIME, wires=op.wires)

    def two_qubit_noise(op, **_):
        # Depolarise each wire the gate touches, then let both decohere for as
        # long as the (much slower) two-qubit gate takes.
        for wire in op.wires:
            qml.DepolarizingChannel(two_qubit_error, wires=wire)
            qml.ThermalRelaxationError(EXCITED_STATE_POPULATION, T1_SECONDS, T2_SECONDS,
                                       TWO_QUBIT_GATE_TIME, wires=wire)

    def readout_noise(op, **_):
        # A measurement that reports the wrong bit is modelled as a bit flip
        # immediately before measuring.
        qml.BitFlip(readout_error, wires=op.wires)

    two_qubit = qml.noise.op_in(TWO_QUBIT_GATES)
    model_map = {
        two_qubit: two_qubit_noise,
        ~two_qubit: one_qubit_noise,
    }
    return qml.NoiseModel(model_map, meas_map={qml.noise.meas_eq(qml.expval): readout_noise})


def build_noisy_model(model_name, scale=1.0, shots=None):
    """The same architecture as the trained model, on a noisy simulator.

    The returned module has exactly the layers the ideal model has, so a
    state_dict saved from training loads straight into it. Only the simulator
    underneath is different: default.mixed tracks a density matrix, which is
    what lets it represent a state damaged by noise.

    shots=None gives exact expectation values from the noisy state; an integer
    samples that many times instead, which adds the statistical error a real
    machine also has.
    """
    from torch import nn

    if model_name == "qnn":
        from models.quantum.qnn import N_LAYERS, N_QUBITS
        embed = lambda inputs: qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
        front = None
    elif model_name == "qnn_amplitude":
        from models.quantum.qnn_amplitude import N_LAYERS, N_QUBITS, NormAwareProjection
        embed = lambda inputs: qml.AmplitudeEmbedding(
            inputs, wires=range(N_QUBITS), pad_with=0.0, normalize=True)
        front = NormAwareProjection()
    else:
        raise ValueError(f"'{model_name}' is not a quantum model.")

    device = qml.device("default.mixed", wires=N_QUBITS, shots=shots)

    @qml.qnode(device, interface="torch")
    def circuit(inputs, weights):
        embed(inputs)
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        return [qml.expval(qml.PauliZ(wire)) for wire in range(N_QUBITS)]

    if scale > 0:
        circuit = qml.add_noise(circuit, ibm_noise_model(scale))

    layer = qml.qnn.TorchLayer(circuit, {"weights": (N_LAYERS, N_QUBITS, 3)})
    layers = [layer, nn.Linear(N_QUBITS, 1)]
    if front is not None:
        layers.insert(0, front)
    return nn.Sequential(*layers)


def describe(scale=1.0):
    """The parameters in force, for the report and for printing in a run."""
    return {
        "scale": scale,
        "one_qubit_gate_error": ONE_QUBIT_GATE_ERROR * scale,
        "two_qubit_gate_error": TWO_QUBIT_GATE_ERROR * scale,
        "readout_error": READOUT_ERROR * scale,
        "T1_microseconds": T1_SECONDS * 1e6,
        "T2_microseconds": T2_SECONDS * 1e6,
        "one_qubit_gate_ns": ONE_QUBIT_GATE_TIME * 1e9,
        "two_qubit_gate_ns": TWO_QUBIT_GATE_TIME * 1e9,
    }
