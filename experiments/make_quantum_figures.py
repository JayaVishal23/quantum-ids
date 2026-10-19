"""
Figures that show the quantum side of the project:

    results/diagrams/4_quantum_architecture.png   where each quantum technique is used
    results/diagrams/5_circuit_full.png           the implemented circuit, gate by gate
    results/diagrams/6_circuit_one_layer.png      one layer, large enough to read

Run from the project root:
    python -m experiments.make_quantum_figures
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pennylane as qml
import torch
from matplotlib.patches import FancyBboxPatch

from common import config
from models.quantum.qnn import N_LAYERS, N_QUBITS, make_circuit

OUTPUT_DIR = config.RESULTS_DIR / "diagrams"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
BLUE, BLUE_FILL = "#2a78d6", "#e6effc"
ORANGE, ORANGE_FILL = "#eb6834", "#fdeae2"
GREEN, GREEN_FILL = "#1baf7a", "#e3f6ef"
VIOLET, VIOLET_FILL = "#4a3aa7", "#ebe9f7"
GREY_FILL = "#f0efec"


def save(figure, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    figure.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    print(f"saved {path}")


# ------------------------------------------------------------------
# 4. Where each quantum technique is used
# ------------------------------------------------------------------

def quantum_architecture():
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    figure, ax = plt.subplots(figsize=(12.5, 7.6), facecolor=SURFACE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(1, 96, "Quantum logic of the detector", fontsize=14, fontweight="bold", color=INK)
    ax.text(1, 91.5, "Each stage names the quantum property or technique it uses.",
            fontsize=9.5, color=MUTED)

    stages = [
        (1, "8 classical features\nscaled to [0, pi]", "input", BLUE, BLUE_FILL,
         "one connection record\nafter local preprocessing"),
        (21, "Angle encoding\nRY(x) on each qubit", "technique", VIOLET, VIOLET_FILL,
         "8 features -> 8 qubits\nclassical data becomes\na quantum state"),
        (41, "Superposition\n8-qubit state", "property", GREEN, GREEN_FILL,
         "the state holds\n2^8 = 256 amplitudes\nat the same time"),
        (61, "Variational layers\nRot + CNOT, x3", "technique", ORANGE, ORANGE_FILL,
         "trainable angles (72)\nCNOT ring creates\nentanglement"),
        (81, "Measurement\n<Z> on each qubit", "property", BLUE, BLUE_FILL,
         "the state collapses;\n8 numbers in [-1, 1]\ncome back to the computer"),
    ]

    for x, title, kind, edge, fill, detail in stages:
        ax.add_patch(FancyBboxPatch((x, 58), 17, 22, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    linewidth=1.5, edgecolor=edge, facecolor=fill))
        ax.text(x + 8.5, 74.5, title, ha="center", va="center", fontsize=9.5,
                fontweight="bold", color=INK)
        ax.text(x + 8.5, 65, detail, ha="center", va="center", fontsize=8, color=MUTED)
        ax.text(x + 8.5, 81.5, kind, ha="center", va="bottom", fontsize=8, color=edge,
                style="italic")
        if x < 81:
            ax.annotate("", xy=(x + 19.6, 69), xytext=(x + 17.4, 69),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=1.6))

    # classical tail
    ax.add_patch(FancyBboxPatch((33, 40), 34, 11, boxstyle="round,pad=0.4,rounding_size=1.2",
                                linewidth=1.5, edgecolor=BLUE, facecolor=BLUE_FILL))
    ax.text(50, 45.5, "Classical layer (9 numbers)  ->  normal or attack",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.annotate("", xy=(55, 51.5), xytext=(89, 58),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=1.6))

    ax.add_patch(FancyBboxPatch((1, 24), 97, 12, boxstyle="round,pad=0.4,rounding_size=1.2",
                                linewidth=1.3, edgecolor=MUTED, facecolor=GREY_FILL, linestyle="--"))
    ax.text(50, 32.5, "Training the quantum model", ha="center", fontsize=9.5,
            fontweight="bold", color=INK)
    ax.text(50, 27.5,
            "The 72 gate angles are trained exactly like neural-network weights. On the simulator the gradients come from\n"
            "backpropagation through the simulated state; on real hardware the parameter-shift rule would be used instead.",
            ha="center", va="center", fontsize=8.5, color=MUTED)

    ax.add_patch(FancyBboxPatch((1, 4), 97, 15, boxstyle="round,pad=0.4,rounding_size=1.2",
                                linewidth=1.3, edgecolor=ORANGE, facecolor=ORANGE_FILL))
    ax.text(50, 15.5, "Where this sits in the federated system", ha="center", fontsize=9.5,
            fontweight="bold", color=INK)
    ax.text(50, 9,
            "Each of the 3 clients runs this same circuit on its own traffic. Only the 81 trained numbers (324 bytes) leave a client,\n"
            "encrypted with ML-KEM-768 + AES-256-GCM. The server averages the three sets of numbers and sends the result back.",
            ha="center", va="center", fontsize=8.5, color=MUTED)

    save(figure, "4_quantum_architecture.png")


# ------------------------------------------------------------------
# 5 and 6. The circuit actually implemented
# ------------------------------------------------------------------

def circuit_drawings():
    inputs = torch.tensor([0.9, 2.1, 0.4, 1.7, 2.8, 1.1, 0.6, 2.4])

    full_weights = torch.zeros(N_LAYERS, N_QUBITS, 3)
    circuit = make_circuit()
    figure, _ = qml.draw_mpl(circuit, level="device", style="pennylane",
                             decimals=None)(inputs, full_weights)
    figure.set_size_inches(17, 6)
    save(figure, "5_circuit_full.png")

    # One layer only, so the gates can be read.
    device = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(device)
    def one_layer(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        return [qml.expval(qml.PauliZ(wire)) for wire in range(N_QUBITS)]

    figure, _ = qml.draw_mpl(one_layer, level="device", style="pennylane",
                             decimals=None)(inputs, torch.zeros(1, N_QUBITS, 3))
    figure.set_size_inches(11, 6)
    save(figure, "6_circuit_one_layer.png")


def main():
    quantum_architecture()
    circuit_drawings()

    # Print the specification, which the report quotes.
    specs = qml.specs(make_circuit(), level="device")(
        torch.zeros(N_QUBITS), torch.zeros(N_LAYERS, N_QUBITS, 3))
    print("\nCircuit specification (as implemented):")
    print(specs["resources"].to_pretty_str())


if __name__ == "__main__":
    main()
