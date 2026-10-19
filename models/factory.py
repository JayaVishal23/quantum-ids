"""
One place to build any model by name, so training scripts don't care whether
the model is classical or quantum.

    build_model("mlp") / build_model("tiny_mlp") / build_model("qnn")
"""

MODEL_NAMES = ["mlp", "tiny_mlp", "qnn"]


def build_model(name, qnn_device="default.qubit", qnn_diff_method="backprop"):
    if name == "mlp":
        from models.classical.mlp import build_mlp
        return build_mlp()

    if name == "tiny_mlp":
        from models.classical.mlp import build_tiny_mlp
        return build_tiny_mlp()

    if name == "qnn":
        # Imported only when needed, because PennyLane is slow to import.
        from models.quantum.qnn import build_qnn
        return build_qnn(qnn_device, qnn_diff_method)

    raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}.")
