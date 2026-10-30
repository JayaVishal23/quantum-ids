"""
One place to build any model by name, so training scripts don't care whether
the model is classical or quantum.

    build_model("mlp") / build_model("tiny_mlp") / build_model("qnn")
    build_model("qnn_amplitude")  needs the 39-feature "wide" arrays
"""

MODEL_NAMES = ["mlp", "tiny_mlp", "qnn", "qnn_amplitude"]

# Amplitude encoding is not limited to one feature per qubit, so it reads the
# all-39-feature arrays instead of the 8 the other models use.
WIDE_FEATURE_MODELS = {"qnn_amplitude"}


def uses_wide_features(name):
    return name in WIDE_FEATURE_MODELS


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

    if name == "qnn_amplitude":
        from models.quantum.qnn_amplitude import build_amplitude_qnn
        return build_amplitude_qnn(qnn_device, qnn_diff_method)

    raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}.")
