"""
Checks that every library the project needs is installed and actually works.

Run it from the project root:
    python -m common.check_env

It doesn't only import things. It runs three tiny real tests:
  1. a 2-qubit quantum circuit trained as a PyTorch layer,
  2. an ML-KEM-768 key exchange followed by AES-256-GCM encryption,
  3. ML-DSA-65 and Ed25519 signatures.
If all three pass, the machine is ready for Review 1.
"""

import sys
import os


def show_versions():
    import numpy
    import pandas
    import sklearn
    import torch
    import pennylane
    import cryptography
    from cryptography.hazmat.backends.openssl.backend import backend

    print("Python      ", sys.version.split()[0])
    print("numpy       ", numpy.__version__)
    print("pandas      ", pandas.__version__)
    print("scikit-learn", sklearn.__version__)
    print("torch       ", torch.__version__)
    print("pennylane   ", pennylane.__version__)
    print("cryptography", cryptography.__version__, f"({backend.openssl_version_text()})")


def check_quantum_layer():
    """Train a 2-qubit circuit for a few steps and make sure the loss goes down."""
    import torch
    import pennylane as qml

    torch.manual_seed(0)
    device = qml.device("default.qubit", wires=2)

    @qml.qnode(device, interface="torch")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=[0, 1], rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=[0, 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(2)]

    model = torch.nn.Sequential(
        qml.qnn.TorchLayer(circuit, {"weights": (1, 2, 3)}),
        torch.nn.Linear(2, 1),
    )

    # Ten random rows with a simple rule, so there is something to learn.
    x = torch.rand(10, 2) * torch.pi
    y = (x[:, 0] > x[:, 1]).float().unsqueeze(1)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    first_loss = loss_fn(model(x), y).item()
    for _ in range(15):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

    last_loss = loss.item()
    print(f"loss {first_loss:.3f} -> {last_loss:.3f}")

    if not last_loss < first_loss:
        raise RuntimeError("Quantum layer did not learn (loss did not go down).")


def check_mlkem_and_aes():
    """Do a real ML-KEM-768 exchange, then encrypt and decrypt with the shared key."""
    from cryptography.hazmat.primitives.asymmetric import mlkem
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    server_key = mlkem.MLKEM768PrivateKey.generate()
    public_key = server_key.public_key()

    client_secret, ciphertext = public_key.encapsulate()
    server_secret = server_key.decapsulate(ciphertext)

    if client_secret != server_secret:
        raise RuntimeError("ML-KEM secrets don't match.")

    print(f"public key {len(public_key.public_bytes_raw())} B, "
          f"ciphertext {len(ciphertext)} B, secret {len(client_secret)} B")

    message = b"model weights go here"
    nonce = os.urandom(12)
    sealed = AESGCM(client_secret).encrypt(nonce, message, b"header")
    opened = AESGCM(server_secret).decrypt(nonce, sealed, b"header")

    if opened != message:
        raise RuntimeError("AES-GCM round trip failed.")


def check_signatures():
    from cryptography.hazmat.primitives.asymmetric import ed25519, mldsa

    for name, key in [
        ("ML-DSA-65", mldsa.MLDSA65PrivateKey.generate()),
        ("Ed25519", ed25519.Ed25519PrivateKey.generate()),
    ]:
        signature = key.sign(b"handshake")
        key.public_key().verify(signature, b"handshake")  # raises if wrong
        print(f"{name}: signature {len(signature)} B, verified")


def main():
    checks = [
        ("Library versions", show_versions),
        ("Quantum layer (PennyLane + PyTorch)", check_quantum_layer),
        ("ML-KEM-768 + AES-256-GCM", check_mlkem_and_aes),
        ("Signatures", check_signatures),
    ]

    failed = 0
    for title, check in checks:
        print(f"\n--- {title}")
        try:
            check()
            print("OK")
        except Exception as error:
            failed += 1
            print(f"FAILED: {type(error).__name__}: {error}")

    print()
    if failed:
        print(f"{failed} check(s) failed. Fix these before continuing.")
        sys.exit(1)
    print("All checks passed. Environment is ready.")


if __name__ == "__main__":
    main()
