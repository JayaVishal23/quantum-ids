"""
Draws the diagrams used to explain the system:

    results/diagrams/1_system_architecture.png   the whole system
    results/diagrams/2_pqc_handshake.png         the handshake, step by step
    results/diagrams/3_mlkem_and_aes.png         how ML-KEM and AES-256-GCM work

Run from the project root:
    python -m experiments.make_diagrams
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from common import config

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
BLUE = "#2a78d6"
BLUE_FILL = "#e6effc"
ORANGE = "#eb6834"
ORANGE_FILL = "#fdeae2"
GREEN = "#1baf7a"
GREEN_FILL = "#e3f6ef"
GREY_FILL = "#f0efec"

OUTPUT_DIR = config.RESULTS_DIR / "diagrams"


def new_figure(width, height):
    figure, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_facecolor(SURFACE)
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    return figure, ax


def box(ax, x, y, width, height, lines, edge=BLUE, fill=BLUE_FILL, size=9, bold_first=True):
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                                boxstyle="round,pad=0.4,rounding_size=1.2",
                                linewidth=1.4, edgecolor=edge, facecolor=fill))
    if isinstance(lines, str):
        lines = [lines]
    step = height / (len(lines) + 1)
    for index, text in enumerate(lines):
        ax.text(x + width / 2, y + height - step * (index + 1), text,
                ha="center", va="center", fontsize=size, color=INK,
                fontweight="bold" if (index == 0 and bold_first) else "normal")


def arrow(ax, x1, y1, x2, y2, label=None, colour=BLUE, style="-|>", dashed=False, label_size=8.5, label_dy=1.6):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=colour, linewidth=1.6,
                                linestyle="--" if dashed else "-",
                                shrinkA=0, shrinkB=0))
    if label:
        ax.text((x1 + x2) / 2, max(y1, y2) + label_dy, label, ha="center", va="bottom",
                fontsize=label_size, color=INK)


def note(ax, x, y, width, height, text, colour=MUTED, fill=GREY_FILL, size=8.5):
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                                boxstyle="round,pad=0.3,rounding_size=0.8",
                                linewidth=1, edgecolor=colour, facecolor=fill, linestyle="--"))
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
            fontsize=size, color=MUTED, style="italic")


def save(figure, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    figure.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    print(f"saved {path}")


# ------------------------------------------------------------------
# 1. The whole system
# ------------------------------------------------------------------

def system_architecture():
    figure, ax = new_figure(12, 7.5)
    ax.text(2, 96, "System architecture", fontsize=14, fontweight="bold", color=INK)
    ax.text(2, 91.5, "Three organisations train one intrusion detector together. "
                     "Raw traffic never leaves a client.", fontsize=9.5, color=MUTED)

    clients = [
        ("CLIENT 1", ["Generic, Analysis", ""], 2),
        ("CLIENT 2", ["Exploits, Fuzzers", ""], 35),
        ("CLIENT 3", ["DoS, Recon, Backdoor,", "Shellcode, Worms"], 68),
    ]
    for title, families, x in clients:
        box(ax, x, 56, 30, 26,
            [title, "own network traffic", *families, "8 features -> 8-qubit QNN", "trains locally"],
            edge=BLUE, fill=BLUE_FILL)
        arrow(ax, x + 15, 56, x + 15, 48, colour=BLUE)
        ax.text(x + 16, 51, "model update\n(324 bytes)", fontsize=8, color=INK, ha="left", va="center")

    box(ax, 2, 36, 96, 11,
        ["SECURE CHANNEL   (fresh handshake every round)",
         "ML-KEM-768 key exchange  +  ML-DSA-65 signatures  +  AES-256-GCM encryption"],
        edge=ORANGE, fill=ORANGE_FILL, size=9.5)

    arrow(ax, 50, 36, 50, 26, colour=ORANGE)
    box(ax, 28, 10, 44, 15,
        ["SERVER", "decrypts the three updates",
         "averages them (FedAvg)", "-> new global detector"],
        edge=GREEN, fill=GREEN_FILL)

    # Return path: around the outside, so it never crosses a client box.
    for (x1, y1, x2, y2) in [(72, 17, 99, 17), (99, 17, 99, 88), (99, 88, 17, 88)]:
        arrow(ax, x1, y1, x2, y2, colour=GREEN, dashed=True, style="-")
    arrow(ax, 17, 88, 17, 82, colour=GREEN, dashed=True)
    ax.text(58, 84.2, "new global model sent back to every client (also encrypted)",
            fontsize=8.5, color=MUTED, ha="center", va="bottom")

    note(ax, 2, 1, 96, 6,
         "What leaves a client: model numbers, feature scores, and min/max summaries.    "
         "What never leaves: the network traffic itself.")
    save(figure, "1_system_architecture.png")


# ------------------------------------------------------------------
# 2. The handshake
# ------------------------------------------------------------------

def pqc_handshake():
    figure, ax = new_figure(12, 9)
    ax.text(2, 97, "Secure channel: one handshake, done fresh every round", fontsize=14,
            fontweight="bold", color=INK)
    ax.text(2, 93, "Message sizes are the ones measured in our implementation.", fontsize=9.5, color=MUTED)

    left, right = 16, 84
    box(ax, left - 13, 84, 26, 7, ["CLIENT"], edge=BLUE, fill=BLUE_FILL, size=10)
    box(ax, right - 13, 84, 26, 7, ["SERVER"], edge=GREEN, fill=GREEN_FILL, size=10)
    for x, colour in ((left, BLUE), (right, GREEN)):
        ax.plot([x, x], [6, 84], linestyle="--", color=colour, linewidth=1, alpha=0.6)

    note(ax, 2, 75, 96, 6,
         "Before training: the server holds every client's public key; every client holds the server's public key (pinning).")

    arrow(ax, left, 70, right, 70, "1.  hello:  client id, round number, random nonce   (103 bytes)", colour=BLUE)
    arrow(ax, right, 60, left, 60,
          "2.  ML-KEM-768 public key (1,184 B)  +  server signature (3,309 B)", colour=GREEN)
    note(ax, 58, 63.5, 40, 5, "server creates a NEW key pair for this round", colour=GREEN)

    note(ax, 20, 50, 52, 7,
         "client checks the signature with the pinned key\n"
         "-> a swapped public key is rejected (man-in-the-middle blocked)", colour=BLUE)

    arrow(ax, left, 43, right, 43,
          "3.  ML-KEM ciphertext (1,088 B)  +  client signature (3,309 B)", colour=BLUE)
    note(ax, 50, 35.5, 48, 6,
         "server checks the client signature -> impostors rejected\n"
         "server decapsulates -> both sides now share one secret", colour=GREEN)

    box(ax, 10, 26, 80, 6.5,
        ["HKDF-SHA256(shared secret + whole handshake)  ->  two AES-256 keys, one per direction"],
        edge=ORANGE, fill=ORANGE_FILL, size=9)

    arrow(ax, right, 19, left, 19, "4.  global model, encrypted with AES-256-GCM", colour=GREEN)
    arrow(ax, left, 12, right, 12, "5.  model update, encrypted  (324 B -> 1,393 B message)", colour=BLUE)

    note(ax, 2, 1, 96, 6,
         "Every message carries the round number and a sequence number, and the keys are new each round, "
         "so replayed messages are rejected.", colour=ORANGE, fill=ORANGE_FILL)
    save(figure, "2_pqc_handshake.png")


# ------------------------------------------------------------------
# 3. ML-KEM and AES-GCM
# ------------------------------------------------------------------

def mlkem_and_aes():
    figure, ax = new_figure(12, 8.5)
    ax.text(2, 97, "How the key is agreed (ML-KEM-768) and the update is encrypted (AES-256-GCM)",
            fontsize=13.5, fontweight="bold", color=INK)

    ax.text(2, 90, "Step A: agree on a secret key, without ever sending it", fontsize=10.5,
            fontweight="bold", color=MUTED)

    box(ax, 2, 68, 28, 16, ["SERVER", "key generation", "-> public key (1,184 B)", "-> private key (kept)"],
        edge=GREEN, fill=GREEN_FILL)
    box(ax, 38, 68, 28, 16, ["CLIENT", "encapsulate", "-> shared secret (32 B)", "-> ciphertext (1,088 B)"],
        edge=BLUE, fill=BLUE_FILL)
    box(ax, 72, 68, 26, 16, ["SERVER", "decapsulate", "with its private key", "-> same 32-byte secret"],
        edge=GREEN, fill=GREEN_FILL)

    arrow(ax, 30, 76, 38, 76, "public key", colour=GREEN, label_dy=4, label_size=8)
    arrow(ax, 66, 76, 72, 76, "ciphertext", colour=BLUE, label_dy=4, label_size=8)

    note(ax, 2, 59, 96, 6.5,
         "The secret is never sent. Only the public key and the ciphertext travel, and recovering\n"
         "the secret from them is the lattice problem no known quantum algorithm solves.")

    ax.text(2, 52, "Step B: turn the secret into an encryption key", fontsize=10.5,
            fontweight="bold", color=MUTED)
    box(ax, 2, 38, 30, 11, ["shared secret", "32 bytes"], edge=ORANGE, fill=ORANGE_FILL)
    box(ax, 38, 38, 26, 11, ["HKDF-SHA256", "ties the key to", "this handshake"], edge=ORANGE, fill=ORANGE_FILL)
    box(ax, 70, 38, 28, 11, ["two AES-256 keys", "client->server", "server->client"], edge=ORANGE, fill=ORANGE_FILL)
    arrow(ax, 32, 43.5, 38, 43.5, colour=ORANGE)
    arrow(ax, 64, 43.5, 70, 43.5, colour=ORANGE)

    ax.text(2, 31, "Step C: encrypt the model update", fontsize=10.5, fontweight="bold", color=MUTED)
    box(ax, 2, 14, 24, 14, ["model update", "81 numbers", "324 bytes"], edge=BLUE, fill=BLUE_FILL)
    box(ax, 32, 14, 30, 14, ["AES-256-GCM", "encrypts the update", "and adds a 16-byte tag"],
        edge=ORANGE, fill=ORANGE_FILL)
    box(ax, 68, 14, 30, 14, ["encrypted message", "unreadable to anyone", "without the key"],
        edge=GREEN, fill=GREEN_FILL)
    arrow(ax, 26, 21, 32, 21, colour=BLUE)
    arrow(ax, 62, 21, 68, 21, colour=ORANGE)

    note(ax, 2, 3, 96, 8,
         "The tag is checked before decryption: if even one bit was changed on the way, the message is rejected.\n"
         "AES-256 stays strong against quantum attacks, which is why only the key exchange had to change.",
         colour=ORANGE, fill=ORANGE_FILL)
    save(figure, "3_mlkem_and_aes.png")


def main():
    system_architecture()
    pqc_handshake()
    mlkem_and_aes()


if __name__ == "__main__":
    main()
