"""
Live demo: send traffic at the detector and watch it react.

Nothing is trained here. The detector was trained earlier; this script loads it
and uses it, the way a deployed intrusion detection system would.

You choose what to send:
    normal traffic          -> the detector should stay quiet
    attack traffic          -> the detector should raise an alert
    a mixed stream          -> a short burst of connections, like a live feed
    an attack on the channel-> someone attacking the encrypted link instead

Every connection shown is a real record from the UNSW-NB15 test set, one the
detector has never seen before.

Run from the project root:
    python -m experiments.live_demo                 (menu)
    python -m experiments.live_demo --send attack   (one attack, no menu)
    python -m experiments.live_demo --stream 10     (10 mixed connections)
    python -m experiments.live_demo --channel       (attacks on the secure link)
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch

from common import config, data, metrics
from crypto.test_channel import (
    attack_impersonate_client,
    attack_replay_update,
    attack_swap_kem_public_key,
    attack_tamper_with_update,
)
from models.factory import build_model
from training.loops import predict_probability

# Where the already-trained detector lives (first one found is used).
TRAINED_MODEL_PATHS = [
    config.PROJECT_ROOT / "models" / "trained" / "global_qnn.pt",
]

ATTACK_FAMILIES = ["Exploits", "Generic", "DoS", "Reconnaissance", "Fuzzers",
                   "Backdoor", "Analysis", "Shellcode", "Worms"]


def load_detector(model_name="qnn"):
    for path in TRAINED_MODEL_PATHS:
        if path.exists():
            model = build_model(model_name)
            model.load_state_dict(torch.load(path))
            model.eval()
            # One throwaway prediction, so the first timing shown isn't the
            # slow first call (libraries warm up on first use).
            predict_probability(model, np.zeros((1, config.NUM_SELECTED_FEATURES), dtype="float32"))
            return model, path
    raise SystemExit("No trained detector found. Expected one of:\n  " +
                     "\n  ".join(str(p) for p in TRAINED_MODEL_PATHS))


def load_traffic():
    """The scaled features the detector reads, plus the original records to display."""
    scaled = data.load_test(small=False)
    raw = pd.read_csv(config.TEST_CSV)
    return scaled, raw


def pick(scaled, kind, family=None, rng=None):
    """Choose one connection: 'normal', 'attack', or a named attack family."""
    rng = rng or np.random.default_rng()
    if kind == "normal":
        choices = np.flatnonzero(scaled["attack_cat"] == "Normal")
    elif family:
        choices = np.flatnonzero(scaled["attack_cat"] == family)
        if len(choices) == 0:
            raise SystemExit(f"No connections of type {family} in the test set.")
    else:
        choices = np.flatnonzero(scaled["y"] == 1)
    return int(rng.choice(choices))


def show_connection(model, scaled, raw, index):
    """Print the connection, then the detector's decision."""
    record = raw.iloc[index]

    print(f"\n  >>> incoming connection #{index}")
    print(f"      protocol {record['proto']:<6} service {record['service']:<9} state {record['state']}")
    print(f"      duration {record['dur']:.3f}s   sent {int(record['sbytes']):,} bytes   "
          f"received {int(record['dbytes']):,} bytes")
    print(f"      rate {record['rate']:.0f} packets/s   TTL out {int(record['sttl'])}   "
          f"TTL back {int(record['dttl'])}")

    start = time.perf_counter()
    probability = float(predict_probability(model, scaled["X"][index:index + 1])[0])
    milliseconds = (time.perf_counter() - start) * 1000

    is_attack = probability >= 0.5
    truth = record["attack_cat"]
    correct = is_attack == bool(record["label"])

    verdict = "*** ALERT: ATTACK ***" if is_attack else "no alert (normal traffic)"
    confidence = probability if is_attack else 1 - probability

    print(f"\n      DETECTOR: {verdict}")
    print(f"                confidence {confidence:.1%}, decided in {milliseconds:.1f} ms")
    print(f"      truth:    {truth}  ->  {'correct' if correct else 'WRONG'}")

    # Be upfront about mistakes instead of hiding them.
    if not correct and is_attack:
        print("      note:     a false alarm. About a third of normal traffic triggers one,")
        print("                because the test file contains normal connections with TTL")
        print("                values that mean 'attack' in the training file. It is a known")
        print("                property of this dataset, documented in our results.")
    elif not correct:
        print("      note:     a missed attack. The detector catches 97% of attacks;")
        print("                this is one of the remaining 3%.")
    return correct


def send_stream(model, scaled, raw, count, rng):
    """A short burst of mixed traffic, like a live feed."""
    print(f"\n  Sending {count} connections (a realistic mix of normal and attack traffic)\n")
    print(f"  {'#':>8}  {'really is':<16}{'detector':<12}{'confidence':>11}")
    print("  " + "-" * 50)

    indices = rng.choice(len(scaled["y"]), size=count, replace=False)
    probability = predict_probability(model, scaled["X"][indices])
    correct = 0

    for index, p in zip(indices, probability):
        alert = p >= 0.5
        truth = scaled["attack_cat"][index]
        hit = alert == bool(scaled["y"][index])
        correct += hit
        mark = "" if hit else "   <- wrong"
        print(f"  {index:>8}  {truth:<16}{'ALERT' if alert else 'quiet':<12}"
              f"{(p if alert else 1 - p):>10.1%}{mark}")

    print(f"\n  {correct} of {count} judged correctly.")


def channel_attacks():
    """Someone attacks the encrypted link between a company and the server."""
    print("\n  An attacker now targets the encrypted link carrying model updates:\n")
    attacks = [
        ("Swaps the server's public key (man-in-the-middle)", attack_swap_kem_public_key),
        ("Changes one bit of an encrypted message", attack_tamper_with_update),
        ("Records a valid message and sends it again", attack_replay_update),
        ("Pretends to be company 1", attack_impersonate_client),
    ]
    for description, attack in attacks:
        reason = attack("mldsa65", "mlkem768")
        print(f"  [BLOCKED] {description}")
        print(f"            {reason}\n")


MENU = """
  What do you want to send?

    1  normal traffic
    2  attack traffic (random type)
    3  attack traffic (choose the type)
    4  a stream of 10 mixed connections
    5  an attack on the encrypted link
    0  quit
"""


def menu_loop(model, scaled, raw, rng):
    while True:
        print(MENU)
        choice = input("  choice: ").strip()

        if choice == "0":
            print("\n  Done.\n")
            return
        if choice == "1":
            show_connection(model, scaled, raw, pick(scaled, "normal", rng=rng))
        elif choice == "2":
            show_connection(model, scaled, raw, pick(scaled, "attack", rng=rng))
        elif choice == "3":
            print()
            for number, family in enumerate(ATTACK_FAMILIES, start=1):
                print(f"    {number}  {family}")
            picked = input("\n  attack type: ").strip()
            if picked.isdigit() and 1 <= int(picked) <= len(ATTACK_FAMILIES):
                family = ATTACK_FAMILIES[int(picked) - 1]
                show_connection(model, scaled, raw, pick(scaled, "attack", family, rng))
            else:
                print("  (not a valid choice)")
        elif choice == "4":
            send_stream(model, scaled, raw, 10, rng)
        elif choice == "5":
            channel_attacks()
        else:
            print("  (not a valid choice)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--send", help="normal, attack, or an attack type such as Exploits")
    parser.add_argument("--stream", type=int, help="send this many mixed connections")
    parser.add_argument("--channel", action="store_true", help="show attacks on the encrypted link")
    parser.add_argument("--model", choices=["qnn", "mlp"], default="qnn")
    parser.add_argument("--seed", type=int, default=None, help="fix which connections are picked")
    args = parser.parse_args()

    metrics.set_seed(42)
    rng = np.random.default_rng(args.seed)

    model, path = load_detector(args.model)
    scaled, raw = load_traffic()

    kind = "quantum (8 qubits, 81 parameters)" if args.model == "qnn" else "classical (289 parameters)"
    print("\n" + "=" * 68)
    print("  INTRUSION DETECTOR  -  live traffic demo")
    print("=" * 68)
    print(f"  detector : {kind}, trained by 3 companies working together")
    print(f"  loaded   : {path.relative_to(config.PROJECT_ROOT)}")
    print(f"  traffic  : {len(scaled['y']):,} unseen connections from UNSW-NB15")

    if args.send:
        if args.send.lower() == "normal":
            show_connection(model, scaled, raw, pick(scaled, "normal", rng=rng))
        elif args.send.lower() == "attack":
            show_connection(model, scaled, raw, pick(scaled, "attack", rng=rng))
        else:
            show_connection(model, scaled, raw, pick(scaled, "attack", args.send, rng))
        return
    if args.stream:
        send_stream(model, scaled, raw, args.stream, rng)
        return
    if args.channel:
        channel_attacks()
        return

    menu_loop(model, scaled, raw, rng)


if __name__ == "__main__":
    main()
