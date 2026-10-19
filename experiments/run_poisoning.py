"""
The poisoning experiment (Objective 3b).

Runs every combination of:
    attack      : none, label flipping, model poisoning
    aggregation : FedAvg (no defence), coordinate-wise median (the defence)

with three seeds each, and saves the results under results/poisoning/.

The "none" row matters as much as the attacks: it shows what the defence costs
when nobody is cheating, which is a fair question with non-IID clients.

Run from the project root:
    python -m experiments.run_poisoning                      (quantum model)
    python -m experiments.run_poisoning --model mlp          (classical, much faster)
    python -m experiments.run_poisoning --rounds 10          (shorter)
"""

import argparse
import subprocess
import sys
import time

from attacks.poisoning import LABEL_FLIP, MODEL_POISON, NO_ATTACK

AGGREGATORS = ["fedavg", "median"]
ATTACKS = [NO_ATTACK, LABEL_FLIP, MODEL_POISON]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["qnn", "mlp"], default="qnn")
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--malicious-client", type=int, default=2)
    args = parser.parse_args()

    combinations = [(aggregator, attack) for attack in ATTACKS for aggregator in AGGREGATORS]
    print(f"{len(combinations)} runs: {args.model}, {args.rounds} rounds, 3 seeds each\n")

    started = time.perf_counter()
    for index, (aggregator, attack) in enumerate(combinations, start=1):
        print(f"[{index}/{len(combinations)}] aggregator={aggregator}  attack={attack}")
        command = [
            sys.executable, "-m", "fl.run_fl",
            "--model", args.model,
            "--rounds", str(args.rounds),
            "--lr", str(args.lr),
            "--local-epochs", str(args.local_epochs),
            "--aggregator", aggregator,
            "--attack", attack,
            "--malicious-client", str(args.malicious_client),
            "--out", "poisoning",
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout[-2000:])
            print(result.stderr[-2000:])
            raise SystemExit(f"run failed: {aggregator} / {attack}")

        for line in result.stdout.splitlines():
            if line.strip().startswith(("accuracy", "recall")):
                print("   ", line.strip())

    print(f"\nAll runs finished in {(time.perf_counter() - started) / 60:.0f} minutes.")
    print("Now run: python -m experiments.poisoning_summary")


if __name__ == "__main__":
    main()
