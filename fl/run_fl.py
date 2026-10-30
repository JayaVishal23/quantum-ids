"""
Step A6 (and later C): run federated learning with 3 clients.

One round:
    server --global weights--> every client
    every client trains locally and sends back new weights
    server combines them (FedAvg)
    every client reports the new global model's loss on its own validation data
    server remembers the round with the lowest validation loss

The test set is used by a separate "evaluator" only to draw curves and to
report the final numbers. It never decides which round is best.

We report two results:
    best round  chosen by validation loss (the honest main number)
    last round  what you get if you just stop after the final round

Examples (from the project root):
    python -m fl.run_fl --model mlp
    python -m fl.run_fl --model qnn --secure mlkem768
"""

import argparse
import time

import pandas as pd
import torch

from attacks.poisoning import ATTACKS, NO_ATTACK
from common import config, data, metrics
from crypto.secure_fl import NO_ENCRYPTION, SecureTransport
from fl.aggregators import AGGREGATORS
from fl.client import Client
from fl.server import Server
from models.factory import MODEL_NAMES, uses_wide_features
from training.loops import predict_probability, weights_size_in_bytes

# Classical rounds are cheap, so they get enough rounds to finish improving
# (with 30 rounds the best round was still the last one, i.e. not converged).
DEFAULTS = {
    "mlp":      {"rounds": 60, "local_epochs": 2, "lr": 0.001, "batch_size": 64},
    "tiny_mlp": {"rounds": 60, "local_epochs": 2, "lr": 0.001, "batch_size": 64},
    "qnn":      {"rounds": 15, "local_epochs": 1, "lr": 0.01,  "batch_size": 32},
    "qnn_amplitude": {"rounds": 15, "local_epochs": 1, "lr": 0.01, "batch_size": 32},
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--aggregator", choices=list(AGGREGATORS), default="fedavg")
    parser.add_argument("--seeds", type=int, nargs="+", default=config.SEEDS)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--local-epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--out", default="runs", help="results sub-folder")
    parser.add_argument("--secure", choices=[NO_ENCRYPTION, "x25519", "mlkem768", "hybrid"], default=NO_ENCRYPTION,
                        help="how the model updates are protected on the way to the server")
    parser.add_argument("--threshold", choices=metrics.THRESHOLD_RULES,
                        default="min_fpr_at_recall",
                        help="how to pick the decision threshold from client validation counts; "
                             "half keeps the untuned 0.5")
    parser.add_argument("--signature", choices=["mldsa65", "ed25519"], default="mldsa65",
                        help="signature scheme used to authenticate the handshake")
    parser.add_argument("--attack", choices=ATTACKS, default=NO_ATTACK,
                        help="what the malicious client does")
    parser.add_argument("--malicious-client", type=int, default=2,
                        help="which client is malicious (default 2, the largest one)")
    parser.add_argument("--qnn-device", default="default.qubit")
    parser.add_argument("--qnn-diff-method", default="backprop")
    args = parser.parse_args()

    for key, value in DEFAULTS[args.model].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def run_one_seed(args, seed, test):
    metrics.set_seed(seed)  # same seed -> same starting global model

    server = Server(args.model, args.aggregator, args.qnn_device, args.qnn_diff_method)
    clients = [
        Client(client_id, args.model, args.qnn_device, args.qnn_diff_method,
               attack=args.attack if client_id == args.malicious_client else NO_ATTACK)
        for client_id in config.CLIENT_ATTACKS
    ]
    transport = SecureTransport(list(config.CLIENT_ATTACKS), args.secure, args.signature)

    history = []
    for round_number in range(1, args.rounds + 1):
        round_start = time.perf_counter()
        global_weights = server.global_weights()
        transport.start_round(round_number)

        # 1-2. Local training on every client. With --secure, the global model
        # and the update travel through the encrypted channel.
        updates = []
        for client in clients:
            received_global = transport.send_global_model(client.client_id, global_weights)
            shuffle_seed = seed * 1000 + round_number * 10 + client.client_id
            update = client.local_train(received_global, args.local_epochs, args.lr,
                                        args.batch_size, shuffle_seed)
            update["weights"] = transport.send_update(client.client_id, update["weights"])
            updates.append(update)

        # 3. Server combines the updates.
        server.aggregate(updates)
        new_global = server.global_weights()

        # 4. Clients check the new global model on their own validation data.
        reports = [client.evaluate_global(new_global) for client in clients]
        val_loss = server.record_validation(round_number, reports)
        round_seconds = time.perf_counter() - round_start

        # Evaluator: test metrics for the curves only.
        test_result = metrics.evaluate(test["y"], predict_probability(server.model, test["X"]))

        history.append({
            "round": round_number,
            "val_loss": val_loss,
            "test_accuracy": test_result["accuracy"],
            "test_f1": test_result["f1"],
            "test_recall": test_result["recall"],
            "mean_client_train_loss": sum(u["train_loss"] for u in updates) / len(updates),
            "local_training_seconds": sum(u["seconds"] for u in updates),
            "round_seconds": round_seconds,
            "update_bytes_per_client": weights_size_in_bytes(updates[0]["weights"]),
            **transport.round_cost(),
        })
        marker = " <- best so far" if server.best_round == round_number else ""
        print(f"  seed {seed} | round {round_number:2d} | val loss {val_loss:.4f} | "
              f"test acc {test_result['accuracy']:.4f} | {round_seconds:.1f}s{marker}")

    history = pd.DataFrame(history)

    # Final numbers: best round (chosen by validation) and last round.
    last_probability = predict_probability(server.model, test["X"])
    server.model.load_state_dict(server.best_weights)
    best_probability = predict_probability(server.model, test["X"])

    result = metrics.evaluate(test["y"], best_probability)
    last = metrics.evaluate(test["y"], last_probability)
    result.update({f"last_round_{name}": last[name] for name in metrics.METRIC_NAMES})

    # The decision threshold, chosen from the clients' validation counts at the
    # best round. The untuned 0.5 numbers above are kept so the two can be
    # compared; "tuned_" are the same model read at a better operating point.
    threshold = metrics.DEFAULT_THRESHOLD
    if server.best_threshold_counts is not None:
        threshold = metrics.choose_threshold(server.best_threshold_counts, rule=args.threshold)
    tuned = metrics.evaluate(test["y"], best_probability, threshold=threshold)
    result["threshold_rule"] = args.threshold
    result["threshold"] = threshold
    result.update({f"tuned_{name}": tuned[name] for name in metrics.METRIC_NAMES})
    result.update({
        "seed": seed,
        "parameters": metrics.count_parameters(server.model),
        "best_round": server.best_round,
        "rounds": args.rounds,
        "total_seconds": history["round_seconds"].sum(),
        "update_bytes_per_client": int(history["update_bytes_per_client"].iloc[0]),
        # What the security cost: averages over the rounds, summed over the 3 clients.
        "secure": args.secure,
        "signature": args.signature if args.secure != NO_ENCRYPTION else "-",
        "aggregator": args.aggregator,
        "attack": args.attack,
        "malicious_client": args.malicious_client if args.attack != NO_ATTACK else 0,
        "handshake_bytes_per_round": history["handshake_bytes"].mean(),
        "upload_bytes_per_round": history["upload_bytes"].mean(),
        "download_bytes_per_round": history["download_bytes"].mean(),
        "total_bytes_per_round": history["total_bytes"].mean(),
        "crypto_seconds_per_round": history["crypto_seconds"].mean(),
        "crypto_share_of_round": history["crypto_seconds"].sum() / history["round_seconds"].sum(),
    })
    return result, history, best_probability, server.best_weights, server.best_threshold_counts


def threshold_study(threshold_inputs, y_true):
    """Every threshold rule read off the same trained models.

    Training does not depend on the rule, so one run answers the question for
    all of them. Each rule is chosen from client validation counts only.
    """
    usable = [(seed, probability, counts) for seed, probability, counts in threshold_inputs
              if counts is not None]
    if not usable:
        return None

    rows = []
    for rule in metrics.THRESHOLD_RULES:
        for seed, probability, counts in usable:
            threshold = metrics.choose_threshold(counts, rule=rule)
            scores = metrics.evaluate(y_true, probability, threshold=threshold)
            rows.append({"rule": rule, "seed": seed, "threshold": threshold,
                         **{name: scores[name] for name in metrics.METRIC_NAMES}})

    table = pd.DataFrame(rows)
    averaged = table.groupby("rule", sort=False).mean(numeric_only=True).drop(columns="seed").reset_index()
    print()
    print("Decision threshold, chosen on client validation data only:")
    print(averaged.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return table


def main():
    args = parse_args()
    test = data.load_test(wide=uses_wide_features(args.model))

    name = f"fl_{args.model}_{args.aggregator}"
    if args.attack != NO_ATTACK:
        name += f"_{args.attack}"
    if args.secure != NO_ENCRYPTION:
        # Separate folder per security setting, so runs don't overwrite each other.
        name += f"_{args.secure}_{args.signature}"
    output_dir = config.RESULTS_DIR / args.out / name
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Federated: model {args.model} | {args.aggregator} | {len(config.CLIENT_ATTACKS)} clients | "
          f"{args.rounds} rounds x {args.local_epochs} local epoch(s) | lr {args.lr} | "
          f"batch {args.batch_size} | seeds {args.seeds}")
    if args.secure != NO_ENCRYPTION:
        print(f"Channel: {args.secure} key exchange + {args.signature} signatures + AES-256-GCM, "
              f"fresh handshake every round")
    else:
        print("Channel: no encryption (updates sent as plain weights)")
    if args.attack != NO_ATTACK:
        print(f"Attack:  client {args.malicious_client} is malicious ({args.attack})")

    results = []
    threshold_inputs = []
    for seed in args.seeds:
        result, history, probability, best_weights, counts = run_one_seed(args, seed, test)
        results.append(result)
        threshold_inputs.append((seed, probability, counts))
        history.to_csv(output_dir / f"history_seed{seed}.csv", index=False)

        if seed == args.seeds[0]:
            title = f"FL {args.model} {args.aggregator} (seed {seed}, best round {result['best_round']})"
            metrics.plot_confusion_matrix(test["y"], probability, title, figures_dir / "confusion.png")
            metrics.plot_roc_curve(test["y"], probability, title, figures_dir / "roc.png")
            metrics.plot_curves(history, "round", ["test_accuracy", "test_f1"], f"FL {args.model}: test metrics per round",
                                "score", figures_dir / "test_metrics_per_round.png")
            metrics.plot_curves(history, "round", ["val_loss", "mean_client_train_loss"], f"FL {args.model}: loss per round",
                                "loss", figures_dir / "loss_per_round.png")
            metrics.recall_per_attack_family(test["y"], probability, test["attack_cat"]).to_csv(
                output_dir / "recall_per_family.csv", header=["share_correct"])
            # Kept so the PQC demo can send a real trained model through the channel.
            torch.save(best_weights, output_dir / f"global_model_seed{seed}.pt")

    study = threshold_study(threshold_inputs, test["y"])
    if study is not None:
        study.to_csv(output_dir / "threshold_study.csv", index=False)

    per_seed = pd.DataFrame(results)
    per_seed.to_csv(output_dir / "per_seed.csv", index=False)
    summary = metrics.summarize_seeds(results)
    summary.to_csv(output_dir / "summary.csv")

    print(f"\n{name}: {int(per_seed['parameters'].iloc[0])} parameters, "
          f"update size {int(per_seed['update_bytes_per_client'].iloc[0])} bytes per client")

    if args.secure != NO_ENCRYPTION:
        print("Security cost per round (all 3 clients together):")
        print(f"  handshake            {summary.loc['handshake_bytes_per_round', 'mean']:>10,.0f} bytes")
        print(f"  encrypted uploads    {summary.loc['upload_bytes_per_round', 'mean']:>10,.0f} bytes")
        print(f"  encrypted downloads  {summary.loc['download_bytes_per_round', 'mean']:>10,.0f} bytes")
        print(f"  total on the wire    {summary.loc['total_bytes_per_round', 'mean']:>10,.0f} bytes")
        print(f"  cryptography time    {summary.loc['crypto_seconds_per_round', 'mean'] * 1000:>10,.1f} ms "
              f"({summary.loc['crypto_share_of_round', 'mean']:.1%} of the round)")
    print("Best round (chosen by validation loss):")
    for metric, value in metrics.format_mean_std(summary).items():
        print(f"  {metric:20s} {value}")
    print(f"  best rounds per seed: {per_seed['best_round'].tolist()}")
    if "threshold" in summary.index:
        rule = per_seed["threshold_rule"].iloc[0]
        print(f"Same model read at a tuned threshold ({rule}, "
              f"{summary.loc['threshold', 'mean']:.2f} from client validation data):")
        tuned_names = [f"tuned_{m}" for m in metrics.METRIC_NAMES]
        for metric, value in metrics.format_mean_std(summary, tuned_names).items():
            print(f"  {metric:32s} {value}")
    print("Last round:")
    last_names = [f"last_round_{m}" for m in metrics.METRIC_NAMES]
    for metric, value in metrics.format_mean_std(summary, last_names).items():
        print(f"  {metric:32s} {value}")
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
