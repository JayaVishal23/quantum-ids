"""
Steps A5 and B: train a model on the pooled sample (the centralised reference).

"Centralised" means: pretend all three clients' samples are in one place.
That breaks privacy, so it's NOT part of our system. It's a reference line
that shows the best we could expect, so we can see what federated training
costs.

For every seed:
  - train on the pooled train sample (9,000 rows)
  - after each epoch, check the loss on the pooled validation sample
  - stop when validation loss hasn't improved for `patience` epochs,
    and go back to the best epoch (early stopping)
  - evaluate once on the test sample (5,000 rows)

Examples (from the project root):
    python -m training.train_centralized --model mlp
    python -m training.train_centralized --model tiny_mlp
    python -m training.train_centralized --model qnn
    python -m training.train_centralized --model qnn --seeds 42 --max-rows 200 --epochs 1   (timing test)
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from common import config, data, metrics
from models.factory import MODEL_NAMES, build_model, uses_wide_features
from training.loops import average_loss, copy_weights, predict_probability, train_one_epoch

# Sensible defaults per model. The QNN is much slower per step, so it gets
# fewer epochs and a higher learning rate (its parameters are angles).
DEFAULTS = {
    "mlp":      {"lr": 0.001, "batch_size": 64, "epochs": 100, "patience": 10},
    "tiny_mlp": {"lr": 0.001, "batch_size": 64, "epochs": 100, "patience": 10},
    "qnn":      {"lr": 0.01,  "batch_size": 32, "epochs": 30,  "patience": 5},
    "qnn_amplitude": {"lr": 0.01, "batch_size": 32, "epochs": 30, "patience": 5},
}

EXPERIMENT_FOLDER = {
    "qnn_amplitude": "e4_qnn_amplitude",
    "mlp": "e1_centralized",
    "tiny_mlp": "e1_centralized",
    "qnn": "e3_qnn_centralized",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=config.SEEDS)
    parser.add_argument("--epochs", type=int, help="maximum epochs (default depends on model)")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--max-rows", type=int, help="use only this many training rows (for timing tests)")
    parser.add_argument("--threshold", choices=metrics.THRESHOLD_RULES,
                        default="min_fpr_at_recall",
                        help="how to pick the decision threshold on validation data")
    parser.add_argument("--qnn-device", default="default.qubit")
    parser.add_argument("--qnn-diff-method", default="backprop")
    args = parser.parse_args()

    for key, value in DEFAULTS[args.model].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def take_first_rows(arrays, n):
    rng = np.random.default_rng(config.SPLIT_SEED)
    chosen = rng.choice(len(arrays["y"]), size=n, replace=False)
    return {name: values[chosen] for name, values in arrays.items()}


# ------------------------------------------------------------------
# One full run for one seed
# ------------------------------------------------------------------

def run_one_seed(args, seed, train, val, test):
    metrics.set_seed(seed)
    model = build_model(args.model, args.qnn_device, args.qnn_diff_method)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = data.make_loader(train, args.batch_size, shuffle=True, seed=seed)

    best_val_loss = float("inf")
    best_weights = copy_weights(model)
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(model, loader, optimizer, loss_fn)
        val_loss = average_loss(model, val, loss_fn)
        epoch_seconds = time.perf_counter() - epoch_start

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "seconds": epoch_seconds})
        print(f"  seed {seed} | epoch {epoch:3d} | train loss {train_loss:.4f} | "
              f"val loss {val_loss:.4f} | {epoch_seconds:.1f}s")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_weights = copy_weights(model)
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"  stopping early: no improvement for {args.patience} epochs")
                break

    training_seconds = time.perf_counter() - start
    model.load_state_dict(best_weights)

    probability = predict_probability(model, test["X"])
    result = metrics.evaluate(test["y"], probability)

    # The decision threshold comes from the validation split, never the test set.
    counts = metrics.confusion_counts_at_thresholds(
        val["y"], predict_probability(model, val["X"]))
    threshold = metrics.choose_threshold(counts, rule=args.threshold)
    tuned = metrics.evaluate(test["y"], probability, threshold=threshold)
    result["threshold_rule"] = args.threshold
    result["threshold"] = threshold
    result.update({f"tuned_{name}": tuned[name] for name in metrics.METRIC_NAMES})

    result.update({
        "seed": seed,
        "parameters": metrics.count_parameters(model),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "training_seconds": training_seconds,
        "seconds_per_epoch": training_seconds / len(history),
        "train_rows": len(train["y"]),
    })
    return result, pd.DataFrame(history), probability


def main():
    args = parse_args()

    wide = uses_wide_features(args.model)
    train = data.load_pooled("train", wide=wide)
    val = data.load_pooled("val", wide=wide)
    test = data.load_test(wide=wide)
    if args.max_rows:
        train = take_first_rows(train, args.max_rows)

    output_dir = config.RESULTS_DIR / "review1" / EXPERIMENT_FOLDER[args.model]
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model} | train {len(train['y']):,} rows | val {len(val['y']):,} | "
          f"test {len(test['y']):,} | lr {args.lr} | batch {args.batch_size} | "
          f"max epochs {args.epochs} | seeds {args.seeds}")

    if args.max_rows:
        # Timing test only: run, report speed, save nothing.
        result, history, _ = run_one_seed(args, args.seeds[0], train, val, test)
        rows_per_second = len(train["y"]) / history["seconds"].mean()
        print(f"\nTiming test: {rows_per_second:.1f} training rows per second (including validation)")
        full_rows = len(data.load_pooled("train", wide=wide)["y"])
        print(f"Estimated time per epoch on {full_rows:,} rows: {full_rows / rows_per_second / 60:.1f} min")
        return

    results = []
    for seed in args.seeds:
        result, history, probability = run_one_seed(args, seed, train, val, test)
        results.append(result)
        history.to_csv(output_dir / f"{args.model}_history_seed{seed}.csv", index=False)

        if seed == args.seeds[0]:
            # Figures and per-family recall for the first seed.
            title = f"{args.model} (centralised, seed {seed})"
            metrics.plot_confusion_matrix(test["y"], probability, title, figures_dir / f"{args.model}_confusion.png")
            metrics.plot_roc_curve(test["y"], probability, title, figures_dir / f"{args.model}_roc.png")
            metrics.plot_curves(history, "epoch", ["train_loss", "val_loss"], title, "loss",
                                figures_dir / f"{args.model}_loss.png")
            metrics.recall_per_attack_family(test["y"], probability, test["attack_cat"]).to_csv(
                output_dir / f"{args.model}_recall_per_family.csv", header=["share_correct"])

    per_seed = pd.DataFrame(results)
    per_seed.to_csv(output_dir / f"{args.model}_per_seed.csv", index=False)

    summary = metrics.summarize_seeds(results)
    summary.to_csv(output_dir / f"{args.model}_summary.csv")

    print(f"\n{args.model}: {int(per_seed['parameters'].iloc[0])} parameters, "
          f"{len(args.seeds)} seed(s)")
    for name, value in metrics.format_mean_std(summary).items():
        print(f"  {name:20s} {value}")
    print(f"  {'training seconds':20s} {summary.loc['training_seconds', 'mean']:.1f}")
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
