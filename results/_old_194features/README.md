# Archived: first baseline runs (preliminary)

These are the results from the first version of the pipeline, archived on
2026-09-14. They are kept for reference and as "preliminary experiments".
They are **not** the final results.

## What is inside

| Folder | Produced by | Result |
|---|---|---|
| `classical/` | `models/classical/mlp.py`, `evaluate_mlp.py` | Centralised MLP: accuracy 87.20%, F1 89.28%, ROC-AUC 97.74% |
| `federated/` | `models/federated/fedavg.py`, `evaluate_fedavg.py` | FedAvg, 4 clients, 10 rounds: best round 87.48% (round 9), last round 87.06% |

The matching data is archived in:
- `data/_old_processed_centralized/`: training and test sets preprocessed together (194 features)
- `data/_old_4clients/`: the 4-client split

## Why they were replaced

1. **Preprocessing happened before the client split.** The scaler, the one-hot
   encoder and feature selection were fitted on all 175,341 training rows, so one
   party saw all the data. That breaks the privacy goal of federated learning.
2. **4 clients** instead of the agreed 3.
3. **194 features.** The QNN will use 8, so these baselines can't be compared with it.
4. **The best FedAvg round was chosen with the test set.** That makes 87.48% look
   better than it really is. The honest number is the last round, 87.06%.
5. **One seed only.** The gap between FedAvg and the MLP (0.28 points) is too small
   to mean anything without repeated runs.

## Don't re-run the old scripts

The old TensorFlow scripts that produced these results are now in
`archive/old_tensorflow_pipeline/`. They still write to the old folder names
(`data/processed`, `data/federated`, `results/classical`, `results/federated`), so
running them would recreate those folders and mix old outputs with new ones. The new
pipeline uses PyTorch and reads its settings from `common/config.py`.
