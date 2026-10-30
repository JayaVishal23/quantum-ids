# Review 1 results

Test sample of 5,000 rows. Mean ± standard deviation over 3 seeds. Same 8 features and same rows for every model.

| Model | Setting | Params | Accuracy | Precision | Recall | F1 | ROC-AUC | False-positive rate | Time (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| MLP | centralised | 289 | 81.6% ± 0.1% | 75.3% ± 0.2% | 99.1% ± 0.4% | 85.6% ± 0.0% | 0.943 ± 0.003 | 39.9% ± 0.6% | 18 |
| Tiny MLP | centralised | 81 | 80.4% ± 0.4% | 73.9% ± 0.4% | 99.3% ± 0.2% | 84.8% ± 0.3% | 0.933 ± 0.008 | 42.9% ± 1.0% | 16 |
| QNN | centralised | 81 | 80.4% ± 0.3% | 74.1% ± 0.3% | 99.1% ± 0.1% | 84.8% ± 0.2% | 0.938 ± 0.005 | 42.4% ± 0.8% | 362 |
| MLP | FedAvg, 3 clients | 289 | 81.9% ± 0.5% | 76.7% ± 0.9% | 96.7% ± 1.0% | 85.5% ± 0.3% | 0.938 ± 0.002 | 36.1% ± 2.1% | 23 |

Notes:
- The tiny MLP and the QNN both have 81 trainable parameters, so they compare like for like.
- The federated model's best round is chosen by the clients' validation loss, not the test set.
- The high false-positive rate affects every model the same way. It comes from a distribution
  shift between the official UNSW-NB15 training and test files (see notes_false_positive_rate.md).
