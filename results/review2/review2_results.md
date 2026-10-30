# Review 2 results: the quantum model inside federated learning

Test sample of 5,000 rows. Mean ± standard deviation over 3 seeds.
Same 8 features, same rows and same federated setup for every model.

| Model | Setting | Params | Update bytes | Accuracy | Precision | Recall | F1 | ROC-AUC | False-positive rate | Time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP | centralised | 289 | - | 81.6% ± 0.1% | 75.3% ± 0.2% | 99.1% ± 0.4% | 85.6% ± 0.0% | 0.943 ± 0.003 | 39.9% ± 0.6% | 18 |
| Tiny MLP | centralised | 81 | - | 80.4% ± 0.4% | 73.9% ± 0.4% | 99.3% ± 0.2% | 84.8% ± 0.3% | 0.933 ± 0.008 | 42.9% ± 1.0% | 16 |
| QNN | centralised | 81 | - | 80.4% ± 0.3% | 74.1% ± 0.3% | 99.1% ± 0.1% | 84.8% ± 0.2% | 0.938 ± 0.005 | 42.4% ± 0.8% | 362 |
| MLP | FedAvg, 60 rounds | 289 | 1,156 | 81.9% ± 0.5% | 76.7% ± 0.9% | 96.7% ± 1.0% | 85.5% ± 0.3% | 0.938 ± 0.002 | 36.1% ± 2.1% | 23 |
| MLP | FedAvg, 30 rounds | 289 | 1,156 | 82.4% ± 0.7% | 77.2% ± 0.9% | 96.8% ± 0.3% | 85.8% ± 0.5% | 0.939 ± 0.008 | 35.1% ± 1.9% | 7 |
| QNN | FedAvg, 30 rounds | 81 | 324 | 81.0% ± 0.6% | 75.2% ± 0.6% | 97.8% ± 0.2% | 85.0% ± 0.4% | 0.936 ± 0.002 | 39.5% ± 1.2% | 566 |

Notes:
- The QNN and the tiny MLP both have 81 parameters, so model size is not the difference.
- 'Update bytes' is what one client sends the server each round (parameters x 4 bytes).
- The two 30-round federated runs use exactly the same budget and settings
  (30 rounds, 1 local epoch, learning rate 0.01), so classical FL and QML-FL
  can be compared directly.
- The best round is chosen from the clients' validation loss, never the test set.
