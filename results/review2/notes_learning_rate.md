# Why the classical federated baseline was re-run (finding, Oct 2026)

## What happened

To compare classical FL and QML-FL fairly, both must get the same budget:
30 rounds, 1 local epoch. The first matched run used each model's old default
learning rate, and the result looked like a big win for the quantum model:

| Run | Learning rate | Accuracy (3 seeds) | ROC-AUC |
|---|--:|--:|--:|
| Classical FL (MLP) | 0.001 | 76.9% ± 1.7% | 0.888 ± 0.023 |
| QML-FL (QNN) | 0.01 | 81.0% ± 0.6% | 0.936 ± 0.002 |

That gap is **not** a quantum advantage. The MLP's best round was 28, 30 and 30
out of 30, which means it was still improving when the run ended: with a learning
rate of 0.001 it simply hadn't finished training yet.

## The check

We re-ran the classical model with the **same learning rate as the QNN** (0.01),
everything else identical:

| Run | Learning rate | Accuracy (3 seeds) | ROC-AUC |
|---|--:|--:|--:|
| Classical FL (MLP) | 0.01 | **82.4% ± 0.7%** | 0.939 ± 0.008 |
| QML-FL (QNN) | 0.01 | 81.0% ± 0.6% | 0.936 ± 0.002 |

With the learning rate matched, the classical model is slightly **ahead** of the
quantum one, by about 1.4 points.

## Decision

The results in `review2_results.md` use the learning-rate-matched run (0.01 for
both models). The earlier 0.001 run is kept only in this note.

## What to say about it

> "When we first matched the round budget, the quantum model looked 4 points
> better. We checked why, and it was the learning rate: the classical model had
> not converged at 0.001. With the learning rate matched, the classical model is
> slightly ahead (82.4% vs 81.0%). So our claim is that the quantum model is
> competitive with a classical one, using 4x fewer parameters, not that it is
> better."

This matters because the quantum model sends **324 bytes** per round against the
MLP's **1,156 bytes**, which is the trade-off worth reporting.
