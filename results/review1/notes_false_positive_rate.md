# Why is the false-positive rate high? (finding from 14 Sep 2026)

## What we saw
With the 8 agreed features, the centralised MLP reaches about 99% recall but
flags about **40% of normal test traffic** as attacks. The old 194-feature model
flagged about 18.5%.

## Is it a bug?
No. The same model on the **validation** data (which comes from the training
file) has a false-positive rate of 19% and accuracy of 92%. The drop happens
only on the official **test** file, so the test file's traffic is distributed
differently from the training file's.

## Evidence: the TTL shortcut
| `sttl = 254` | Training file | Test file |
|---|--:|--:|
| share of normal rows | 20% | 41% |
| share of attack rows | 87% | — |

In the training file, `sttl = 254` almost always means "attack", so the model
learns that rule. In the test file, many normal rows also have `sttl = 254`.
**85% of the wrongly flagged normal rows have `sttl = 254`.**

## Does removing the TTL features fix it?
We re-ran feature selection with `sttl`, `dttl` and `ct_state_ttl` excluded.
The new top 8 were `sbytes, smean, sload, dbytes, rate, dur, dmean, dinpkt`.

| Features | Test accuracy | False-positive rate | ROC-AUC |
|---|--:|--:|--:|
| Agreed 8 (with TTL) | 81.6% | 39.7% | 0.943 |
| 8 without TTL (2 seeds) | 78.3% | 43.6–46.4% | 0.911–0.916 |

Removing TTL makes results **worse**. The shift between the two files affects
other features too, and 8 features aren't enough to separate the classes the
way 194 did.

## Decision
Keep the 8 features chosen by federated mutual information, and report the
false-positive rate honestly as a limitation. It comes from the dataset and the
small feature budget (8 features for 8 qubits), not from federated learning or
the quantum model. Every model in the project is affected in the same way, so
the comparisons between models stay fair.

## How to explain it in the viva
> "Our models catch about 99% of attacks but raise false alarms on about 40% of
> normal traffic. We checked why: the official UNSW-NB15 training and test files
> are distributed differently. For example, a TTL value of 254 marks 87% of
> attacks in training but also 41% of normal rows in testing. We tried removing
> the TTL features and results got worse, so we kept the data-driven choice and
> report this as a limitation of using only 8 features."
