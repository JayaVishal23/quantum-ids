# Proof that encryption does not change the results (Oct 2026)

## Why this check exists

A reviewer can reasonably ask: "if you encrypt every model update, how do we know
the federated training still works the same?" Encryption should cost bytes and
milliseconds, and change nothing else. This check proves it.

## Method

The same federated run was done three times with **the same seed (42)**,
changing only how the updates travel:

| Mode | What happens to the weights |
|---|---|
| `none` | sent as plain weights (normal simulation) |
| `x25519` | X25519 key exchange + Ed25519/ML-DSA signatures + AES-256-GCM |
| `mlkem768` | ML-KEM-768 key exchange + signatures + AES-256-GCM |

Setup: classical MLP, 3 clients, 5 rounds, 1 local epoch, learning rate 0.01.

Command:

```bash
python -m fl.run_fl --model mlp --seeds 42 --rounds 5 --lr 0.01 --local-epochs 1 --secure mlkem768
```

## Result

| Mode | Accuracy | F1 | ROC-AUC | Bytes per round | Crypto time per round |
|---|--:|--:|--:|--:|--:|
| none | 0.7702 | 0.825882709501 | 0.877545570306 | 0 | 0 ms |
| x25519 | 0.7702 | 0.825882709501 | 0.877545570306 | 29,502 | 7.9 ms |
| mlkem768 | 0.7702 | 0.825882709501 | 0.877545570306 | 36,138 | 8.8 ms |

The metrics are identical to 12 significant figures. The weights that arrive at
the server after decryption are the same bits that left the client, so the
training path is unchanged.

**The only difference is cost**, which is exactly what the project set out to
measure:

- ML-KEM-768 costs **6,636 more bytes per round** than X25519 (about 2,212 bytes
  per client: a 1,184-byte public key and a 1,088-byte ciphertext, instead of two
  32-byte X25519 keys).
- Cryptography took about **1.7% of the round time** in both cases, because local
  training dominates.

These numbers come from a short 5-round run with the classical model. The full
measurement (per-operation timings, more rounds, the quantum model) is the next step.

## What to say about it

> "We ran the same seed with no encryption, with classical X25519 and with
> post-quantum ML-KEM. The accuracy is identical to twelve digits, which shows the
> channel delivers the weights unchanged. The post-quantum option costs about
> 6.6 KB more per round, and cryptography was under 2% of the round time."
