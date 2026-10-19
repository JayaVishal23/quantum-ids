"""
What a malicious client does.

A malicious client is already part of the federation and holds valid keys, so
encryption cannot stop it: its messages are perfectly formed, only their
content is harmful. We implement the two standard attacks.

1. LABEL FLIPPING (data poisoning)
   The client trains on reversed labels, so it learns "attacks are normal".
   Its update then pulls the shared model towards missing attacks.

2. MODEL POISONING (update poisoning)
   The client trains normally, then sends the opposite of what it learned,
   multiplied by a boost factor, so its update overwhelms the honest ones.
       sent = global + (-boost) x (local - global)
"""

NO_ATTACK = "none"
LABEL_FLIP = "label_flip"
MODEL_POISON = "model_poison"

ATTACKS = [NO_ATTACK, LABEL_FLIP, MODEL_POISON]

BOOST = 5.0


def flip_labels(arrays):
    """Return the same data with every label reversed (0 becomes 1, 1 becomes 0)."""
    return {**arrays, "y": 1 - arrays["y"]}


def poison_update(global_weights, honest_weights, boost=BOOST):
    """Turn an honest update into a harmful one, pointing the other way."""
    return {
        name: global_weights[name] + (-boost) * (honest_weights[name] - global_weights[name])
        for name in honest_weights
    }
