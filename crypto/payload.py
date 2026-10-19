"""
Turning model weights into bytes and back.

Encryption works on bytes, not on PyTorch tensors, so before a client sends
its update the weights are flattened into bytes. The receiver rebuilds exactly
the same tensors.

Format:
    4 bytes      length of the header
    header       JSON with each tensor's name and shape
    body         all values as little-endian float32, one tensor after another
"""

import json

import numpy as np
import torch


def weights_to_bytes(weights):
    names = list(weights)
    header = json.dumps({
        "names": names,
        "shapes": [list(weights[name].shape) for name in names],
    }).encode("utf-8")

    body = b"".join(
        weights[name].detach().cpu().numpy().astype("<f4").tobytes()
        for name in names
    )
    return len(header).to_bytes(4, "big") + header + body


def bytes_to_weights(data):
    header_length = int.from_bytes(data[:4], "big")
    header = json.loads(data[4:4 + header_length])

    weights = {}
    offset = 4 + header_length
    for name, shape in zip(header["names"], header["shapes"]):
        count = int(np.prod(shape))
        values = np.frombuffer(data, dtype="<f4", count=count, offset=offset)
        weights[name] = torch.from_numpy(values.reshape(shape).copy())
        offset += count * 4

    if offset != len(data):
        raise ValueError("Payload length doesn't match its header.")
    return weights
