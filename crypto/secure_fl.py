"""
Puts the secure channel inside federated learning.

Without this file the channel is only a demo; with it, every model update that
moves between a client and the server is really encrypted.

One round looks like this:

    transport.start_round(r)
    for each client:
        handshake                      (new keys, signatures checked both ways)
        server -> client: global model (encrypted)
        client trains locally
        client -> server: its update   (encrypted)

The transport also counts what the security costs: bytes on the wire and
seconds spent on cryptography. Those numbers are the point of Objective 3.

Setting `key_exchange="none"` turns encryption off and passes the weights
straight through, so the same federated code runs in all three modes:
    none       no encryption at all (the "before" picture)
    x25519     today's classical key exchange (the comparison baseline)
    mlkem768   post-quantum key exchange (what we propose)
"""

import time

from crypto.channel import ClientChannel, ServerChannel, generate_signing_key, message_size
from crypto.payload import bytes_to_weights, weights_to_bytes

NO_ENCRYPTION = "none"


class RoundCost:
    """What one federated round cost in bytes and seconds, summed over clients."""

    def __init__(self):
        self.handshake_bytes = 0
        self.upload_bytes = 0
        self.download_bytes = 0
        self.crypto_seconds = 0.0

    def as_dict(self):
        return {
            "handshake_bytes": self.handshake_bytes,
            "upload_bytes": self.upload_bytes,
            "download_bytes": self.download_bytes,
            "total_bytes": self.handshake_bytes + self.upload_bytes + self.download_bytes,
            "crypto_seconds": self.crypto_seconds,
        }


class SecureTransport:
    """Carries model weights between the clients and the server.

    With key_exchange="none" it hands the weights over unchanged, which is how
    federated learning normally works in a simulation.
    """

    def __init__(self, client_ids, key_exchange=NO_ENCRYPTION, signature="mldsa65"):
        self.key_exchange = key_exchange
        self.signature = signature
        self.cost = RoundCost()

        if key_exchange == NO_ENCRYPTION:
            self.server = None
            self.clients = {}
            return

        # Enrolment, done once before training:
        #   the server keeps every client's public key,
        #   every client keeps the server's public key (pinning).
        server_signing_key = generate_signing_key(signature)
        client_signing_keys = {cid: generate_signing_key(signature) for cid in client_ids}

        self.server = ServerChannel(
            server_signing_key,
            registered_clients={cid: key.public_key() for cid, key in client_signing_keys.items()},
            key_exchange=key_exchange,
        )
        self.clients = {
            cid: ClientChannel(cid, key, server_signing_key.public_key(), key_exchange=key_exchange)
            for cid, key in client_signing_keys.items()
        }

    @property
    def encrypts(self):
        return self.key_exchange != NO_ENCRYPTION

    def start_round(self, round_number):
        self.cost = RoundCost()
        if self.encrypts:
            self.server.start_round(round_number)
            self._round = round_number
            self._ready = set()

    def _handshake(self, client_id):
        """Fresh keys for this client and this round."""
        client = self.clients[client_id]
        start = time.perf_counter()

        hello = client.start_handshake(self._round)
        reply = self.server.reply_to_hello(hello)
        finish = client.finish_handshake(reply)   # checks the server's signature
        self.server.complete_handshake(finish)    # checks the client's signature

        self.cost.crypto_seconds += time.perf_counter() - start
        self.cost.handshake_bytes += message_size(hello) + message_size(reply) + message_size(finish)
        self._ready.add(client_id)

    def send_global_model(self, client_id, weights):
        """Server -> client. Returns the weights as the client receives them."""
        if not self.encrypts:
            return weights

        if client_id not in self._ready:
            self._handshake(client_id)

        start = time.perf_counter()
        message = self.server.encrypt_global_model(client_id, weights_to_bytes(weights))
        received = bytes_to_weights(self.clients[client_id].decrypt_global_model(message))
        self.cost.crypto_seconds += time.perf_counter() - start
        self.cost.download_bytes += message_size(message)
        return received

    def send_update(self, client_id, weights):
        """Client -> server. Returns the weights as the server receives them."""
        if not self.encrypts:
            return weights

        start = time.perf_counter()
        message = self.clients[client_id].encrypt_update(weights_to_bytes(weights))
        received = bytes_to_weights(self.server.decrypt_update(message))
        self.cost.crypto_seconds += time.perf_counter() - start
        self.cost.upload_bytes += message_size(message)
        return received

    def round_cost(self):
        return self.cost.as_dict()
