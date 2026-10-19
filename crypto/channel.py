"""
Secure channel for federated model updates.

WHAT IT PROTECTS AGAINST
  - Eavesdroppers, today and in the future ("harvest now, decrypt later"):
    the AES key comes from ML-KEM-768, which quantum computers can't break.
  - Man-in-the-middle: the server signs its public key, and every client
    already holds the server's real verification key ("pinning").
  - Impostor clients: each client signs the handshake with its registered key.
  - Tampering: AES-256-GCM rejects any changed byte.
  - Replay: fresh keys every round, round numbers checked, sequence numbers
    checked, and one accepted update per client per round.

THE HANDSHAKE (a new one every round)

  client                                              server
    |  hello: client_id, round, suite, client_nonce      |
    |--------------------------------------------------->|  new key pair
    |  public key, server_nonce, server signature        |
    |<---------------------------------------------------|
    |  verify signature with the pinned server key       |
    |  encapsulate -> ciphertext + shared secret         |
    |  ciphertext, client signature                      |
    |--------------------------------------------------->|  verify client signature
    |                                                    |  decapsulate -> same secret
    |        both: 2 AES keys = HKDF(secret, transcript) |
    |  update encrypted with AES-256-GCM                 |
    |--------------------------------------------------->|

The "transcript" is every handshake message so far, in order. Both signatures
cover it, so changing any message breaks a signature.

TWO KEY-EXCHANGE SUITES, so the cost of going post-quantum can be measured:
  "mlkem768"  ML-KEM-768, the post-quantum scheme standardised from Kyber
  "x25519"    the classical elliptic-curve exchange today's TLS uses

Signatures can use ML-DSA-65 (post-quantum) or Ed25519 (classical, smaller).
"""

import hashlib
import os

from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519, mldsa, mlkem, x25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

PROTOCOL_LABEL = b"fl-pqc-v1"
RANDOM_NONCE_BYTES = 32

SIGNATURE_SCHEMES = {
    "mldsa65": mldsa.MLDSA65PrivateKey,
    "ed25519": ed25519.Ed25519PrivateKey,
}


class ChannelError(Exception):
    """Raised whenever a message has to be rejected. The message says why."""


# ------------------------------------------------------------------
# The two key-exchange suites
# ------------------------------------------------------------------
#
# Both are used the same way, so the rest of the file never needs to know
# which one is running:
#     private_key           = suite.generate()
#     public_bytes          = suite.public_bytes(private_key)
#     secret, ciphertext    = suite.encapsulate(public_bytes)   (client side)
#     secret                = suite.decapsulate(private, ct)    (server side)

class MlKem768Suite:
    """Post-quantum key encapsulation (NIST FIPS 203)."""

    name = "mlkem768"

    @staticmethod
    def generate():
        return mlkem.MLKEM768PrivateKey.generate()

    @staticmethod
    def public_bytes(private_key):
        return private_key.public_key().public_bytes_raw()

    @staticmethod
    def encapsulate(public_bytes):
        public_key = mlkem.MLKEM768PublicKey.from_public_bytes(public_bytes)
        shared_secret, ciphertext = public_key.encapsulate()
        return shared_secret, ciphertext

    @staticmethod
    def decapsulate(private_key, ciphertext):
        return private_key.decapsulate(ciphertext)


class X25519Suite:
    """The classical exchange used by TLS today, our comparison baseline.

    X25519 is a Diffie-Hellman exchange, not a KEM, so we wrap it to look the
    same: the client's "ciphertext" is simply its own ephemeral public key, and
    both sides compute the shared secret from the other side's public key.
    """

    name = "x25519"

    @staticmethod
    def generate():
        return x25519.X25519PrivateKey.generate()

    @staticmethod
    def public_bytes(private_key):
        return private_key.public_key().public_bytes_raw()

    @staticmethod
    def encapsulate(public_bytes):
        server_public_key = x25519.X25519PublicKey.from_public_bytes(public_bytes)
        client_private_key = x25519.X25519PrivateKey.generate()
        shared_secret = client_private_key.exchange(server_public_key)
        return shared_secret, client_private_key.public_key().public_bytes_raw()

    @staticmethod
    def decapsulate(private_key, ciphertext):
        return private_key.exchange(x25519.X25519PublicKey.from_public_bytes(ciphertext))


KEY_EXCHANGE_SUITES = {
    MlKem768Suite.name: MlKem768Suite,
    X25519Suite.name: X25519Suite,
}

DEFAULT_KEY_EXCHANGE = MlKem768Suite.name


# ------------------------------------------------------------------
# Small building blocks
# ------------------------------------------------------------------

def pack(fields):
    """Turn a message (dict of bytes/ints) into bytes, in a fixed order.

    Used both to sign messages and to count how many bytes they take on the wire.
    Each field: 1 byte name length, name, 4 bytes value length, value.
    """
    out = bytearray()
    for name, value in fields.items():
        if isinstance(value, int):
            value = value.to_bytes(8, "big")
        name_bytes = name.encode("utf-8")
        out += len(name_bytes).to_bytes(1, "big") + name_bytes
        out += len(value).to_bytes(4, "big") + value
    return bytes(out)


def message_size(message):
    """How many bytes this message would take on the wire."""
    return len(pack(message))


def generate_signing_key(scheme):
    return SIGNATURE_SCHEMES[scheme].generate()


def signature_is_valid(public_key, signature, data):
    try:
        public_key.verify(signature, data)
        return True
    except InvalidSignature:
        return False


def derive_session_keys(shared_secret, transcript):
    """Two 32-byte AES keys from the shared secret, tied to this exact handshake.

    One key for client -> server, one for server -> client, so the two
    directions never share a key.
    """
    transcript_hash = hashlib.sha256(transcript).digest()
    key_material = HKDF(
        algorithm=hashes.SHA256(),
        length=64,
        salt=None,
        info=PROTOCOL_LABEL + transcript_hash,
    ).derive(shared_secret)
    return key_material[:32], key_material[32:]


def _aes_nonce(sequence_number):
    # Keys are new every round and each direction has its own key, so a
    # counter is enough to make sure a nonce is never reused with the same key.
    return sequence_number.to_bytes(12, "big")


def seal(key, header, payload):
    """Encrypt payload. The header isn't secret, but it is protected from changes."""
    return AESGCM(key).encrypt(_aes_nonce(header["seq"]), payload, pack(header))


def open_sealed(key, header, ciphertext):
    try:
        return AESGCM(key).decrypt(_aes_nonce(header["seq"]), ciphertext, pack(header))
    except InvalidTag:
        raise ChannelError("Decryption failed: the message or its header was changed.") from None


def _suite(name):
    if name not in KEY_EXCHANGE_SUITES:
        raise ValueError(f"Unknown key exchange '{name}'. Choose from {list(KEY_EXCHANGE_SUITES)}.")
    return KEY_EXCHANGE_SUITES[name]


# ------------------------------------------------------------------
# Client side
# ------------------------------------------------------------------

class ClientChannel:
    def __init__(self, client_id, signing_key, pinned_server_public_key,
                 key_exchange=DEFAULT_KEY_EXCHANGE):
        self.client_id = client_id
        self.signing_key = signing_key
        self.pinned_server_public_key = pinned_server_public_key
        self.suite = _suite(key_exchange)
        self._forget_session()

    def _forget_session(self):
        self.round = None
        self.transcript = b""
        self.send_key = None
        self.receive_key = None
        self.send_seq = 0
        self.last_received_seq = -1

    def start_handshake(self, round_number):
        self._forget_session()
        self.round = round_number
        hello = {
            "type": b"hello",
            "client_id": self.client_id,
            "round": round_number,
            "suite": self.suite.name.encode(),
            "client_nonce": os.urandom(RANDOM_NONCE_BYTES),
        }
        self.transcript = pack(hello)
        return hello

    def finish_handshake(self, server_reply):
        if server_reply["client_id"] != self.client_id or server_reply["round"] != self.round:
            raise ChannelError("Server reply is for a different client or round.")
        if server_reply["suite"] != self.suite.name.encode():
            raise ChannelError("Server replied with a different key-exchange suite.")

        signed_part = {name: server_reply[name] for name in
                       ("type", "client_id", "round", "suite", "kem_public_key", "server_nonce")}
        transcript = self.transcript + pack(signed_part)

        # The key step against man-in-the-middle attacks.
        if not signature_is_valid(self.pinned_server_public_key, server_reply["signature"], transcript):
            raise ChannelError("Server signature is invalid: the public key may have been "
                               "swapped by a man-in-the-middle. Handshake stopped.")

        shared_secret, kem_ciphertext = self.suite.encapsulate(server_reply["kem_public_key"])

        finish = {
            "type": b"finish",
            "client_id": self.client_id,
            "round": self.round,
            "kem_ciphertext": kem_ciphertext,
        }
        transcript += server_reply["signature"] + pack(finish)
        finish["signature"] = self.signing_key.sign(transcript)

        self.send_key, self.receive_key = derive_session_keys(shared_secret, transcript)
        self.transcript = transcript
        return finish

    def encrypt_update(self, payload):
        if self.send_key is None:
            raise ChannelError("Handshake not finished.")
        header = {"type": b"update", "client_id": self.client_id, "round": self.round, "seq": self.send_seq}
        self.send_seq += 1
        return {**header, "ciphertext": seal(self.send_key, header, payload)}

    def decrypt_global_model(self, message):
        if self.receive_key is None:
            raise ChannelError("Handshake not finished.")
        if message["type"] != b"global" or message["client_id"] != self.client_id or message["round"] != self.round:
            raise ChannelError("Global model message is for a different client or round.")
        if message["seq"] <= self.last_received_seq:
            raise ChannelError("Global model message was already received (replay).")

        header = {name: message[name] for name in ("type", "client_id", "round", "seq")}
        payload = open_sealed(self.receive_key, header, message["ciphertext"])
        self.last_received_seq = message["seq"]
        return payload


# ------------------------------------------------------------------
# Server side
# ------------------------------------------------------------------

class ServerChannel:
    def __init__(self, signing_key, registered_clients, key_exchange=DEFAULT_KEY_EXCHANGE):
        """registered_clients: {client_id: that client's public verification key}"""
        self.signing_key = signing_key
        self.registered_clients = registered_clients
        self.suite = _suite(key_exchange)
        self.current_round = None
        self.sessions = {}
        self.accepted_updates = set()  # (client_id, round) pairs already accepted

    def start_round(self, round_number):
        # Old sessions (and their keys) are thrown away every round.
        self.current_round = round_number
        self.sessions = {}

    def reply_to_hello(self, hello):
        client_id = hello["client_id"]

        if hello["type"] != b"hello":
            raise ChannelError("Expected a hello message.")
        if client_id not in self.registered_clients:
            raise ChannelError(f"Client {client_id} is not registered. Handshake refused.")
        if hello["round"] != self.current_round:
            raise ChannelError(f"Hello is for round {hello['round']}, but the current round is "
                               f"{self.current_round}: old or replayed message.")
        if hello.get("suite") != self.suite.name.encode():
            raise ChannelError("Client asked for a key-exchange suite the server doesn't use.")
        if (client_id, self.current_round) in self.accepted_updates:
            raise ChannelError(f"Client {client_id} already sent its update this round (replay).")

        # A brand-new key pair for this client and this round only.
        kem_private_key = self.suite.generate()

        signed_part = {
            "type": b"server_reply",
            "client_id": client_id,
            "round": self.current_round,
            "suite": self.suite.name.encode(),
            "kem_public_key": self.suite.public_bytes(kem_private_key),
            "server_nonce": os.urandom(RANDOM_NONCE_BYTES),
        }
        transcript = pack(hello) + pack(signed_part)
        signature = self.signing_key.sign(transcript)

        self.sessions[client_id] = {
            "kem_private_key": kem_private_key,
            "transcript": transcript + signature,
            "receive_key": None,
            "send_key": None,
            "last_received_seq": -1,
            "send_seq": 0,
        }
        return {**signed_part, "signature": signature}

    def complete_handshake(self, finish):
        client_id = finish["client_id"]
        session = self.sessions.get(client_id)

        if session is None or session["kem_private_key"] is None or finish["round"] != self.current_round:
            raise ChannelError(f"No handshake in progress for client {client_id} in round {self.current_round}.")

        unsigned = {name: finish[name] for name in ("type", "client_id", "round", "kem_ciphertext")}
        transcript = session["transcript"] + pack(unsigned)

        if not signature_is_valid(self.registered_clients[client_id], finish["signature"], transcript):
            raise ChannelError(f"Client {client_id}'s signature is invalid: an impostor, or a "
                               f"changed or replayed handshake. Rejected.")

        try:
            shared_secret = self.suite.decapsulate(session["kem_private_key"], finish["kem_ciphertext"])
        except ValueError:
            raise ChannelError("Key-exchange ciphertext is malformed.") from None

        # Delete the private key as soon as it has been used.
        session["kem_private_key"] = None
        session["receive_key"], session["send_key"] = derive_session_keys(shared_secret, transcript)

    def decrypt_update(self, message):
        client_id = message["client_id"]
        session = self.sessions.get(client_id)

        if session is None or session["receive_key"] is None:
            raise ChannelError(f"No finished handshake with client {client_id} this round.")
        if message["type"] != b"update" or message["round"] != self.current_round:
            raise ChannelError("Update is for a different round.")
        if (client_id, self.current_round) in self.accepted_updates:
            raise ChannelError(f"An update from client {client_id} was already accepted this round (replay).")
        if message["seq"] <= session["last_received_seq"]:
            raise ChannelError("Sequence number already used (replay).")

        header = {name: message[name] for name in ("type", "client_id", "round", "seq")}
        payload = open_sealed(session["receive_key"], header, message["ciphertext"])

        session["last_received_seq"] = message["seq"]
        self.accepted_updates.add((client_id, self.current_round))
        return payload

    def encrypt_global_model(self, client_id, payload):
        session = self.sessions.get(client_id)
        if session is None or session["send_key"] is None:
            raise ChannelError(f"No finished handshake with client {client_id} this round.")

        header = {"type": b"global", "client_id": client_id, "round": self.current_round, "seq": session["send_seq"]}
        session["send_seq"] += 1
        return {**header, "ciphertext": seal(session["send_key"], header, payload)}
