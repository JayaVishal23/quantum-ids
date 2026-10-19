"""
Shared settings for the whole project.

Every later script (client split, feature selection, training, PQC channel,
poisoning) reads its settings from this one file. That way all experiments
use the same clients, the same seeds and the same sample sizes, and we can
compare their results fairly.

If you change a value here, re-run every step that comes after it.
"""

from pathlib import Path


# ------------------------------------------------------------------
# Folders
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Each client's private data. Only that client's code may read its folder.
CLIENTS_DIR = PROJECT_ROOT / "data" / "clients"

# Things the server is allowed to hold: the agreed feature list and the
# scaling parameters. Never raw rows.
SHARED_DIR = PROJECT_ROOT / "data" / "shared"

# The official test set, used only for final evaluation.
TEST_DIR = PROJECT_ROOT / "data" / "test"

RESULTS_DIR = PROJECT_ROOT / "results"


# ------------------------------------------------------------------
# Dataset files
# ------------------------------------------------------------------

# The downloaded file names are swapped. The file called "testing-set"
# has 175,341 rows, which is the official size of the training set, so
# we train on it and test on the other one.
TRAIN_CSV = RAW_DIR / "UNSW_NB15_testing-set.csv"
TEST_CSV = RAW_DIR / "UNSW_NB15_training-set.csv"

EXPECTED_TRAIN_ROWS = 175_341
EXPECTED_TEST_ROWS = 82_332

LABEL_COLUMN = "label"          # 0 = normal, 1 = attack
CATEGORY_COLUMN = "attack_cat"  # used for splitting and analysis, never as input
ID_COLUMN = "id"
CATEGORICAL_COLUMNS = ["proto", "service", "state"]


# ------------------------------------------------------------------
# Seeds
# ------------------------------------------------------------------

# The client split is made once and never changes.
SPLIT_SEED = 42

# Every experiment is repeated once per seed; we report mean ± std.
SEEDS = [42, 43, 44]


# ------------------------------------------------------------------
# Federated clients
# ------------------------------------------------------------------

NUM_CLIENTS = 3

# Normal traffic is divided equally between the three clients.
# Each attack category belongs to exactly one client, so no client has
# seen the attacks the others have. That is the non-IID setting.
CLIENT_ATTACKS = {
    1: ["Generic", "Analysis"],
    2: ["Exploits", "Fuzzers"],
    3: ["DoS", "Reconnaissance", "Backdoor", "Shellcode", "Worms"],
}

# Each client keeps this share of its own data for validation
# (choosing the best round, early stopping). The test set is never used for that.
VALIDATION_FRACTION = 0.20


# ------------------------------------------------------------------
# Features
# ------------------------------------------------------------------

# The public schema: names of the 39 numeric features in UNSW-NB15.
# These are just column names from the dataset documentation, not data,
# so the server is allowed to publish them to every client.
# (proto, service and state are text columns and are not candidates.)
NUMERIC_FEATURES = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl",
    "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat",
    "smean", "dmean", "trans_depth", "response_body_len", "ct_srv_src",
    "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd",
    "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
]

# 8 features -> 8 qubits in the quantum model.
NUM_SELECTED_FEATURES = 8

# Features that measure volume or speed (bytes, load, rate, packet sizes,
# timings). Most values are small and a few are enormous, so they get
# log1p(x) = log(1 + x) before scaling. TTL values (0-255) and small
# counters are not heavy-tailed and are left as they are.
HEAVY_TAILED_FEATURES = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sload", "dload",
    "sinpkt", "dinpkt", "sjit", "djit", "smean", "dmean", "response_body_len",
]

# Every feature is scaled into this range, because each value becomes the
# angle of a rotation gate on one qubit (see ROADMAP section 10.3).
ANGLE_RANGE = (0.0, 3.141592653589793)

# Mutual information is slow on big tables, so each client scores
# features on a random sample of its rows.
MI_ROWS_PER_CLIENT = 20_000


# ------------------------------------------------------------------
# Subsample sizes
# ------------------------------------------------------------------

# Simulating an 8-qubit circuit is slow, so all models (classical and
# quantum) are trained and tested on the same smaller stratified sample.
TRAIN_ROWS_PER_CLIENT = 3_000
VAL_ROWS_PER_CLIENT = 750
TEST_ROWS = 5_000


# ------------------------------------------------------------------
# Quick self-check, so a typo in the client map is caught early
# ------------------------------------------------------------------

ALL_ATTACK_CATEGORIES = {
    "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance",
    "Analysis", "Backdoor", "Shellcode", "Worms",
}

_assigned = [cat for cats in CLIENT_ATTACKS.values() for cat in cats]

if len(_assigned) != len(set(_assigned)):
    raise ValueError("An attack category is given to more than one client.")

if set(_assigned) != ALL_ATTACK_CATEGORIES:
    missing = ALL_ATTACK_CATEGORIES - set(_assigned)
    raise ValueError(f"These attack categories have no client: {missing}")

if len(CLIENT_ATTACKS) != NUM_CLIENTS:
    raise ValueError("CLIENT_ATTACKS must have one entry per client.")
