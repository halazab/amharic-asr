# Kaggle kernel entrypoint for the XLS-R/CTC fine-tune path.
# Runs ONE time-budgeted session, then pushes resumable state to the state dataset.
#
# Why this shape (learned from the first three ERROR runs):
#   * Kaggle's internet allowlist covers PyPI + HuggingFace but NOT github.com,
#     so `git clone` from GitHub fails with "Could not resolve host". We ship the
#     training code as a *mounted Kaggle dataset* instead, which needs no
#     internet. Kaggle auto-extracts a lone zip, so the package mounts directly
#     at /kaggle/input/amharic-xlsr-code/xlsr_finetune/*.py. Internet is still ON
#     because XLS-R-300M weights come from the HF Hub (allowlisted).
#   * Kaggle already ships torch/transformers/datasets/librosa/soundfile/
#     accelerate/evaluate, so we never force-reinstall pinned versions.
import os
import subprocess
import sys

WORK = "/kaggle/working"
CODE_ROOT = "/kaggle/input/amharic-xlsr-code"

# SMOKE: flip to a short, tiny run purely to validate the end-to-end save+push of
# the ~1.2GB weight bundle to the state dataset. Set False for production.
SMOKE = False
if SMOKE:
    os.environ["AMH_TIME_BUDGET"] = "150"
    os.environ["AMH_MAX_TRAIN"] = "64"
    os.environ["AMH_MAX_EVAL"] = "32"
    os.environ["AMH_BS"] = "4"
    os.environ["AMH_GRAD_ACCUM"] = "1"
    os.environ["AMH_SAVE_STEPS"] = "1000"   # only the final save_model matters now


def _sh(*cmd, check=True):
    print("[kaggle] $", " ".join(cmd))
    return subprocess.run(list(cmd), check=check)


# --- 1. import the training package straight from the mounted code dataset ---
pkg = os.path.join(CODE_ROOT, "xlsr_finetune")
assert os.path.isdir(pkg), f"xlsr_finetune not mounted under {CODE_ROOT}: {os.listdir(CODE_ROOT)}"
sys.path.insert(0, CODE_ROOT)
print("[kaggle] code mounted:", sorted(os.listdir(pkg)))

# --- 2. install only genuinely-missing deps (never downgrade the base stack) ---
_missing = []
for mod, pkg_name in (("librosa", "librosa"), ("soundfile", "soundfile"), ("evaluate", "evaluate")):
    try:
        __import__(mod)
    except Exception:
        _missing.append(pkg_name)
if _missing:
    _sh(sys.executable, "-m", "pip", "install", "-q", *_missing, check=False)

os.environ["AMH_ASR_INPUT"] = "/kaggle/input"
os.environ["AMH_ASR_WORK"] = WORK

# --- 3. remote pre-flight probe: verify mounts + data shape (logged, never local) ---
import shutil  # noqa: E402
from xlsr_finetune import config as C  # noqa: E402

inp = "/kaggle/input"
print("[preflight] /kaggle/input =", sorted(os.listdir(inp)))
print("[preflight] resolved CORPUS_MOUNT =", C.CORPUS_MOUNT, "exists:", os.path.isdir(C.CORPUS_MOUNT))
print("[preflight] resolved STATE_MOUNT =", C.STATE_MOUNT, "exists:", os.path.isdir(C.STATE_MOUNT))
for split in ("train", "test"):
    d = os.path.join(C.CORPUS_MOUNT, split)
    wav_scp = os.path.join(d, "wav.scp")
    txt = os.path.join(d, "text")
    n_text = sum(1 for _ in open(txt, encoding="utf-8")) if os.path.exists(txt) else 0
    first_t = open(txt, encoding="utf-8").readline().strip() if os.path.exists(txt) else ""
    first_s = open(wav_scp, encoding="utf-8").readline().strip() if os.path.exists(wav_scp) else ""
    print(f"[preflight] {split}: text_lines={n_text}")
    print(f"[preflight]   text[0]={first_t[:90]!r}")
    print(f"[preflight]   wav.scp[0]={first_s[:90]!r}")
gpu = ""
if shutil.which("nvidia-smi"):
    gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
print("[preflight] GPU:", gpu)

from xlsr_finetune.train import main  # noqa: E402

main()
