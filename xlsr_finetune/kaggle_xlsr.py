# Kaggle kernel entrypoint for the XLS-R/CTC fine-tune path.
# Runs ONE time-budgeted session, then pushes resumable state to the state dataset.
#
# Why this shape (learned from the first two ERROR runs):
#   * Kaggle's internet allowlist covers PyPI + HuggingFace but NOT github.com,
#     so `git clone` from GitHub fails with "Could not resolve host". We ship the
#     training code as a *mounted Kaggle dataset* (xlsr_code.zip) instead, which
#     needs no internet. Internet is still ON because XLS-R-300M weights come
#     from the HF Hub (allowlisted).
#   * Kaggle already ships torch/transformers/datasets/librosa/soundfile/
#     accelerate/evaluate, so we never force-reinstall pinned versions.
import os
import subprocess
import sys
import zipfile

WORK = "/kaggle/working"
CODE_ZIP = "/kaggle/input/amharic-xlsr-code/xlsr_code.zip"


def _sh(*cmd, check=True):
    print("[kaggle] $", " ".join(cmd))
    return subprocess.run(list(cmd), check=check)


# --- 1. unpack the training package from the mounted code dataset ---
extract = os.path.join(WORK, "src")
os.makedirs(extract, exist_ok=True)
with zipfile.ZipFile(CODE_ZIP) as z:
    z.extractall(extract)
pkg = os.path.join(extract, "xlsr_finetune")
assert os.path.isdir(pkg), f"xlsr_finetune not found under {extract}"
sys.path.insert(0, extract)
print("[kaggle] code unpacked:", sorted(os.listdir(pkg)))

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
import glob  # noqa: E402
inp = "/kaggle/input"
print("[preflight] /kaggle/input =", sorted(os.listdir(inp)))
corpus = os.path.join(inp, "amharic-speech-corpus", "AMHARIC", "data")
for split in ("train", "test"):
    d = os.path.join(corpus, split)
    wavs = glob.glob(os.path.join(d, "**", "*.wav"), recursive=True)
    txt = os.path.join(d, "text")
    n_text = sum(1 for _ in open(txt, encoding="utf-8")) if os.path.exists(txt) else 0
    first = open(txt, encoding="utf-8").readline().strip() if os.path.exists(txt) else ""
    print(f"[preflight] {split}: {len(wavs)} wavs, {n_text} text lines, sample: {first[:80]!r}")
if os.path.isdir(os.path.join(inp, "amharic-xlsr-state")):
    print("[preflight] state mounted:", sorted(os.listdir(os.path.join(inp, "amharic-xlsr-state"))))
print("[preflight] GPU:", subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip())

from xlsr_finetune.train import main  # noqa: E402

main()
