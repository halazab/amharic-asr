# Kaggle kernel entrypoint for the XLS-R/CTC fine-tune path.
# Runs ONE time-budgeted session, then pushes resumable state to the state dataset.
#
# Design notes (learned from the first ERROR run):
#   * git clone MUST happen before we can read anything from the repo, so no
#     pip install referencing repo paths before the clone.
#   * Kaggle already ships torch/transformers/datasets/librosa/soundfile/
#     accelerate/evaluate, so we do NOT force-reinstall pinned versions (that
#     risks downgrading transformers and breaking Wav2Vec2). We install only
#     packages that are genuinely missing, best-effort.
#   * The session MUST have internet enabled (XLS-R weights come from HF Hub);
#     `enable_internet: true` lives in kernel-metadata-xlsr.json.
import os
import subprocess
import sys

REPO_URL = os.environ.get("AMH_REPO_URL", "https://github.com/halazab/amharic-asr.git")
BRANCH = os.environ.get("AMH_BRANCH", "feat/xlsr-finetune")
WORK = "/kaggle/working"
repo = os.path.join(WORK, "repo")


def _sh(*cmd, check=True):
    print("[kaggle] $", " ".join(cmd))
    return subprocess.run(list(cmd), check=check)


# --- 1. fetch the training code ---
if os.path.isdir(os.path.join(repo, ".git")):
    _sh("git", "-C", repo, "fetch", "origin", BRANCH, check=False)
    _sh("git", "-C", repo, "checkout", BRANCH, check=False)
    _sh("git", "-C", repo, "reset", "--hard", f"origin/{BRANCH}", check=False)
else:
    _sh("git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, repo)
sys.path.insert(0, repo)

# --- 2. install only genuinely-missing deps (never downgrade the base stack) ---
_missing = []
for mod, pkg in (("librosa", "librosa"), ("soundfile", "soundfile"), ("evaluate", "evaluate")):
    try:
        __import__(mod)
    except Exception:
        _missing.append(pkg)
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
