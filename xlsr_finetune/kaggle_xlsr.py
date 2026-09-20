# Kaggle kernel entrypoint for the XLS-R/CTC fine-tune path.
# Runs ONE time-budgeted session, then pushes resumable state to the state dataset.
import os
import subprocess
import sys

REPO_URL = os.environ.get("AMH_REPO_URL", "https://github.com/halazab/amharic-asr.git")
BRANCH = os.environ.get("AMH_BRANCH", "feat/xlsr-finetune")
WORK = "/kaggle/working"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "xlsr_finetune/requirements.txt"], check=False)

repo = os.path.join(WORK, "repo")
if os.path.isdir(repo):
    subprocess.run(["git", "-C", repo, "fetch", "origin", BRANCH], check=False)
    subprocess.run(["git", "-C", repo, "checkout", BRANCH], check=False)
    subprocess.run(["git", "-C", repo, "reset", "--hard", f"origin/{BRANCH}"], check=False)
else:
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, repo], check=True)
sys.path.insert(0, repo)

os.environ["AMH_ASR_INPUT"] = "/kaggle/input"
os.environ["AMH_ASR_WORK"] = WORK

# --- remote pre-flight probe: verify mounts + data shape (logged, never local) ---
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
