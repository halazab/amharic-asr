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

from xlsr_finetune.train import main  # noqa: E402

main()
