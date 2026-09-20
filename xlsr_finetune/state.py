"""Resume/checkpoint persistence across Kaggle sessions.

Kaggle kills GPU sessions at any time and the writable FS does not survive, so
the durable resume point is a private Kaggle dataset (STATE_SLUG). Each session:
  pull_resume() -> train until time budget -> stage_for_push() -> push_state().
Losing a session costs at most `save_steps` of progress.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess


def _latest_checkpoint(root: str) -> str | None:
    if not os.path.isdir(root):
        return None
    ckpts = [
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and d.startswith("checkpoint-")
    ]
    return max(ckpts, key=lambda p: int(p.rsplit("-", 1)[-1])) if ckpts else None


def pull_resume(state_mount: str, output_dir: str) -> int:
    """Copy newest checkpoint from mounted state dataset into output_dir.
    Returns resumed global_step (0 if fresh).
    """
    if not os.path.isdir(state_mount):
        print(f"[state] no mounted state at {state_mount}; fresh start")
        return 0
    src = _latest_checkpoint(state_mount)
    if src is None:
        print("[state] state dataset empty; fresh start")
        return 0
    dst = os.path.join(output_dir, os.path.basename(src))
    if not os.path.exists(dst):
        shutil.copytree(src, dst)
    step = int(src.rsplit("-", 1)[-1])
    print(f"[state] resumed from {src} (step {step})")
    return step


def stage_for_push(output_dir: str, push_dir: str) -> str | None:
    ckpt = _latest_checkpoint(output_dir)
    if ckpt is None:
        if os.path.isdir(output_dir) and os.listdir(output_dir):
            ckpt = output_dir
        else:
            return None
    if os.path.exists(push_dir):
        shutil.rmtree(push_dir)
    shutil.copytree(ckpt, push_dir)
    print(f"[state] staged {ckpt} -> {push_dir}")
    return push_dir


def write_metrics(push_dir: str, metrics: dict) -> None:
    os.makedirs(push_dir, exist_ok=True)
    with open(os.path.join(push_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def push_state(push_dir: str, state_slug: str, message: str, title: str = "Amharic XLS-R state") -> bool:
    if not push_dir or not os.path.isdir(push_dir) or not os.listdir(push_dir):
        print("[state] nothing to push")
        return False
    meta = os.path.join(push_dir, ".kaggle", "dataset-metadata.json")
    os.makedirs(os.path.dirname(meta), exist_ok=True)
    if not os.path.exists(meta):
        with open(meta, "w", encoding="utf-8") as f:
            json.dump({"title": title, "id": state_slug, "isPrivate": True}, f)
    cmd = ["kaggle", "datasets", "version", "-p", push_dir, "-q", "-m", message]
    print("[state] pushing:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        print("[state] push OK")
        return True
    except FileNotFoundError:
        print("[state] kaggle CLI not found (expected on Kaggle)")
    except subprocess.CalledProcessError as e:
        print(f"[state] push failed: {e}")
    return False
