"""Resume/checkpoint persistence across Kaggle sessions.

Kaggle kills GPU sessions at any time and the writable FS does not survive, so
the durable resume point is a private Kaggle dataset (STATE_SLUG). Each session:
  pull_resume() -> train until time budget -> save compact weights ->
  stage_for_push() -> write_metrics() -> push_state().

We persist a *compact, weight-only* bundle (`state/` = model weights + config +
processor), NOT a full HF Trainer checkpoint (which carries optimizer.pt, ~3x
the size and fragile to upload). Resume loads those weights fresh; for a CTC
fine-tune climbing WER over many sessions this is exactly what we want, and it
keeps the durable state small per your "small checkpoints" requirement.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

STATE_MODEL_DIR = "state"        # weight-only model dir inside output_dir
STATE_PROCESSOR_DIR = "processor"
STATE_BUNDLE = "state.tar"       # single top-level file holding both dirs


def pull_resume(state_mount: str, output_dir: str) -> str | None:
    """Extract the durable weight bundle from the mounted state dataset into
    output_dir. Returns the local model dir to load, or None for a fresh start.

    We ship the weights as ONE top-level tarball (`state.tar`) because the
    in-kernel `kaggle` CLI silently drops *nested directories* on `datasets
    version` while uploading top-level files fine — a single file is the only
    reliable large-payload shape.
    """
    bundle = os.path.join(state_mount, STATE_BUNDLE)
    dst_model = os.path.join(output_dir, STATE_MODEL_DIR)
    if os.path.isdir(bundle) or os.path.isdir(os.path.join(state_mount, STATE_MODEL_DIR)):
        # legacy layout (dirs committed directly) — resume if weights present
        src_model = os.path.join(state_mount, STATE_MODEL_DIR)
        if os.path.isdir(src_model) and os.listdir(src_model) and not os.path.isdir(dst_model):
            shutil.copytree(src_model, dst_model)
    if not os.path.isfile(bundle):
        print("[state] no durable weights in state dataset; fresh start")
        return None
    if not os.path.isdir(dst_model) or not os.listdir(dst_model):
        os.makedirs(output_dir, exist_ok=True)
        import tarfile
        with tarfile.open(bundle, "r") as tar:
            try:
                tar.extractall(output_dir, filter="data")
            except TypeError:  # older tarfile without filter kwarg
                tar.extractall(output_dir)
        print(f"[state] extracted {bundle} -> {output_dir}")
    if os.path.isdir(dst_model) and os.listdir(dst_model):
        print(f"[state] resumed weights from {dst_model}")
        return dst_model
    print("[state] bundle present but no model dir after extract; fresh start")
    return None


def stage_for_push(output_dir: str, push_dir: str) -> str | None:
    """Pack the compact weight bundle (state/ + processor/) into a single
    top-level `state.tar` inside push_dir, ready for `kaggle datasets version`.
    """
    src_model = os.path.join(output_dir, STATE_MODEL_DIR)
    if not os.path.isdir(src_model) or not os.listdir(src_model):
        return None
    if os.path.exists(push_dir):
        shutil.rmtree(push_dir)
    os.makedirs(push_dir, exist_ok=True)
    import tarfile
    bundle = os.path.join(push_dir, STATE_BUNDLE)
    with tarfile.open(bundle, "w") as tar:  # no compression: safetensors won't shrink
        tar.add(src_model, arcname=STATE_MODEL_DIR)
        src_proc = os.path.join(output_dir, STATE_PROCESSOR_DIR)
        if os.path.isdir(src_proc):
            tar.add(src_proc, arcname=STATE_PROCESSOR_DIR)
    size_mb = os.path.getsize(bundle) / 1e6
    print(f"[state] packed {src_model} -> {bundle} ({size_mb:.0f} MB)")
    return push_dir


def write_metrics(push_dir: str, metrics: dict) -> None:
    os.makedirs(push_dir, exist_ok=True)
    with open(os.path.join(push_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def push_state(push_dir: str, state_slug: str, message: str, title: str = "Amharic XLS-R state") -> bool:
    if not push_dir or not os.path.isdir(push_dir) or not os.listdir(push_dir):
        print("[state] nothing to push")
        return False
    body = {"title": title, "id": state_slug, "isPrivate": True}
    # Kaggle's CLI looks for dataset-metadata.json in different places across
    # versions; write it to BOTH the dir root and .kaggle/ so `version` never
    # fails with "Metadata file not found".
    for meta in (os.path.join(push_dir, "dataset-metadata.json"),
                 os.path.join(push_dir, ".kaggle", "dataset-metadata.json")):
        os.makedirs(os.path.dirname(meta), exist_ok=True)
        if not os.path.exists(meta):
            with open(meta, "w", encoding="utf-8") as f:
                json.dump(body, f)
    cmd = ["kaggle", "datasets", "version", "-p", push_dir, "-q", "-m", message]
    files = []
    for dirpath, _dirs, fs in os.walk(push_dir):
        for fn in fs:
            if fn == "dataset-metadata.json":
                continue
            p = os.path.join(dirpath, fn)
            files.append((os.path.relpath(p, push_dir), os.path.getsize(p)))
    print(f"[state] about to upload {len(files)} file(s):")
    for rel, sz in files:
        print(f"[state]   {rel}  ({sz/1e6:.1f} MB)")
    print("[state] pushing:", " ".join(cmd))
    try:
        r = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[state] push OK")
        if r.stdout:
            print("[state] (kaggle stdout) " + r.stdout.strip()[:500])
        return True
    except FileNotFoundError:
        print("[state] kaggle CLI not found (expected on Kaggle)")
    except subprocess.CalledProcessError as e:
        print(f"[state] push failed rc={e.returncode}")
        print(f"[state] stderr: {(e.stderr or '').strip()[:800]}")
        print(f"[state] stdout: {(e.stdout or '').strip()[:800]}")
    return False
