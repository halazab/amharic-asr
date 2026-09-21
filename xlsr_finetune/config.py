"""Config for the XLS-R fine-tune path (self-contained; does not touch the
main from-scratch pipeline).

Dataset defaults to the same source the from-scratch pipeline uses
(HuggingFace `Harbidel/amharic-asr-merged` = WAXAL Amharic + merged sets), so
both approaches train on identical data and WER is directly comparable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


INPUT_DIR = _env("AMH_ASR_INPUT", "/kaggle/input")
WORK_DIR = _env("AMH_ASR_WORK", "/kaggle/working")


def find_mount(name: str, default: str, max_depth: int = 6) -> str:
    """Resolve a Kaggle-mounted dataset dir by *basename*, at any depth.

    Kaggle mounts kernel dataset sources inconsistently across runs — sometimes
    `/kaggle/input/<slug>`, sometimes `/kaggle/input/datasets/<owner>/<slug>` —
    so we never hardcode the layout. We walk INPUT_DIR and return the first
    directory whose basename matches `name`; `default` is used if not found
    (e.g. local/dev runs without mounts).
    """
    want = name.lower()
    for dirpath, dirnames, _files in os.walk(INPUT_DIR):
        depth = dirpath[len(INPUT_DIR):].count(os.sep)
        if depth >= max_depth:
            dirnames[:] = []
            continue
        for d in dirnames:
            if d.lower() == want:
                return os.path.join(dirpath, d)
    return default


# Durable resume state = a private Kaggle dataset owned by the user.
STATE_SLUG = _env("AMH_STATE_SLUG", "halaza/amharic-xlsr-state")
STATE_MOUNT = _env("AMH_STATE_MOUNT", find_mount(
    "amharic-xlsr-state", os.path.join(INPUT_DIR, "amharic-xlsr-state")))

# Kaggle-hosted Amharic corpus (public, mounts as a kernel input; no HF auth).
CORPUS_DIR = _env("AMH_CORPUS_DIR", find_mount(
    "amharic-speech-corpus", os.path.join(INPUT_DIR, "amharic-speech-corpus")))
CORPUS_MOUNT = os.path.join(CORPUS_DIR, "AMHARIC", "data")

OUTPUT_DIR = os.path.join(WORK_DIR, "xlsr_checkpoints")
PUSH_DIR = os.path.join(WORK_DIR, "xlsr_state_push")


@dataclass
class Settings:
    # --- data source: "kaggle_kaldi" (default, remote-native) or "hf" ---
    data_source: str = _env("AMH_DATA_SOURCE", "kaggle_kaldi")
    kaldi_train_dir: str = os.path.join(CORPUS_MOUNT, "train")
    kaldi_test_dir: str = os.path.join(CORPUS_MOUNT, "test")

    hf_dataset: str = field(default_factory=lambda: _env("AMH_HF_DATASET", "Harbidel/amharic-asr-merged"))
    text_columns: tuple = ("text", "transcription", "sentence", "target", "label")
    sampling_rate: int = 16000
    max_train_samples: int = int(_env("AMH_MAX_TRAIN", "0"))
    max_eval_samples: int = int(_env("AMH_MAX_EVAL", "200"))
    max_test_samples: int = int(_env("AMH_MAX_TEST", "0"))

    # --- model ---
    model_name: str = field(default_factory=lambda: _env("AMH_MODEL", "facebook/wav2vec2-xls-r-300m"))

    # --- optimization (fits 16GB P100/T4) ---
    batch_size: int = int(_env("AMH_BS", "16"))
    grad_accum: int = int(_env("AMH_GRAD_ACCUM", "4"))
    learning_rate: float = float(_env("AMH_LR", "1e-4"))
    # absolute warmup so it completes within a short per-session budget (a ratio
    # over 30 epochs never finishes in ~60 steps and keeps LR ~0 -> no learning)
    warmup_steps: int = int(_env("AMH_WARMUP_STEPS", "30"))
    max_audio_sec: float = float(_env("AMH_MAX_SEC", "20"))  # cap batch padding -> step speed
    num_epochs: int = int(_env("AMH_EPOCHS", "30"))
    freeze_feature_encoder: bool = _env("AMH_FREEZE_ENC", "1") == "1"

    # --- resilience (small checkpoints + auto-resume) ---
    save_steps: int = int(_env("AMH_SAVE_STEPS", "200"))
    eval_steps: int = int(_env("AMH_EVAL_STEPS", "200"))
    time_budget_sec: int = int(_env("AMH_TIME_BUDGET", "2700"))
    keep_last_n_checkpoints: int = int(_env("AMH_KEEP_CKPT", "2"))

    output_dir: str = OUTPUT_DIR
    push_dir: str = PUSH_DIR
    state_slug: str = STATE_SLUG
    state_mount: str = STATE_MOUNT


def get_settings() -> Settings:
    return Settings()
