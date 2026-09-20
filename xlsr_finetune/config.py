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

# Durable resume state = a private Kaggle dataset owned by the user.
STATE_SLUG = _env("AMH_STATE_SLUG", "halaza/amharic-xlsr-state")
STATE_MOUNT = os.path.join(INPUT_DIR, "amharic-xlsr-state")

OUTPUT_DIR = os.path.join(WORK_DIR, "xlsr_checkpoints")
PUSH_DIR = os.path.join(WORK_DIR, "xlsr_state_push")


@dataclass
class Settings:
    # --- data (same source as from-scratch pipeline) ---
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
    warmup_ratio: float = 0.06
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
