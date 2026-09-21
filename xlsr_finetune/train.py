"""Fine-tune XLS-R (Wav2Vec2) + CTC on the merged Amharic dataset, resumable.

Run: python -m xlsr_finetune.train
Launch once per Kaggle session. Each launch pulls the compact weight bundle from
the durable state dataset, continues training until the time budget, evaluates
WER/CER, saves weights back, and pushes state to Kaggle. Weight-only resume (no
optimizer) keeps the durable state small and uploads reliable.
"""
from __future__ import annotations

import json
import os
import time

import torch
from transformers import Trainer, TrainingArguments, Wav2Vec2ForCTC, TrainerCallback

from . import state as st
from .config import Settings, get_settings
from .data import DataCollatorCTCWithPadding, build_processor, build_vocab, load_datasets
from .metrics import compute_metrics


class TimeBudgetCallback(TrainerCallback):
    def __init__(self, budget_sec: int):
        self.budget = budget_sec
        self.t0 = time.time()
        self._done = False

    def on_step_end(self, args, state, control, **kwargs):
        if not self._done and (time.time() - self.t0) > self.budget:
            print(f"[train] time budget {self.budget}s hit -> saving + exiting cleanly")
            self._done = True
            control.should_training_stop = True
        return control


def load_history(state_mount: str) -> list:
    p = os.path.join(state_mount, "metrics.json")
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8")).get("history", [])
        except Exception:
            pass
    return []


def main(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    os.makedirs(settings.output_dir, exist_ok=True)

    # 1) resume: pull previous compact weight bundle (state/) from durable dataset
    resume_model_dir = st.pull_resume(settings.state_mount, settings.output_dir)
    history = load_history(settings.state_mount)

    # 2) data
    print(f"[train] data_source={settings.data_source}")
    train_ds, eval_ds = load_datasets(settings)
    print(f"[train] train={len(train_ds)} eval={len(eval_ds)}")
    if settings.max_train_samples:
        train_ds = train_ds.select(range(min(settings.max_train_samples, len(train_ds))))
    if settings.max_eval_samples:
        eval_ds = eval_ds.select(range(min(settings.max_eval_samples, len(eval_ds))))

    # 3) vocab/processor from actual data (deterministic across sessions)
    vocab = build_vocab(train_ds["text"])
    processor = build_processor(vocab)
    vocab_size = len(vocab)
    print(f"[train] vocab={vocab_size} chars")
    pdir = os.path.join(settings.output_dir, "processor")
    os.makedirs(pdir, exist_ok=True)
    processor.save_pretrained(pdir)

    # 4) model (resume from durable weights if present, else base XLS-R)
    load_from = resume_model_dir or settings.model_name
    print(f"[train] loading {load_from} vocab_size={vocab_size}")
    load_kwargs = dict(vocab_size=vocab_size, pad_token_id=0, ctc_loss_reduction="mean")
    if resume_model_dir:
        # this transformers build *raises* on a shape mismatch; a vocab change
        # (e.g. corpus grew, or a prior smoke used a tiny subset) must only
        # re-init the CTC head and keep the pretrained encoder/transformer.
        load_kwargs["ignore_mismatched_sizes"] = True
    model = Wav2Vec2ForCTC.from_pretrained(load_from, **load_kwargs)
    if settings.freeze_feature_encoder:
        model.freeze_feature_encoder()
    model.config.use_cache = False

    # 5) args
    training_args = TrainingArguments(
        output_dir=settings.output_dir,
        per_device_train_batch_size=settings.batch_size,
        gradient_accumulation_steps=settings.grad_accum,
        per_device_eval_batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        warmup_ratio=settings.warmup_ratio,
        num_train_epochs=settings.num_epochs,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        save_strategy="steps", save_steps=settings.save_steps,
        save_total_limit=settings.keep_last_n_checkpoints + 1,
        eval_strategy="steps", eval_steps=settings.eval_steps,
        logging_steps=25, report_to=[],
        dataloader_num_workers=4, dataloader_persistent_workers=True,
        dataloader_pin_memory=True, dataloader_prefetch_factor=4,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=DataCollatorCTCWithPadding(processor, settings.sampling_rate),
        compute_metrics=compute_metrics(processor),
        callbacks=[TimeBudgetCallback(settings.time_budget_sec)],
    )

    # 6) train (weight-only resume: fresh optimizer each session)
    print(f"[train] resume_from={resume_model_dir or 'base model'}")
    trainer.train()

    # 7) evaluate
    metrics = trainer.evaluate()
    wer, cer, step = metrics.get("eval_wer"), metrics.get("eval_cer"), trainer.state.global_step
    print(f"[train] step={step} WER={wer:.4f} CER={cer:.4f}")
    history.append({"step": step, "wer": wer, "cer": cer, "ts": int(time.time())})

    # 8) save compact weights + push durable state
    trainer.save_model(os.path.join(settings.output_dir, st.STATE_MODEL_DIR))
    st.stage_for_push(settings.output_dir, settings.push_dir)
    best = min((h["wer"] for h in history if h.get("wer") is not None), default=None)
    st.write_metrics(settings.push_dir, {"history": history, "best_wer": best, "model": settings.model_name})
    st.push_state(settings.push_dir, settings.state_slug, f"xlsr step {step} WER {wer}")
    print("[train] session complete.")


if __name__ == "__main__":
    main()
