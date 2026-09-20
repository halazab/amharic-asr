"""Fine-tune XLS-R (Wav2Vec2) + CTC on the merged Amharic dataset, resumable.

Run: python -m xlsr_finetune.train
Launch once per Kaggle session. Each launch resumes from the latest checkpoint
(pulled from the durable state dataset), trains until the time budget, saves,
records WER/CER, then pushes state back to Kaggle.
"""
from __future__ import annotations

import json
import os
import time

import torch
from transformers import Trainer, TrainingArguments, Wav2Vec2ForCTC, TrainerCallback

from . import state as st
from .config import Settings, get_settings
from .data import DataCollatorCTCWithPadding, build_processor, build_vocab, load_merged
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

    # 1) resume first (Trainer discovers checkpoints in output_dir)
    start_step = st.pull_resume(settings.state_mount, settings.output_dir)
    history = load_history(settings.state_mount)

    # 2) data
    print(f"[train] loading HF dataset: {settings.hf_dataset}")
    train_ds, eval_ds, _test, text_col = load_merged(settings.hf_dataset, settings.text_columns)
    print(f"[train] text_col={text_col} train={len(train_ds)} eval={len(eval_ds)}")
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

    # 4) model
    print(f"[train] loading {settings.model_name} vocab_size={vocab_size}")
    model = Wav2Vec2ForCTC.from_pretrained(
        settings.model_name, vocab_size=vocab_size, pad_token_id=0,
        ctc_loss_reduction="sum", ignore_index=0,
    )
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
        logging_steps=25, report_to=[], dataloader_num_workers=2,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=DataCollatorCTCWithPadding(processor, settings.sampling_rate),
        compute_metrics=compute_metrics(processor),
        callbacks=[TimeBudgetCallback(settings.time_budget_sec)],
    )

    # 6) train (resume if a checkpoint exists)
    resume = start_step > 0 and st._latest_checkpoint(settings.output_dir) is not None
    print(f"[train] resume={resume} start_step={start_step}")
    trainer.train(resume_from_checkpoint=resume)

    # 7) evaluate
    metrics = trainer.evaluate()
    wer, cer, step = metrics.get("eval_wer"), metrics.get("eval_cer"), trainer.state.global_step
    print(f"[train] step={step} WER={wer:.4f} CER={cer:.4f}")
    history.append({"step": step, "wer": wer, "cer": cer, "ts": int(time.time())})

    # 8) save final + push state
    trainer.save_model(os.path.join(settings.output_dir, "latest"))
    st.stage_for_push(settings.output_dir, settings.push_dir)
    best = min((h["wer"] for h in history if h.get("wer") is not None), default=None)
    st.write_metrics(settings.push_dir, {"history": history, "best_wer": best, "model": settings.model_name})
    st.push_state(settings.push_dir, settings.state_slug, f"xlsr step {step} WER {wer}")
    print("[train] session complete.")


if __name__ == "__main__":
    main()
