"""Load the merged Amharic HF dataset into a HuggingFace CTC dataset.

Works with any HF audio dataset exposing an `audio` feature + a text column
(train/validation/test). Amharic text is kept as fidel; non-fidel characters
(punctuation, latin, digits) are dropped so the CTC vocab stays tight.
"""
from __future__ import annotations

import json
import os
import tempfile
import unicodedata

from datasets import load_dataset
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
W = "|"  # word delimiter; tokenizer converts " " -> "|" itself


def is_fidel(ch: str) -> bool:
    return 0x1200 <= ord(ch) <= 0x137F


def normalize_amharic(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").strip()
    out = []
    for ch in text:
        if ch.isspace():
            out.append(" ")
        elif is_fidel(ch):
            out.append(ch)
    return " ".join("".join(out).split())


def pick_text_column(columns, candidates) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None


def _pick_split(ds, names):
    for n in names:
        if n in ds:
            return ds[n]
    return None


def load_merged(hf_dataset: str, text_columns: tuple):
    ds = load_dataset(hf_dataset)
    train = _pick_split(ds, ["train"])
    val = _pick_split(ds, ["validation", "val", "dev"])
    test = _pick_split(ds, ["test"])
    cols = train.column_names
    text_col = pick_text_column(cols, text_columns)
    if text_col is None:
        raise ValueError(f"no text column among {text_columns}; got {cols}")

    def prep(split):
        def _map(ex):
            ex["text"] = normalize_amharic(ex[text_col])
            return ex
        split = split.map(_map, remove_columns=[c for c in split.column_names if c not in ("audio", "text")])
        # drop empty transcripts
        split = split.filter(lambda x: len(x["text"]) > 0, num_proc=1)
        return split

    train = prep(train)
    val = prep(val) if val is not None else train.shard(index=1, num_shards=10, contiguous=False)
    test = prep(test) if test is not None else None
    return train, val, test, text_col


def build_vocab(train_texts: list[str]) -> dict[str, int]:
    chars = set()
    for t in train_texts:
        chars.update(c for c in t if c != " ")
    vocab = {PAD_TOKEN: 0, BOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3, W: 4}
    for i, c in enumerate(sorted(chars), start=5):
        vocab[c] = i
    return vocab


def build_processor(vocab: dict[str, int]) -> Wav2Vec2Processor:
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False)
        tokenizer = Wav2Vec2CTCTokenizer(
            path,
            unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, bos_token=BOS_TOKEN, eos_token=EOS_TOKEN,
            word_delimiter_token=W, replace_word_delimiter_char=" ", do_lower_case=False,
            keep_special_tokens=False,
        )
    finally:
        os.remove(path)
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True
    )
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


class DataCollatorCTCWithPadding:
    """Resample to 16k if needed, extract features, pad waveforms + labels."""

    def __init__(self, processor: Wav2Vec2Processor, target_sr: int = 16000):
        self.processor = processor
        self.target_sr = target_sr

    def _to_16k(self, array, sr):
        if sr == self.target_sr:
            return array
        import librosa
        return librosa.resample(array.astype("float32"), orig_sr=sr, target_sr=self.target_sr)

    def __call__(self, features: list[dict]) -> dict:
        import torch

        waves = [self._to_16k(f["audio"]["array"], f["audio"].get("sampling_rate", self.target_sr)) for f in features]
        input_values = self.processor(
            waves, sampling_rate=self.target_sr, return_tensors="pt", padding=True
        ).input_values

        label_ids = [self.processor.tokenizer(f["text"]).input_ids for f in features]
        max_len = max(len(l) for l in label_ids)
        pad = self.processor.tokenizer.pad_token_id
        labels = torch.full((len(label_ids), max_len), pad, dtype=torch.long)
        for i, lab in enumerate(label_ids):
            labels[i, : len(lab)] = torch.tensor(lab)
        return {"input_values": input_values, "labels": labels}
