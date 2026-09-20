"""Load the merged Amharic HF dataset into a HuggingFace CTC dataset.

Works with any HF audio dataset exposing an `audio` feature + a text column
(train/validation/test). Amharic text is kept as fidel; non-fidel characters
(punctuation, latin, digits) are dropped so the CTC vocab stays tight.
"""
from __future__ import annotations

import glob
import json
import os
import tempfile
import unicodedata

from datasets import Dataset, Features, Value, Audio, load_dataset
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


def _read_kaldi_table(path: str) -> dict:
    table = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            utt, _, rest = line.partition(" ")
            table[utt.strip()] = rest.strip()
    return table


def _resolve_wav(scp_token: str, split_dir: str):
    base = os.path.basename(scp_token)
    if not base.endswith(".wav"):
        base += ".wav"
    for c in (os.path.join(split_dir, base), os.path.join(split_dir, "wav", base)):
        if os.path.exists(c):
            return c
    hits = glob.glob(os.path.join(split_dir, "**", base), recursive=True)
    return hits[0] if hits else None


def load_split_kaldi(split_dir: str) -> Dataset:
    """Kaldi wav.scp + text -> HF Dataset with an Audio(16k) feature."""
    wavs = _read_kaldi_table(os.path.join(split_dir, "wav.scp"))
    texts = _read_kaldi_table(os.path.join(split_dir, "text"))
    rows, missing = [], 0
    for utt, raw in texts.items():
        tok = wavs.get(utt)
        path = _resolve_wav(tok, split_dir) if tok else None
        if path is None:
            missing += 1
            continue
        text = normalize_amharic(raw)
        if not text:
            continue
        rows.append({"id": utt, "audio": {"path": path}, "text": text})
    if missing:
        print(f"[data] {split_dir}: dropped {missing} utts w/o resolvable audio")
    ds = Dataset.from_list(rows)
    feats = Features({"id": Value("string"), "audio": Audio(sampling_rate=16000), "text": Value("string")})
    return ds.cast(feats=feats)


def load_datasets(settings):
    """Dispatch to the configured source; returns (train, eval)."""
    if settings.data_source == "kaggle_kaldi":
        train = load_split_kaldi(settings.kaldi_train_dir)
        test = load_split_kaldi(settings.kaldi_test_dir) if os.path.exists(
            os.path.join(settings.kaldi_test_dir, "wav.scp")) else None
        return train, (test if test is not None and len(test) else train.shard(index=1, num_shards=10, contiguous=False))
    train, val, _test, _col = load_merged(settings.hf_dataset, settings.text_columns)
    return train, val


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
