"""WER / CER via edit distance, with correct CTC greedy decoding.

Greedy collapse (drop repeated tokens + blank) then map the word delimiter "|"
back to a space; otherwise Amharic words run together and WER is meaningless.
"""
from __future__ import annotations

_SPECIALS = {"<unk>", "<s>", "</s>", "<pad>"}


def _edit_distance(ref: list, hyp: list) -> int:
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cost = 0 if r == h else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def wer(refs: list[str], preds: list[str]) -> float:
    dist = total = 0
    for r, p in zip(refs, preds):
        rt, pt = r.split(), p.split()
        dist += _edit_distance(rt, pt)
        total += len(rt)
    return dist / total if total else 0.0


def cer(refs: list[str], preds: list[str]) -> float:
    dist = total = 0
    for r, p in zip(refs, preds):
        rt, pt = list(r.replace(" ", "")), list(p.replace(" ", ""))
        dist += _edit_distance(rt, pt)
        total += len(rt)
    return dist / total if total else 0.0


def compute_metrics(processor):
    tokenizer = processor.tokenizer
    pad_tok = tokenizer.pad_token

    def _tokens_to_text(tok_seq):
        out = []
        for t in tok_seq:
            if t in _SPECIALS or t is None:
                continue
            out.append(" " if t == "|" else t)
        return "".join(out).strip()

    def _decode_pred(ids):
        collapsed, prev = [], object()
        for i in ids:
            if i != prev:
                collapsed.append(i)
            prev = i
        toks = tokenizer.convert_ids_to_tokens(collapsed)
        return _tokens_to_text([t for t in toks if t != pad_tok])

    def _decode_label(ids):
        toks = tokenizer.convert_ids_to_tokens([i for i in ids if i >= 0])
        return _tokens_to_text([t for t in toks if t != pad_tok])

    def _metrics(eval_pred):
        import numpy as np
        logits, labels = eval_pred
        pred_ids = np.argmax(logits, axis=-1)
        preds = [_decode_pred(list(p)) for p in pred_ids]
        refs = [_decode_label([i for i in lab if i != -100]) for lab in labels]
        return {"wer": wer(refs, preds), "cer": cer(refs, preds)}

    return _metrics
