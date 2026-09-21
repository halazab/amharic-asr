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

    _seen = {"n": 0}

    def _metrics(eval_pred):
        import numpy as np
        logits, labels = eval_pred
        pred_ids = np.argmax(logits, axis=-1)
        preds = [_decode_pred(list(p)) for p in pred_ids]
        refs = [_decode_label([i for i in lab if i != -100]) for lab in labels]
        if _seen["n"] == 0:
            _seen["n"] = 1
            import unicodedata
            for k in range(min(3, len(refs))):
                print(f"[decode] LOGIT_LEN={logits.shape[1]} pred_len={len(preds[k])} ref_len={len(refs[k])}")
                print(f"[decode] REF: {refs[k][:100]!r}")
                print(f"[decode] PRED:{preds[k][:100]!r}")
            _nc = [int((pred_ids == 0).sum(1).mean()), int((pred_ids != 0).sum(1).mean()),
                   logits.shape[1] * logits.shape[0]]
            print(f"[decode] blank-id0 frac/frame={_nc[0]}/{logits.shape[1]} nonzero_mean={_nc[1]}")
        return {"wer": wer(refs, preds), "cer": cer(refs, preds)}

    return _metrics
