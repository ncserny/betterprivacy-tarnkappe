"""Train the PII tagger on human-annotated corpora + synthetic PII text,
evaluate on held-out real test sets, export to ONNX (float32 + int8).

    bash scripts/fetch_data.sh && python scripts/export_vectors.py
    python train.py
"""
import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from tarnkappe import corpora, data
from tarnkappe.features import (CHAR_VOCAB, MAX_CHARS, TAGS, TYPES, VEC_DIM, WORD_BUCKETS, Vectors,
                          featurize, spans_to_tags, tags_to_spans, tokenize)


class Tagger(nn.Module):
    """Pretrained word vectors + hashed word embedding + char-CNN -> 2-layer BiGRU -> tags."""

    def __init__(self, word_dim=48, char_dim=32, char_out=96, vec_out=160, hidden=160, n_tags=len(TAGS)):
        super().__init__()
        self.word = nn.Embedding(WORD_BUCKETS, word_dim, padding_idx=0)
        self.char = nn.Embedding(CHAR_VOCAB, char_dim, padding_idx=0)
        self.conv = nn.Conv1d(char_dim, char_out, kernel_size=3, padding=1)
        self.vec = nn.Linear(VEC_DIM, vec_out)
        self.drop = nn.Dropout(0.3)
        self.gru = nn.GRU(word_dim + char_out + vec_out, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.25)
        self.out = nn.Linear(2 * hidden, n_tags)

    def forward(self, words, chars, vecs, lengths=None):
        B, T, C = chars.shape
        c = self.char(chars.reshape(B * T, C)).transpose(1, 2)
        c = torch.relu(self.conv(c)).max(dim=2).values.reshape(B, T, -1)
        x = self.drop(torch.cat([self.word(words), c, torch.relu(self.vec(vecs))], dim=-1))
        if lengths is not None:
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            h, _ = self.gru(packed)
            h, _ = pad_packed_sequence(h, batch_first=True, total_length=T)
        else:
            h, _ = self.gru(x)
        return self.out(self.drop(h))


def encode(examples, vectors, max_tokens=400):
    enc = []
    for text, spans in examples:
        toks = tokenize(text)[:max_tokens]
        if not toks:
            continue
        strs = [t for t, _, _ in toks]
        w, c = featurize(strs)
        enc.append((w, c, vectors.ids(strs), spans_to_tags(toks, spans), spans, toks))
    return enc


def batches(enc, bs, shuffle, tables, word_dropout=0.0):
    idx = list(range(len(enc)))
    if shuffle:
        random.shuffle(idx)
        # bucket by length so batches are not mostly padding
        idx = sum((sorted(idx[i:i + bs * 50], key=lambda j: len(enc[j][0])) for i in range(0, len(idx), bs * 50)), [])
    order = list(range(0, len(idx), bs))
    if shuffle:
        random.shuffle(order)
    for i in order:
        chunk = [enc[j] for j in idx[i:i + bs]]
        T = max(len(e[0]) for e in chunk)
        W = np.zeros((len(chunk), T), dtype=np.int64)
        C = np.zeros((len(chunk), T, MAX_CHARS), dtype=np.int64)
        V = np.zeros((len(chunk), T, 2), dtype=np.int64)
        Y = np.full((len(chunk), T), -100, dtype=np.int64)
        L = np.array([len(e[0]) for e in chunk])
        for k, e in enumerate(chunk):
            n = len(e[0])
            W[k, :n], C[k, :n], V[k, :n], Y[k, :n] = e[0], e[1], e[2], e[3]
        if word_dropout:
            # hide word identity (mostly on entity tokens) so the model must use spelling
            # and context - that is what generalises to names it has never seen
            p = np.where(Y > 0, word_dropout, word_dropout / 5)
            W[np.random.rand(*W.shape) < p] = 0
            V[np.random.rand(*W.shape) < p / 2] = 0
        V = torch.from_numpy(V)
        vec = torch.cat([tables[0][V[..., 0]], tables[1][V[..., 1]]], dim=-1)
        yield torch.from_numpy(W), torch.from_numpy(C), vec, torch.from_numpy(Y), torch.from_numpy(L), chunk


def evaluate(model, enc, tables, types=TYPES):
    """Strict entity-level precision / recall / F1 (exact span + type match)."""
    model.eval()
    tp = {t: 0 for t in types}; fp = dict(tp); fn = dict(tp)
    with torch.no_grad():
        for W, C, V, _, L, chunk in batches(enc, 256, False, tables):
            pred = model(W, C, V, L).argmax(-1).numpy()
            for k, e in enumerate(chunk):
                got = {(s["start"], s["end"], s["type"]) for s in tags_to_spans(e[5], pred[k][:len(e[5])])}
                gold = set(e[4])
                for g in got:
                    if g[2] in tp:
                        (tp if g in gold else fp)[g[2]] += 1
                for g in gold - got:
                    if g[2] in fn:
                        fn[g[2]] += 1
    def prf(a, b, c):
        p, r = a / max(1, a + b), a / max(1, a + c)
        return {"precision": round(p, 4), "recall": round(r, 4),
                "f1": round(2 * p * r / max(1e-9, p + r), 4), "support": a + c}
    res = {t: prf(tp[t], fp[t], fn[t]) for t in types if tp[t] + fn[t]}
    res["micro"] = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    return res


def casing_noise(examples, rng):
    """Informal typing: 12% of texts all lower-case, 2% all upper-case, and in a further 12%
    each entity is lower-cased with probability one half ("met laura becker at Siemens")."""
    out = []
    for t, s in examples:
        k = rng.random()
        if k < 0.12:
            t = t.lower()
        elif k < 0.14:
            t = t.upper()
        elif k < 0.26:
            for a, b, _ in s:
                if rng.random() < 0.5 and len(t[a:b].lower()) == b - a:
                    t = t[:a] + t[a:b].lower() + t[b:]
        out.append((t, s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", type=int, default=25000)
    ap.add_argument("--presidio", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--extra", nargs="*", default=[], help="JSONL files with extra labelled text (see scripts/distil_label.py)")
    ap.add_argument("--out", default="tarnkappe/model")
    a = ap.parse_args()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    rng = random.Random(0)

    t0 = time.time()
    vectors = Vectors(os.path.join(a.out, "vectors.npz"))
    tables = [torch.from_numpy(t) for t in vectors.tables]
    informal = corpora.wnut("train") + corpora.btc("train")
    real = corpora.ewt("train") + corpora.germeval("train") + informal
    real = casing_noise(corpora.join_neighbours(real, rng), rng)
    if a.extra:      # teacher-labelled text is formal; show the informal text twice so it keeps its weight
        real += casing_noise(corpora.join_neighbours(informal, rng), rng)
    for path in a.extra:
        extra = casing_noise(corpora.jsonl(path), rng)
        print(f"extra: {len(extra)} examples from {path}")
        real += extra
    train = encode(real + data.make("train", a.synthetic) + data.make_presidio(a.presidio), vectors)
    NER = ["PERSON", "ORG", "LOCATION"]
    tests = {
        "ewt_test (English web text)": (encode(corpora.ewt("test"), vectors), NER),
        "germeval_test (German Wikipedia/news)": (encode(corpora.germeval("test"), vectors), NER),
        "pud_de (German news/Wikipedia)": (encode(corpora.pud_de(), vectors), NER),
        "wnut17_test (social media)": (encode(corpora.wnut("test"), vectors), NER),
        "btc_test (tweets)": (encode(corpora.btc("test"), vectors), NER),
        "synthetic_new_phrasing (PII templates)": (encode(data.make("test", 3000), vectors), TYPES),
    }
    dev = encode(corpora.ewt("dev") + corpora.germeval("dev") + corpora.wnut("dev") + corpora.btc("dev")[:1000]
                 + data.make("test", 1500, seed=5), vectors)
    print(f"data: {len(real)} real + {a.synthetic + a.presidio} synthetic train examples, {len(dev)} dev, {time.time() - t0:.0f}s")

    model = Tagger()
    print("parameters:", sum(p.numel() for p in model.parameters()))
    steps = a.epochs * (len(train) // 64 + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps, pct_start=0.15)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    best, best_state = -1, None
    for ep in range(a.epochs):
        model.train(); tot = n = 0
        for W, C, V, Y, L, _ in batches(train, 64, True, tables, word_dropout=0.5):
            loss = loss_fn(model(W, C, V, L).reshape(-1, len(TAGS)), Y.reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item(); n += 1
        f1 = evaluate(model, dev, tables)["micro"]["f1"]
        if f1 > best:
            best, best_state = f1, copy.deepcopy(model.state_dict())
        print(f"epoch {ep + 1}: loss {tot / n:.4f}  dev micro F1 {f1:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)

    res = {name: evaluate(model, enc, tables, types) for name, (enc, types) in tests.items()}
    for name, r in res.items():
        print(f"{name:42s} F1 {r['micro']['f1']:.3f}   " +
              "  ".join(f"{t} {r[t]['f1']:.2f}" for t in r if t != "micro"))
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/metrics.json", "w"), indent=2)
    torch.save(model.state_dict(), f"{a.out}/tagger.pt")

    # first-name list used by the runtime to split PERSON into first / last name
    from faker import Faker
    names = set()
    for loc in ["en_US", "en_GB", "de_DE", "de_AT"] + data.NAME_LOCALES:
        f = Faker(loc); f.seed_instance(1)
        names |= {f.first_name().lower() for _ in range(4000)}
    open(f"{a.out}/first_names.txt", "w", encoding="utf-8").write("\n".join(sorted(names)))

    # ---- ONNX export (batch of 1, variable length) + dynamic int8 quantisation ----
    model.eval()
    W, C, V = (torch.zeros(1, 7, dtype=torch.long), torch.zeros(1, 7, MAX_CHARS, dtype=torch.long),
               torch.zeros(1, 7, VEC_DIM))
    torch.onnx.export(model, (W, C, V), f"{a.out}/tagger.onnx", input_names=["words", "chars", "vecs"],
                      output_names=["logits"], opset_version=17, dynamo=False,
                      dynamic_axes={"words": {1: "T"}, "chars": {1: "T"}, "vecs": {1: "T"}, "logits": {1: "T"}})
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(f"{a.out}/tagger.onnx", f"{a.out}/tagger.int8.onnx", weight_type=QuantType.QInt8)


if __name__ == "__main__":
    main()
