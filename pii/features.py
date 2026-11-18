"""Tokenisation and feature extraction shared by training and inference.

Pure Python + numpy so the runtime has no dependency on PyTorch or spaCy.
"""
import os
import re
import zlib

import numpy as np

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
WORD_BUCKETS = 30000      # hashed word vocabulary (id 0 = padding / unknown)
CHAR_VOCAB = 320          # 0 pad, 1..255 latin-1, 256..319 hashed other chars
MAX_CHARS = 20
VEC_DIM = 600             # 300-d English + 300-d German pretrained vectors

TYPES = ["PERSON", "ORG", "LOCATION", "STREET", "ZIP", "EMAIL", "PHONE"]
TAGS = ["O"] + [f"{p}-{t}" for t in TYPES for p in ("B", "I")]
TAG2ID = {t: i for i, t in enumerate(TAGS)}

DEFAULT_VECTORS = os.path.join(os.path.dirname(__file__), "..", "model", "vectors.npz")


def tokenize(text):
    """Returns list of (token, start, end)."""
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def _char_id(c):
    o = ord(c)
    return o if 0 < o < 256 else 256 + o % 64


class Vectors:
    """Frozen pretrained word vectors (see scripts/export_vectors.py)."""

    def __init__(self, path=DEFAULT_VECTORS):
        z = np.load(path)
        self.tables, self.index = [], []
        for lang in ("en", "de"):
            words = z[f"{lang}_words"].tobytes().decode("utf-8").split("\n")
            self.index.append(dict(zip(words, z[f"{lang}_rows"].tolist())))
            self.tables.append(z[f"{lang}_table"].astype(np.float32))

    def ids(self, tokens):
        """[T, 2] row ids (0 = unknown): exact match, then lower-case, then Title-case."""
        out = np.zeros((len(tokens), 2), dtype=np.int64)
        for i, t in enumerate(tokens):
            for k, idx in enumerate(self.index):
                out[i, k] = idx.get(t) or idx.get(t.lower()) or idx.get(t.capitalize()) or 0
        return out

    def dense(self, ids):
        """[..., 2] ids -> [..., 600] float32."""
        return np.concatenate([self.tables[0][ids[..., 0]], self.tables[1][ids[..., 1]]], axis=-1)


def featurize(tokens):
    """tokens: list of str -> (word_ids [T], char_ids [T, MAX_CHARS]) as int64."""
    n = len(tokens)
    words = np.zeros(n, dtype=np.int64)
    chars = np.zeros((n, MAX_CHARS), dtype=np.int64)
    for i, tok in enumerate(tokens):
        words[i] = zlib.crc32(tok.lower().encode("utf-8")) % (WORD_BUCKETS - 1) + 1
        for j, c in enumerate(tok[:MAX_CHARS]):
            chars[i, j] = _char_id(c)
    return words, chars


def spans_to_tags(tokens, spans):
    """Character spans -> BIO tag ids aligned to tokens."""
    tags = np.zeros(len(tokens), dtype=np.int64)
    for a, b, lab in spans:
        first = True
        for i, (_, s, e) in enumerate(tokens):
            if s >= a and e <= b:
                tags[i] = TAG2ID[("B-" if first else "I-") + lab]
                first = False
    return tags


def tags_to_spans(tokens, tag_ids, probs=None):
    """BIO tag ids -> list of dicts with char offsets (tolerant of I- without B-)."""
    out, cur = [], None
    for i, (_, s, e) in enumerate(tokens):
        tag = TAGS[tag_ids[i]]
        typ = tag[2:] if tag != "O" else None
        if typ and cur and cur["type"] == typ and not tag.startswith("B-"):
            cur["end"] = e
            cur["_p"].append(probs[i] if probs is not None else 1.0)
        else:
            if cur:
                out.append(cur)
            cur = {"type": typ, "start": s, "end": e,
                   "_p": [probs[i] if probs is not None else 1.0]} if typ else None
    if cur:
        out.append(cur)
    for sp in out:
        p = sp.pop("_p")
        sp["score"] = round(float(sum(p) / len(p)), 4)
    return out
