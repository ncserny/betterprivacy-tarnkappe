"""Runtime: regex recognisers for structured PII + ONNX tagger for the rest.

Only needs numpy and onnxruntime.
"""
import os
import re

import numpy as np
import onnxruntime as ort

from .features import Vectors, featurize, tags_to_spans, tokenize

HERE = os.path.join(os.path.dirname(__file__), "..", "model")

# Structured identifiers: a pattern (plus checksum where one exists) beats any model.
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b")
CARD_RE = re.compile(r"(?<!\d )(?<![\w.])\d(?:[ -]?\d){12,18}\b")
IPV4_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
EXT = r"(?:\s?(?:x|ext\.?)\s?\d{1,5})?"
PHONE_RE = re.compile(r"(?<!\d )(?<![\w.])(?:\+|00)\d{1,3}[\s\-/.]?(?:\(?\d{1,5}\)?[\s\-/.]?){2,5}\d{2,4}" + EXT + r"(?!\w)")
# national formats that are unambiguous: 555-123-4567, 555.123.4567x89, (030) 1234 5678
PHONE2_RE = re.compile(r"(?<!\d )(?<![\w.\-])(?:\(\d{2,5}\)\s?\d{3,4}|\d{3}[-.]\d{3})[-.\s]\d{4}" + EXT + r"(?![\w\-])")

TITLES = {"mr", "mrs", "ms", "miss", "dr", "prof", "herr", "herrn", "frau", "sir", "madam", "mx"}
PARTICLES = {"von", "van", "de", "der", "den", "del", "della", "di", "da", "le", "la", "zu", "bin", "al", "el", "ten"}
# model label -> public label
PUBLIC = {"ORG": "ORGANIZATION"}


def _luhn(s):
    d = [int(c) for c in s if c.isdigit()][::-1]
    return len(d) >= 13 and sum(d[0::2] + [sum(divmod(2 * x, 10)) for x in d[1::2]]) % 10 == 0


def _iban_ok(s):
    s = s.replace(" ", "")
    if not 15 <= len(s) <= 34:
        return False
    n = "".join(str(int(c, 36)) for c in s[4:] + s[:4])
    return int(n) % 97 == 1


def regex_spans(text):
    out = []
    for m in EMAIL_RE.finditer(text):
        out.append(("EMAIL", m.start(), m.end()))
    for m in IBAN_RE.finditer(text):
        if _iban_ok(m.group()):
            out.append(("IBAN", m.start(), m.end()))
    for m in CARD_RE.finditer(text):
        if _luhn(m.group()):
            out.append(("CREDIT_CARD", m.start(), m.end()))
    for m in IPV4_RE.finditer(text):
        out.append(("IP_ADDRESS", m.start(), m.end()))
    for rx in (PHONE_RE, PHONE2_RE):       # clear formats only; the model covers the rest
        for m in rx.finditer(text):
            if sum(c.isdigit() for c in m.group()) >= 8:
                out.append(("PHONE", m.start(), m.end()))
    return [{"type": t, "start": a, "end": b, "score": 1.0, "source": "regex"} for t, a, b in out]


class Detector:
    def __init__(self, model_dir=HERE, threads=1, model_file="tagger.int8.onnx"):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(os.path.join(model_dir, model_file), so,
                                         providers=["CPUExecutionProvider"])
        self.vectors = Vectors(os.path.join(model_dir, "vectors.npz"))
        p = os.path.join(model_dir, "first_names.txt")
        self.first_names = set(open(p, encoding="utf-8").read().split("\n")) if os.path.exists(p) else set()

    # ---- model ----
    def _model_spans(self, text):
        toks = tokenize(text)
        if not toks:
            return []
        strs = [t for t, _, _ in toks]
        w, c = featurize(strs)
        v = self.vectors.dense(self.vectors.ids(strs))
        logits = self.sess.run(None, {"words": w[None], "chars": c[None], "vecs": v[None]})[0][0]
        e = np.exp(logits - logits.max(-1, keepdims=True))
        probs = e / e.sum(-1, keepdims=True)
        spans = tags_to_spans(toks, probs.argmax(-1), probs.max(-1))
        for s in spans:
            s["source"] = "model"
        return spans

    # ---- PERSON -> FIRST_NAME / LAST_NAME ----
    def _split_person(self, text, s):
        parts = [(m.group(), s["start"] + m.start(), s["start"] + m.end())
                 for m in re.finditer(r"[^\s,]+", text[s["start"]:s["end"]])]
        had_title = False
        while parts and parts[0][0].rstrip(".").lower() in TITLES:
            parts.pop(0); had_title = True
        before = re.search(r"(\w+)\.?\s*$", text[:s["start"]])
        had_title = had_title or bool(before and before.group(1).lower() in TITLES)
        if not parts:
            return []
        mk = lambda typ, a, b: {"type": typ, "start": a, "end": b, "score": s["score"], "source": "model"}
        comma = "," in text[parts[0][2]:parts[-1][1]] if len(parts) > 1 else False
        if len(parts) == 1:
            w = parts[0][0].lower()
            typ = "LAST_NAME" if had_title or w not in self.first_names else "FIRST_NAME"
            return [mk(typ, parts[0][1], parts[0][2])]
        if comma:                                   # "Last, First"
            return [mk("LAST_NAME", parts[0][1], parts[0][2]), mk("FIRST_NAME", parts[1][1], parts[-1][2])]
        cut = len(parts) - 1                        # last name = final word, or from a particle ("van der ...")
        for i in range(1, len(parts) - 1):
            if parts[i][0].lower() in PARTICLES:
                cut = i
                break
        return [mk("FIRST_NAME", parts[0][1], parts[cut - 1][2]), mk("LAST_NAME", parts[cut][1], parts[-1][2])]

    # ---- public API ----
    def detect(self, text, min_score=0.5, split_names=True):
        rx = regex_spans(text)
        rx.sort(key=lambda s: s["start"] - s["end"])       # longer regex hit wins
        kept = []
        for s in rx + [m for m in self._model_spans(text) if m["score"] >= min_score]:
            if all(s["end"] <= k["start"] or s["start"] >= k["end"] for k in kept):
                kept.append(s)
        out = []
        for s in kept:
            if s["type"] == "PERSON" and split_names:
                out += self._split_person(text, s)
            else:
                s["type"] = PUBLIC.get(s["type"], s["type"])
                out.append(s)
        out.sort(key=lambda s: s["start"])
        for s in out:
            s["text"] = text[s["start"]:s["end"]]
        return out

    def redact(self, text, entities=None):
        entities = self.detect(text) if entities is None else entities
        out, pos = [], 0
        for e in entities:
            out += [text[pos:e["start"]], f"[{e['type']}]"]
            pos = e["end"]
        return "".join(out) + text[pos:]
