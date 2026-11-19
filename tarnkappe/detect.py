"""Runtime: regex recognisers for structured PII + ONNX tagger for the rest.

Only needs numpy and onnxruntime.
"""
import json
import os
import re

import numpy as np
import onnxruntime as ort

from .features import TAG2ID, TYPES, Vectors, featurize, tags_to_spans, tokenize

try:                                    # Google's libphonenumber, optional but strongly recommended
    import phonenumbers
except ImportError:                     # pragma: no cover
    phonenumbers = None

# Countries whose national phone formats are recognised (international "+.." numbers always are).
PHONE_REGIONS = ("DE", "AT", "CH", "GB", "US")
# Longest stretch of text given to the model at once (training texts were at most 400 tokens).
MAX_TOKENS = 320

HERE = os.path.join(os.path.dirname(__file__), "model")

# Structured identifiers: a pattern (plus checksum where one exists) beats any model.
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b")
CARD_RE = re.compile(r"(?<!\d )(?<![\w.])\d(?:[ -]?\d){12,18}\b")
IPV4_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
EXT = r"(?:\s?(?:x|ext\.?)\s?\d{1,5})?"
PHONE_RE = re.compile(r"(?<!\d )(?<![\w.])(?:\+|00)\d{1,3}[\s\-/.]?(?:\(?\d{1,5}\)?[\s\-/.]?){2,5}\d{2,4}" + EXT + r"(?!\w)")
# national formats that are unambiguous: 555-123-4567, 555.123.4567x89, (030) 1234 5678
PHONE2_RE = re.compile(r"(?<!\d )(?<![\w.\-])(?:\(\d{2,5}\)\s?\d{3,4}|\d{3}[-.]\d{3})[-.\s]\d{4}" + EXT + r"(?![\w\-])")

# ---- postcodes -------------------------------------------------------------------------
# Formats that are distinctive enough to accept on sight.
ZIP_SURE = re.compile(
    r"\b(?:[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}"           # UK      NW1 6XE
    r"|[A-Z]\d[A-Z]\s?\d[A-Z]\d"                        # Canada  K1A 0B1
    r"|(?:D|DE|A|AT|CH|F|B|NL|PL|DK|L)-\d{4,5}"           # prefixed D-10115
    r"|\d{5}-\d{4})\b")                                 # US ZIP+4
# Formats that are just numbers: accepted only next to an address part or a keyword.
ZIP_MAYBE = re.compile(
    r"\b(?:[A-Z]\d{2}\s?[A-Z0-9]{4}"                    # Ireland T12 X8PN
    r"|\d{4}\s?[A-Z]{2}"                                # Netherlands 1012 AB
    r"|\d{2}-\d{3}"                                     # Poland 53-320
    r"|\d{3}\s\d{2}"                                    # Sweden, Czechia 114 55
    r"|\d{5}|\d{4})\b")
ZIP_KEYWORD = re.compile(r"(?:PLZ|Postleitzahl|ZIP(?: code)?|Zip(?: code)?|zip(?: code)?|Postcode|postcode|"
                         r"Postal code|postal code)\s*(?:[:#]|is|ist|lautet|=)?\s*$")
US_STATE = re.compile(r"\b(?:A[LKZR]|C[AOT]|D[EC]|FL|GA|HI|I[ADLN]|K[SY]|LA|M[ADEINOST]|N[CDEHJMVY]|"
                      r"O[HKR]|PA|RI|S[CD]|T[NX]|UT|V[AT]|W[AIVY]),?\s$")
GAP = re.compile(r"^[\s,;|·\-\"']{0,4}$")              # what may sit between a postcode and its neighbour

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


def regex_spans(text, regions=PHONE_REGIONS):
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
    if phonenumbers is not None and sum(c.isdigit() for c in text) >= 7:
        seen = {(a, b) for t, a, b in out if t == "PHONE"}
        for region in regions:
            for m in phonenumbers.PhoneNumberMatcher(text, region):
                if (m.start, m.end) not in seen:
                    seen.add((m.start, m.end))
                    out.append(("PHONE", m.start, m.end))
    return [{"type": t, "start": a, "end": b, "score": 1.0, "source": "regex"} for t, a, b in out]


def zip_spans(text, entities):
    """Postcodes by country pattern. Plain numbers count only next to a detected street or
    location, after a US state abbreviation, or after a keyword such as 'PLZ' or 'ZIP'."""
    free = lambda a, b: all(b <= e["start"] or a >= e["end"] for e in entities)
    places = [e for e in entities if e["type"] in ("LOCATION", "STREET")]
    out = []
    for m in ZIP_SURE.finditer(text):
        if free(m.start(), m.end()):
            out.append((m.start(), m.end()))
    for m in ZIP_MAYBE.finditer(text):
        a, b = m.start(), m.end()
        if not free(a, b) or any(a < y and b > x for x, y in out):
            continue
        before, after = text[max(0, a - 24):a], text[b:b + 6]
        if after[:1] in ("-", "/", ".", ":") and after[1:2].isdigit() or before[-1:] in ("/", "-", ":", ".", "#", "€", "$"):
            continue                                  # part of a date, time, amount or id
        near_after = any(e["start"] >= b and GAP.match(text[b:e["start"]]) for e in places if e["type"] == "LOCATION")
        near_before = any(e["end"] <= a and GAP.match(text[e["end"]:a]) for e in places)
        four = len(m.group()) == 4 and m.group().isdigit()
        ok = bool(ZIP_KEYWORD.search(before)) or near_after or (near_before and not four) or \
            (bool(US_STATE.search(before)) and len(m.group()) == 5)
        if ok:
            out.append((a, b))
    return [{"type": "ZIP", "start": a, "end": b, "score": 1.0, "source": "regex"} for a, b in out]


class Detector:
    def __init__(self, model_dir=HERE, threads=1, model_file="tagger.int8.onnx", phone_regions=PHONE_REGIONS):
        self.phone_regions = phone_regions
        # per-type decision thresholds, tuned for recall by scripts/tune_thresholds.py
        self.thresholds = {t: 0.5 for t in TYPES}
        tp = os.path.join(model_dir, "thresholds.json")
        if os.path.exists(tp):
            self.thresholds.update(json.load(open(tp)))
        # named threshold sets for particular kinds of text, e.g. "pages" for scanned web pages
        pp = os.path.join(model_dir, "profiles.json")
        self.profiles = json.load(open(pp)) if os.path.exists(pp) else {}
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(os.path.join(model_dir, model_file), so,
                                         providers=["CPUExecutionProvider"])
        self.vectors = Vectors(os.path.join(model_dir, "vectors.npz"))
        p = os.path.join(model_dir, "first_names.txt")
        self.first_names = set(open(p, encoding="utf-8").read().split("\n")) if os.path.exists(p) else set()

    # ---- model ----
    def _probs(self, text):
        """Tag probabilities for every token. Long texts are processed in windows of at most
        MAX_TOKENS tokens, cut at line or sentence ends, so that the model always sees inputs
        of the lengths it was trained on and memory use stays flat."""
        toks = tokenize(text)
        if not toks:
            return toks, None
        parts = []
        for a, b in self._windows(text, toks):
            strs = [t for t, _, _ in toks[a:b]]
            w, c = featurize(strs)
            v = self.vectors.dense(self.vectors.ids(strs))
            logits = self.sess.run(None, {"words": w[None], "chars": c[None], "vecs": v[None]})[0][0]
            e = np.exp(logits - logits.max(-1, keepdims=True))
            parts.append(e / e.sum(-1, keepdims=True))
        return toks, parts[0] if len(parts) == 1 else np.concatenate(parts)

    @staticmethod
    def _windows(text, toks, max_tokens=None):
        """Token index ranges [(start, end), ...] covering the text. A window ends at the last
        line break in its second half, else at the last sentence end there, else at max_tokens."""
        max_tokens = max_tokens or MAX_TOKENS
        out, start, n = [], 0, len(toks)
        while n - start > max_tokens:
            end, sentence = None, None
            for i in range(start + max_tokens, start + max_tokens // 2, -1):
                gap = text[toks[i - 1][2]:toks[i][1]]
                if toks[i - 1][0][0] == "\n":
                    end = i
                    break
                if sentence is None and toks[i - 1][0] in ".!?" and gap:
                    sentence = i
            end = end or sentence or start + max_tokens
            out.append((start, end))
            start = end
        out.append((start, n))
        return out

    @staticmethod
    def decode(toks, probs, thresholds):
        """A token is an entity when the probability that it is *some* entity reaches the
        threshold of its most likely type. Lower thresholds find more and flag more."""
        b, i = probs[:, 1::2], probs[:, 2::2]                # [T, n_types] begin / inside
        tp = b + i
        best = tp.argmax(-1)
        thr = np.array([thresholds[t] for t in TYPES])[best]
        is_ent = (1.0 - probs[:, 0]) >= thr
        is_ent &= np.array([t[0][0] != "\n" for t in toks])       # a line break always ends an entity
        rows = np.arange(len(best))
        begin = b[rows, best] >= i[rows, best]
        tags = np.where(is_ent, 1 + 2 * best + np.where(begin, 0, 1), 0)
        spans = tags_to_spans(toks, tags, tp[rows, best])
        for s in spans:
            s["source"] = "model"
        return spans

    def _model_spans(self, text, thresholds=None):
        toks, probs = self._probs(text)
        return [] if probs is None else self.decode(toks, probs, thresholds or self.thresholds)

    # ---- PERSON -> FIRST_NAME / LAST_NAME ----
    def _split_person(self, text, s):
        parts = [(m.group(), s["start"] + m.start(), s["start"] + m.end())
                 for m in re.finditer(r"[^\s,]+", text[s["start"]:s["end"]])]
        had_title = False
        while parts and parts[0][0].rstrip(".").lower() in TITLES:
            parts.pop(0); had_title = True
        before = re.search(r"(\w+)\.?\s*$", text[max(0, s["start"] - 24):s["start"]])   # a title just before the name
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
    def detect(self, text, min_score=None, split_names=True, known_companies=(), profile=None):
        """min_score: one threshold for all model types (0..1); default: the tuned per-type values.
        profile: a named threshold set from profiles.json. The default thresholds favour finding
        everything, also in tweets and chat; "pages" flags fewer menu words and product names on
        web pages at the cost of names in informal text.
        known_companies: names to flag as ORGANIZATION wherever they appear (see tarnkappe/scan.py)."""
        thr = None if min_score is None else {t: min_score for t in TYPES}
        if profile and min_score is None:
            if profile not in self.profiles:
                raise ValueError(f"unknown profile {profile!r}; available: {sorted(self.profiles)}")
            thr = {**self.thresholds, **self.profiles[profile]}
        from .scan import LEGAL, legal_form_spans
        rx = regex_spans(text, self.phone_regions)
        rx.sort(key=lambda s: s["start"] - s["end"])       # longer regex hit wins
        model = [s for s in self._model_spans(text, thr)
                 if re.search(r"[^\W\d_]{2}", text[s["start"]:s["end"]])                    # not just "&", "B" or an emoji
                 and not (s["type"] == "ORG" and re.fullmatch(LEGAL, text[s["start"]:s["end"]]))]   # not "GmbH" alone
        # "<Name> GmbH" is a company by rule, unless the tagger already has a longer company span around it
        rules = [r for r in legal_form_spans(text)
                 if not any(m["type"] == "ORG" and m["start"] <= r["start"] and m["end"] >= r["end"] for m in model)]
        kept = []
        for s in rx + rules + model:
            if all(s["end"] <= k["start"] or s["start"] >= k["end"] for k in kept):
                kept.append(s)
        kept += zip_spans(text, kept)
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
        if known_companies:
            from .scan import apply_known
            out = apply_known(text, out, known_companies)
        return out

    def redact(self, text, entities=None):
        entities = self.detect(text) if entities is None else entities
        out, pos = [], 0
        for e in entities:
            out += [text[pos:e["start"]], f"[{e['type']}]"]
            pos = e["end"]
        return "".join(out) + text[pos:]
