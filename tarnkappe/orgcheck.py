"""Second check on company detections.

The tagger flags many capitalised words as companies on web pages: German nouns ("Allgemeinmedizin"), product names
("USDC"), page text ("Email & Phone Number"), places. This check looks at each company candidate with features the
tagger does not weigh well, and gives the probability that it is a company:

- how the words behave elsewhere: how often each word is part of a company name in labelled text, how common it is
  (org_lexicon.json, built by scripts/build_org_lexicon.py);
- its form: length, casing, digits, a dot (domain names), "&";
- its surroundings: the word before it ("at", "bei", "von", "@"), whether it is a whole short line (menus, headings),
  whether the same text appears elsewhere on the page;
- the tagger's own confidence.

The classifier (orgcheck.onnx, gradient-boosted trees) is trained by scripts/train_orgcheck.py and run with onnxruntime.
"""
import json
import math
import os
import re

import numpy as np

HERE = os.path.join(os.path.dirname(__file__), "model")
WORD = re.compile(r"\w+", re.UNICODE)
CUE_BEFORE = {"at", "bei", "von", "vom", "for", "für", "with", "mit", "by", "from", "of", "@", "joins", "joined", "founder",
              "gründer", "ceo", "cto", "partner", "inhaber", "geschäftsführer", "director", "employer", "arbeitgeber",
              "kunde", "client", "customer", "acquired", "übernahm", "übernimmt", "verstärkt", "wechselt", "zu", "to"}
NOT_BEFORE = {"in", "im", "nach", "near", "»", "›", "|", "#", "·"}
LEGAL = {"gmbh", "ag", "ug", "kg", "se", "ltd", "inc", "llc", "plc", "corp", "corporation", "limited", "gbr", "ohg", "ev",
         "co", "sa", "bv", "nv", "srl", "spa", "ab", "oy", "as", "gesellschaft", "group", "holding", "verlag", "bank"}


# endings of ordinary nouns and job titles, not of names ("Allgemeinmedizin", "Wahlärztin", "Number")
NOUNISH = ("medizin", "ärztin", "arzt", "ärzte", "ung", "heit", "keit", "ismus", "nummer", "number", "zeiten", "ierung",
           "ologie", "ology", "tion", "ment", "ness", "ship", "ität", "ity", "ing", "ern", "chen", "isten", "erin", "eurin")
INNER = {"für", "lack", "and", "und", "the", "der", "die", "das", "is", "ist", "a"}


class OrgCheck:
    def __init__(self, model_dir=HERE):
        p = os.path.join(model_dir, "orgcheck.onnx")
        self.ready = os.path.exists(p)
        self.lex = json.load(open(os.path.join(model_dir, "org_lexicon.json"), encoding="utf-8"))
        if self.ready:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            self.sess = ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
            self.inp = self.sess.get_inputs()[0].name

    def features(self, text, e, entities=()):
        span = text[e["start"]:e["end"]]
        words = WORD.findall(span)
        lw = [w.lower() for w in words] or [""]
        stats = [self.lex.get(w) for w in lw]
        org_rate = [(s[1] + 0.5) / (s[0] + 2) if s else 0.25 for s in stats]
        # the rate for the word as it is written here: capitalised, or not
        cased = []
        for w, s in zip(words or [""], stats):
            if not s or len(s) < 5:
                cased.append(0.25)
            elif w[:1].isupper():
                cased.append((s[4] + 0.5) / (s[3] + 2))
            else:
                cased.append((s[1] - s[4] + 0.5) / (s[0] - s[3] + 2))
        other_rate = [(s[2] + 0.5) / (s[0] + 2) if s else 0.25 for s in stats]
        common = [math.log1p(s[0]) if s else 0.0 for s in stats]
        known = [1.0 if s else 0.0 for s in stats]
        letters = [c for c in span if c.isalpha()]
        upper = sum(c.isupper() for c in letters) / max(1, len(letters))
        line_start = text.rfind("\n", 0, e["start"]) + 1
        line_end = text.find("\n", e["end"])
        line_end = len(text) if line_end < 0 else line_end
        line = text[line_start:line_end].strip()
        before = WORD.findall(text[max(0, e["start"] - 30):e["start"]])
        prev = before[-1].lower() if before else ""
        prev_char = text[e["start"] - 1] if e["start"] > 0 else "\n"
        next_words = WORD.findall(text[e["end"]:e["end"] + 25])
        nxt = next_words[0].lower() if next_words else ""
        line_words = len(WORD.findall(line))
        sentence_start = e["start"] == line_start or bool(re.search(r"[.!?:]\s*$", text[max(0, e["start"] - 4):e["start"]]))
        repeats = text.count(span) - 1 if len(span) >= 3 else 0
        src = e.get("source", "model")
        return [
            float(e.get("score", 1.0)),
            1.0 if src == "rule" else 0.0,
            1.0 if src == "list" else 0.0,
            float(len(words)),
            math.log1p(len(span)),
            upper,
            1.0 if span.isupper() and len(letters) > 1 else 0.0,
            1.0 if span.islower() else 0.0,
            1.0 if words and all(w[:1].isupper() for w in words) else 0.0,
            1.0 if any(c.isdigit() for c in span) else 0.0,
            1.0 if "." in span.strip(".") else 0.0,
            1.0 if "&" in span else 0.0,
            1.0 if any(w in LEGAL for w in lw) else 0.0,
            max(org_rate), min(org_rate), sum(org_rate) / len(org_rate),
            max(other_rate),
            max(common), min(common),
            min(known),
            1.0 if prev in CUE_BEFORE or prev_char == "@" else 0.0,
            1.0 if prev in NOT_BEFORE else 0.0,
            1.0 if nxt in LEGAL else 0.0,
            1.0 if line == span.strip() else 0.0,
            math.log1p(line_words),
            1.0 if sentence_start else 0.0,
            math.log1p(repeats),
            1.0 if len(words) == 1 and lw[0] in self.lex and self.lex[lw[0]][1] == 0 else 0.0,
            min(cased), max(cased), sum(cased) / len(cased),
            1.0 if lw[-1].endswith(NOUNISH) else 0.0,
            1.0 if any(w in INNER for w in lw[1:-1]) else 0.0,
            1.0 if len(words) == 1 and words[0].isupper() and len(words[0]) <= 5 else 0.0,
        ]

    def probs(self, text, ents):
        """Probability that each entity is a company (1.0 when no classifier is installed)."""
        if not self.ready or not ents:
            return [1.0] * len(ents)
        x = np.array([self.features(text, e) for e in ents], dtype=np.float32)
        probabilities = self.sess.run(None, {self.inp: x})[1]          # [label, probabilities]
        return [float(p[1]) for p in probabilities]

    def prob(self, text, e, entities=()):
        return self.probs(text, [e])[0]
