"""Loader for hand-annotated test files in the bracket markup used by eval/challenge_v1.txt.

    [[P|person]] [[O|organisation]] [[L|location]] [[S|street]] [[Z|postcode]] [[E|email]] [[T|phone]] [[I|IBAN]]

The same format is produced by scripts/prelabel.py, so real pages can be pre-labelled by the
model, corrected by hand in a text editor and loaded here.
"""
import os
import re

CODES = {"P": "PERSON", "O": "ORGANIZATION", "L": "LOCATION", "S": "STREET", "Z": "ZIP",
         "E": "EMAIL", "T": "PHONE", "I": "IBAN", "C": "CREDIT_CARD", "A": "IP_ADDRESS"}
TO_CODE = {v: k for k, v in CODES.items()}
TO_CODE.update({"FIRST_NAME": "P", "LAST_NAME": "P"})
MARK = re.compile(r"\[\[([A-Z])\|(.+?)\]\]", re.S)
DEFAULT = os.path.join(os.path.dirname(__file__), "challenge_v1.txt")


def parse(marked):
    """Marked-up text -> (plain text, [(start, end, TYPE)])."""
    out, spans, pos = [], [], 0
    n = 0
    for m in MARK.finditer(marked):
        out.append(marked[pos:m.start()]); n += m.start() - pos
        spans.append((n, n + len(m.group(2)), CODES[m.group(1)]))
        out.append(m.group(2)); n += len(m.group(2))
        pos = m.end()
    out.append(marked[pos:])
    return "".join(out), spans


def load(path=DEFAULT):
    """Returns [{'category', 'lang', 'text', 'spans'}]."""
    records = []
    for block in open(path, encoding="utf-8").read().split("\n---\n"):
        lines = [l for l in block.strip("\n").split("\n") if not l.startswith("#")]
        if not lines or not lines[0].startswith("@"):
            continue
        cat, lang = [x.strip() for x in lines[0][1:].split("|")]
        text, spans = parse("\n".join(lines[1:]))
        records.append({"category": cat, "lang": lang, "text": text, "spans": spans})
    return records


def mark(text, entities):
    """Inverse of parse(): detector output -> marked-up text (adjacent first/last names are merged)."""
    ents, out, pos = [], [], 0
    for e in sorted(entities, key=lambda e: e["start"]):
        code = TO_CODE.get(e["type"])
        if not code:
            continue
        if ents and code == "P" and ents[-1][2] == "P" and text[ents[-1][1]:e["start"]] in (" ", ", "):
            ents[-1] = (ents[-1][0], e["end"], "P")
        else:
            ents.append((e["start"], e["end"], code))
    for a, b, code in ents:
        out += [text[pos:a], f"[[{code}|{text[a:b]}]]"]
        pos = b
    return "".join(out) + text[pos:]
