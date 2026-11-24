"""Word statistics for the company check (tarnkappe/orgcheck.py).

    python scripts/build_org_lexicon.py      # writes tarnkappe/model/org_lexicon.json

For every word (lower-cased) it counts how often it appears inside a company name, inside another entity, and in
total, over the human-labelled training corpora (UNER EWT, GermEval, WNUT, BTC) and the teacher-labelled Wikipedia
paragraphs (confident labels only), separately for capitalised occurrences: "university" is rarely a company,
"University" often is. A word that is common but almost never part of a company ("Allgemeinmedizin",
"university", "Email") is a poor company on its own. Only training data is used; test sets are not touched.
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tarnkappe import corpora                      # noqa: E402
from tarnkappe.features import tokenize            # noqa: E402

KEEP = {"ORG": 0.8, "PERSON": 0.7, "LOCATION": 0.6}


def add(stats, text, spans):
    toks = tokenize(text)
    for t, s, e in toks:
        if t[0] == "\n" or not any(c.isalpha() for c in t):
            continue
        w = t.lower()
        lab = next((l for a, b, l in spans if s >= a and e <= b), None)
        st = stats[w]
        st[0] += 1
        if lab == "ORG":
            st[1] += 1
        elif lab:
            st[2] += 1
        if t[0].isupper():                  # the same counts for capitalised occurrences only
            st[3] += 1
            if lab == "ORG":
                st[4] += 1


if __name__ == "__main__":
    stats = collections.defaultdict(lambda: [0, 0, 0, 0, 0])   # total, in ORG, in other entity, capitalised, capitalised in ORG
    human = corpora.ewt("train") + corpora.germeval("train") + corpora.wnut("train") + corpora.btc("train")
    for text, spans in human:
        add(stats, text, spans)
    path = os.path.join(os.path.dirname(__file__), "..", "data", "wiki_gliner2.jsonl")
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        spans = [(a, b, l) for a, b, l, sc in r["spans"] if sc >= KEEP.get(l, 1.1)]
        add(stats, r["text"], spans)
    out = {w: v for w, v in stats.items() if v[0] >= 3}
    dst = os.path.join(os.path.dirname(__file__), "..", "tarnkappe", "model", "org_lexicon.json")
    json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    print(f"{len(out)} words -> {dst} ({os.path.getsize(dst) // 1024} KB)")
    for w in ["allgemeinmedizin", "university", "email", "siemens", "gmbh", "twitter", "usdc", "agentur", "crypto", "herald", "ärzte"]:
        print(w, stats.get(w))
