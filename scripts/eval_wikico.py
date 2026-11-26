"""Recall of the target company in the Wikipedia company excerpts (company-content-1000; not in this repository).

    python scripts/eval_wikico.py data/wikico_all.json validation [model_dir]

Only the article's own company is labelled in that file, so only recall is measured here.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tarnkappe import Detector  # noqa: E402

path, split = sys.argv[1], sys.argv[2]
det = Detector(*sys.argv[3:4])
rows = [r for r in json.load(open(path)) if r["split"] == split]
n = hit = typed = 0
miss = []
for r in rows:
    ents = det.detect(r["text"], profile="pages")
    for a, b, _ in r["spans"]:
        n += 1
        ov = [e for e in ents if e["start"] < b and e["end"] > a]
        hit += bool(ov)
        ok = any(e["type"] == "ORGANIZATION" for e in ov)
        typed += ok
        if not ok:
            miss.append(r["text"][a:b])
print(f"{split}: target companies {n}, found as organisation {typed / n:.1%} (any type {hit / n:.1%})")
print("missed:", sorted(set(miss))[:40])
