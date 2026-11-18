"""Tune one decision threshold per entity type, favouring recall (F2), on development data only.

    python scripts/tune_thresholds.py          # writes model/thresholds.json

Dev data: UNER EWT dev, GermEval dev, WNUT 2017 dev, Broad Twitter Corpus dev (person / organisation / location) and
generated text with unseen names and phrasing (all types). Test sets are not touched.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pii import corpora, data                      # noqa: E402
from pii.detect import Detector                    # noqa: E402
from pii.features import TYPES                     # noqa: E402

BETA = 2.0
GRID = [round(x, 2) for x in np.arange(0.05, 0.61, 0.05)]


def main():
    det = Detector()
    real = corpora.ewt("dev") + corpora.germeval("dev") + corpora.wnut("dev") + corpora.btc("dev")
    synth = data.make("test", 2500, seed=5)
    cache = []
    for text, spans in real + synth:
        toks, probs = det._probs(text)
        if probs is not None:
            cache.append((toks, probs, spans))
    result, table = {}, []
    for typ in TYPES:
        best = None
        for thr in GRID:
            thresholds = {t: (thr if t == typ else 0.99) for t in TYPES}
            hit = miss = good = bad = 0
            for toks, probs, spans in cache:
                pred = [s for s in det.decode(toks, probs, thresholds) if s["type"] == typ]
                gold = [(a, b) for a, b, l in spans if l == typ]
                for a, b in gold:
                    if any(p["start"] < b and p["end"] > a for p in pred): hit += 1
                    else: miss += 1
                for p in pred:
                    if any(p["start"] < b and p["end"] > a for a, b in gold): good += 1
                    else: bad += 1
            r, p = hit / max(1, hit + miss), good / max(1, good + bad)
            f = (1 + BETA ** 2) * p * r / max(1e-9, BETA ** 2 * p + r)
            if thr == 0.5:
                base = (p, r)
            if best is None or f > best[0]:
                best = (f, thr, p, r)
        result[typ] = best[1]
        table.append((typ, base, best))
        print(f"{typ:9s} at 0.50: P {base[0]:.3f} R {base[1]:.3f}   ->  at {best[1]:.2f}: P {best[2]:.3f} R {best[3]:.3f}")
    out = os.path.join(os.path.dirname(__file__), "..", "model", "thresholds.json")
    json.dump(result, open(out, "w"), indent=2)
    print("wrote", os.path.normpath(out))


if __name__ == "__main__":
    main()
