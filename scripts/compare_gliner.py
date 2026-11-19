"""Tarnkappe next to GLiNER (v1 PII) and GLiNER2-PII on the same texts.

    pip install gliner gliner2
    python scripts/compare_gliner.py --predict     # slow: runs the GLiNER models, caches predictions
    python scripts/compare_gliner.py               # scores from the cache

GLiNER models take a few hundred milliseconds per text on a CPU, so the large test sets are
subsampled (fixed seed). Predictions are cached with their scores so thresholds can be compared
without running the models again.
"""
import collections
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval import challenge              # noqa: E402
from tarnkappe import corpora                 # noqa: E402

CACHE = os.path.join(os.path.dirname(__file__), "..", "eval", "gliner_predictions.json")
LABELS = ["person", "organization", "location", "street address", "postal code", "phone number", "email", "iban"]
NER = {"person": "PERSON", "organization": "ORG", "location": "LOCATION"}
# Presidio labels every system was asked to find (its other labels, such as dates and job titles, are ignored)
PRESIDIO_TYPES = {"PERSON", "ORGANIZATION", "STREET_ADDRESS", "GPE", "PHONE_NUMBER", "EMAIL_ADDRESS", "ZIP_CODE", "IBAN_CODE"}
N = 400


def datasets():
    rng = random.Random(13)
    pick = lambda ex: rng.sample(ex, min(N, len(ex)))
    ch = [(x["text"], x["spans"]) for x in challenge.load()]
    return {
        "ewt_test": ("ner", pick(corpora.ewt("test"))),
        "wnut17_test": ("ner", pick(corpora.wnut("test"))),
        "germeval_test": ("ner", pick(corpora.germeval("test"))),
        "pud_de": ("ner", pick(corpora.pud_de())),
        "challenge_v1": ("pii", ch),
        "presidio_unseen": ("pii", pick(corpora.presidio())),
    }


def systems():
    def ours():
        from tarnkappe.detect import Detector
        det = Detector()
        back = {"ORGANIZATION": "organization", "LOCATION": "location", "STREET": "street address", "ZIP": "postal code",
                "PHONE": "phone number", "EMAIL": "email", "IBAN": "iban"}
        return lambda t: [[e["start"], e["end"], back.get(e["type"], "person" if e["type"] == "PERSON" else e["type"]),
                           1.0] for e in det.detect(t, split_names=False)]

    def gliner_v1():
        from gliner import GLiNER
        m = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1"); m.eval()
        return lambda t: [[e["start"], e["end"], e["label"], float(e["score"])]
                          for e in m.predict_entities(t, LABELS, threshold=0.3)]

    def gliner2_pii():
        from gliner2 import GLiNER2
        m = GLiNER2.from_pretrained("fastino/gliner2-privacy-filter-PII-multi")

        def run(t):
            res = m.extract_entities(t, LABELS, threshold=0.3, include_confidence=True, include_spans=True)
            return [[v["start"], v["end"], lab, float(v["confidence"])]
                    for lab, vals in res.get("entities", {}).items() for v in vals]
        return run
    return {"tarnkappe": ours, "gliner_multi_pii-v1": gliner_v1, "gliner2-pii": gliner2_pii}


def predict():
    data = datasets()
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    for name, make in systems().items():
        if name in cache:
            continue
        run = make()
        run("warm up")
        cache[name] = {}
        for dname, (_, ex) in data.items():
            t = time.perf_counter()
            preds = [run(text) for text, _ in ex]
            cache[name][dname] = {"ms": (time.perf_counter() - t) * 1000 / len(ex), "preds": preds}
            print(name, dname, f"{cache[name][dname]['ms']:.1f} ms/text", flush=True)
        json.dump(cache, open(CACHE, "w"))


def prf(tp, fp, fn):
    p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return p, r, 2 * p * r / max(1e-9, p + r)


def score(threshold):
    data = datasets()
    cache = json.load(open(CACHE))
    out = {}
    for name in cache:
        out[name] = {}
        for dname, (kind, ex) in data.items():
            c = cache[name][dname]
            tp = fp = fn = hit = miss = pgood = pbad = 0
            found, total = collections.Counter(), collections.Counter()
            bad = n_ent = 0
            for (text, gold), preds in zip(ex, c["preds"]):
                preds = [p for p in preds if p[3] >= threshold]
                if kind == "ner":
                    got = {(a, b, NER[l]) for a, b, l, _ in preds if l in NER}
                    gs = set(gold)
                    tp += len(got & gs); fp += len(got - gs); fn += len(gs - got)
                    for a, b, l in gold:
                        if l == "PERSON":
                            if any(p[0] < b and p[1] > a for p in preds): hit += 1
                            else: miss += 1
                    for p in preds:                 # is a "person" detection on a real person?
                        if p[2] == "person":
                            if any(p[0] < b and p[1] > a for a, b, l in gold if l == "PERSON"): pgood += 1
                            else: pbad += 1
                else:
                    for a, b, l in gold:
                        if dname.startswith("presidio") and l not in PRESIDIO_TYPES:
                            continue
                        total[l] += 1
                        found[l] += any(p[0] < b and p[1] > a for p in preds)
                    n_ent += len(preds)
                    bad += sum(not any(p[0] < b and p[1] > a for a, b, _ in gold) for p in preds)
            if kind == "ner":
                out[name][dname] = {"strict_f1": round(prf(tp, fp, fn)[2], 3), "strict_p": round(prf(tp, fp, fn)[0], 3),
                                    "strict_r": round(prf(tp, fp, fn)[1], 3),
                                    "persons_found": round(hit / max(1, hit + miss), 3),
                                    "person_precision": round(pgood / max(1, pgood + pbad), 3), "ms": round(c["ms"], 1)}
            else:
                out[name][dname] = {"found": round(sum(found.values()) / sum(total.values()), 3),
                                    "precision": round(1 - bad / max(1, n_ent), 3), "ms": round(c["ms"], 1),
                                    "by_type": {l: round(found[l] / total[l], 3) for l in total}}
    return out


if __name__ == "__main__":
    if "--predict" in sys.argv:
        predict()
    for thr in (0.5, 0.3):
        res = score(thr)
        print(f"\n=== GLiNER threshold {thr} (Tarnkappe always uses its own tuned thresholds) ===")
        for dname in next(iter(res.values())):
            print(dname)
            for name, r in res.items():
                d = r[dname]
                if "strict_f1" in d:
                    print(f"   {name:22s} strict F1 {d['strict_f1']:.3f} (P {d['strict_p']:.3f} R {d['strict_r']:.3f})"
                          f"   persons found {d['persons_found']:.1%}, person detections correct {d['person_precision']:.1%}"
                          f"   {d['ms']:.1f} ms/text")
                else:
                    print(f"   {name:22s} PII found {d['found']:.1%}   detections real {d['precision']:.1%}   {d['ms']:.1f} ms/text")
    json.dump({str(t): score(t) for t in (0.5, 0.3)}, open(CACHE.replace("predictions", "scores"), "w"), indent=1)
