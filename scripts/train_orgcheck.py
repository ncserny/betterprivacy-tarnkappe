"""Train the company check (tarnkappe/orgcheck.py) on company candidates from text the tagger was not trained on.

    python scripts/train_orgcheck.py data/wiki2_gliner2.jsonl data/company_pages.jsonl [model_dir]
    # writes <model_dir>/orgcheck.onnx (default tarnkappe/model)

Candidates are the tagger's company detections at a low threshold. A candidate is positive when it overlaps a labelled
company. Sources: the development splits of UNER EWT, GermEval, WNUT and BTC (human labels) and fresh Wikipedia
paragraphs labelled by GLiNER2-PII (confident labels only; candidates on unsure teacher labels are skipped), half of
them wrapped in menu and footer lines like build_distilled.py does, and 2,000 page-like examples built from a list of
real company names (scripts/company_examples.py). Hand-labelled real pages are not used: they are kept for testing.
"""
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from tarnkappe import corpora                    # noqa: E402
from tarnkappe.detect import Detector            # noqa: E402
from tarnkappe.orgcheck import OrgCheck          # noqa: E402
from build_distilled import wrap                 # noqa: E402

CANDIDATE_THRESHOLD = 0.1


def candidates(det, chk, text, gold, unsure=()):
    out = []
    for e in det.detect(text, split_names=False, min_score=None):
        if e["type"] != "ORGANIZATION":
            continue
        a, b = e["start"], e["end"]
        if any(x < b and y > a for x, y in unsure) and not any(x < b and y > a for x, y, l in gold if l == "ORG"):
            continue
        y = 1 if any(x < b and yy > a and l == "ORG" for x, yy, l in gold) else 0
        out.append((chk.features(text, e), y))
    return out


def main():
    rng = random.Random(3)
    model_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(__file__), "..", "tarnkappe", "model")
    det = Detector(model_dir)
    det.thresholds["ORG"] = CANDIDATE_THRESHOLD
    chk = OrgCheck(model_dir)
    X, Y = [], []
    human = corpora.ewt("dev") + corpora.germeval("dev") + corpora.wnut("dev") + corpora.btc("dev")
    for text, spans in human:
        for x, y in candidates(det, chk, text, spans):
            X.append(x); Y.append(y)
    n_h = len(X)
    for line in open(sys.argv[1], encoding="utf-8"):
        r = json.loads(line)
        text = r["text"]
        gold = [(a, b, l) for a, b, l, s in r["spans"] if l == "ORG" and s >= 0.8]
        unsure = [(a, b) for a, b, l, s in r["spans"] if l == "ORG" and s < 0.8]
        if rng.random() < 0.5:
            spans = [(a, b, "ORG") for a, b, _ in gold] + [(a, b, "?") for a, b in unsure]
            text, spans = wrap(rng, text, spans)
            gold = [(a, b, l) for a, b, l in spans if l == "ORG"]
            unsure = [(a, b) for a, b, l in spans if l == "?"]
        for x, y in candidates(det, chk, text, gold, unsure):
            X.append(x); Y.append(y)
    n_w = len(X)
    comp = [json.loads(l) for l in open(sys.argv[2], encoding="utf-8")]
    for r in rng.sample(comp, min(2000, len(comp))):
        for x, y in candidates(det, chk, r["text"], [tuple(s) for s in r["spans"]]):
            X.append(x); Y.append(y)
    X, Y = np.array(X, dtype=np.float32), np.array(Y)
    print(f"{len(Y)} candidates ({n_h} human dev, {n_w - n_h} Wikipedia, {len(Y) - n_w} company pages), "
          f"{Y.mean():.1%} real companies")
    from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    make = lambda: GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.08, subsample=0.8,
                                              random_state=0)
    print("5-fold AUC", cross_val_score(make(), X, Y, cv=5, scoring="roc_auc").mean().round(3))
    clf = make().fit(X, Y)
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    onx = convert_sklearn(clf, initial_types=[("x", FloatTensorType([None, X.shape[1]]))],
                          options={id(clf): {"zipmap": False}}, target_opset={"": 17, "ai.onnx.ml": 3})
    dst = os.path.join(model_dir, "orgcheck.onnx")
    open(dst, "wb").write(onx.SerializeToString())
    print("wrote", os.path.normpath(dst), os.path.getsize(dst) // 1024, "KB")


if __name__ == "__main__":
    main()
