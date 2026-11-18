"""Run GLiNER (v1 PII) and GLiNER2-PII on the same test sets as bench.py.

NOT RUN in the prototype sandbox (no Hugging Face access). Needs:
    pip install gliner "gliner2[local]"
    python scripts/compare_gliner.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench import strict            # noqa: E402
from pii import corpora             # noqa: E402

LABELS = {"person": "PERSON", "organization": "ORG", "location": "LOCATION"}


def gliner_v1():
    from gliner import GLiNER
    m = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")
    return lambda text: {(e["start"], e["end"], LABELS[e["label"]])
                         for e in m.predict_entities(text, list(LABELS), threshold=0.5)}


def gliner2_pii():
    from gliner2 import GLiNER2
    m = GLiNER2.from_pretrained("fastino/gliner2-privacy-filter-PII-multi")

    def predict(text):
        res = m.extract_entities(text, list(LABELS), threshold=0.5, include_spans=True)
        return {(v["start"], v["end"], LABELS[label])
                for label, values in res.get("entities", {}).items() for v in values}
    return predict


if __name__ == "__main__":
    sets = [("ewt_test", corpora.ewt("test")), ("wnut17_test", corpora.wnut("test")),
            ("germeval_test", corpora.germeval("test")), ("pud_de", corpora.pud_de())]
    for name, make in [("gliner_multi_pii-v1", gliner_v1), ("gliner2-pii", gliner2_pii)]:
        predict = make()
        for sname, ex in sets:
            t = time.perf_counter()
            (p, r, f), (_, _, pf) = strict(predict, ex)
            print(f"{name:20s} {sname:14s} all F1 {f:.3f} (P {p:.3f} R {r:.3f})  person F1 {pf:.3f}  "
                  f"{(time.perf_counter() - t) * 1000 / len(ex):.1f} ms/sentence")
