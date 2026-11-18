"""Label unlabelled text with a larger teacher model, for training the small tagger on it.

    python scripts/distil_label.py snippets.txt data/distilled.jsonl --teacher gliner2
    python train.py --extra data/distilled.jsonl

Input: one text snippet per line. Output: JSONL {"text": ..., "spans": [[start, end, LABEL], ...]}.
Teachers:
    spacy:<model>   any installed spaCy pipeline
    gliner          urchade/gliner_multi_pii-v1
    gliner2         fastino/gliner2-privacy-filter-PII-multi

Caution: both GLiNER models over-flag persons in free text (pronouns, "owner", "children"; see
RESULTS.md). Keep a high --threshold for person labels or use them for address-type labels only.
About 200 ms per snippet on a 2-core CPU; use a GPU for more than a few thousand snippets.
"""
import argparse
import json

GL = {"person": "PERSON", "organization": "ORG", "location": "LOCATION", "street address": "STREET",
      "postal code": "ZIP", "email": "EMAIL", "phone number": "PHONE"}
GL2 = {"full_name": "PERSON", "first_name": "PERSON", "last_name": "PERSON", "organization": "ORG",
       "city": "LOCATION", "country": "LOCATION", "street_address": "STREET", "postal_code": "ZIP",
       "email": "EMAIL", "phone_number": "PHONE"}
SP = {"PERSON": "PERSON", "PER": "PERSON", "ORG": "ORG", "GPE": "LOCATION", "LOC": "LOCATION"}


def teacher(name, threshold):
    if name.startswith("spacy:"):
        import spacy
        nlp = spacy.load(name[6:], disable=["lemmatizer"])
        return lambda t: [[e.start_char, e.end_char, SP[e.label_]] for e in nlp(t).ents if e.label_ in SP]
    if name == "gliner":
        from gliner import GLiNER
        m = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")
        return lambda t: [[e["start"], e["end"], GL[e["label"]]]
                          for e in m.predict_entities(t, list(GL), threshold=threshold)]
    if name == "gliner2":
        from gliner2 import GLiNER2
        m = GLiNER2.from_pretrained("fastino/gliner2-privacy-filter-PII-multi")

        def run(t):
            res = m.extract_entities(t, list(GL2), threshold=threshold, include_spans=True)
            return [[v["start"], v["end"], GL2[lab]] for lab, vals in res.get("entities", {}).items() for v in vals]
        return run
    raise SystemExit(f"unknown teacher {name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("infile"); ap.add_argument("outfile")
    ap.add_argument("--teacher", default="gliner2")
    ap.add_argument("--threshold", type=float, default=0.7)
    a = ap.parse_args()
    label = teacher(a.teacher, a.threshold)
    n = 0
    with open(a.outfile, "w", encoding="utf-8") as out:
        for line in open(a.infile, encoding="utf-8"):
            text = line.strip()
            if not text:
                continue
            spans = sorted(label(text))
            keep, end = [], -1                     # drop overlapping spans, first one wins
            for s in spans:
                if s[0] >= end:
                    keep.append(s); end = s[1]
            out.write(json.dumps({"text": text, "spans": keep}, ensure_ascii=False) + "\n")
            n += 1
    print(f"labelled {n} snippets with {a.teacher}")
