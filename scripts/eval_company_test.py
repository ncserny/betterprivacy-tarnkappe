"""Score a model on held-out company examples (scripts/company_examples.py --split test)."""
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tarnkappe.detect import Detector
det = Detector(*(sys.argv[2:3]))
prof = os.environ.get("PROFILE") or None
n = f = typed = pn = pf = k = ok = 0
bare_n = bare_f = 0
for line in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(line); t = r["text"]
    ents = det.detect(t, split_names=False, profile=prof)
    for a, b, l in r["spans"]:
        if l == "ORG":
            n += 1
            hit = [e for e in ents if e["start"] < b and e["end"] > a]
            f += bool(hit); typed += any(e["type"] == "ORGANIZATION" for e in hit)
            if not any(w in t[a:b] for w in ("GmbH", "UG", "e. V.", "KG", "AG", "mbH", "eG")):
                bare_n += 1; bare_f += any(e["type"] == "ORGANIZATION" for e in hit)
        if l == "PERSON":
            pn += 1; pf += any(e["start"] < b and e["end"] > a and e["type"] == "PERSON" for e in ents)
    for e in ents:
        if e["type"] == "ORGANIZATION":
            k += 1; ok += any(e["start"] < b and e["end"] > a and l == "ORG" for a, b, l in r["spans"])
print(f"companies labelled as company {typed}/{n} = {typed/n:.1%} (bare names {bare_f}/{bare_n} = {bare_f/max(1,bare_n):.1%}), "
      f"company detections correct {ok}/{k} = {ok/max(1,k):.1%}, persons as person {pf}/{pn} = {pf/pn:.1%}")
