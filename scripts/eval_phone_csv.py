import sys
import csv, sys, collections
"""Phone recall on a CSV of labelled phone examples (columns text, span_start, span_end, split, format_style).
Usage: python scripts/eval_phone_csv.py . examples.csv test"""
sys.path.insert(0, sys.argv[1])
from tarnkappe import Detector
det = Detector(sys.argv[1] + "/tarnkappe/model")
rows = [r for r in csv.DictReader(open(sys.argv[2], encoding="utf-8-sig")) if r["split"] == sys.argv[3]]
c = collections.Counter(); bystyle = collections.defaultdict(collections.Counter); miss = []
for r in rows:
    t = r["text"]; a, b = int(r["span_start"]), int(r["span_end"])
    ents = det.detect(t, profile="pages")
    ph = [e for e in ents if e["type"] == "PHONE"]
    hit = any(e["start"] < b and e["end"] > a for e in ph)
    full = any(e["start"] <= a and e["end"] >= b - (len(r["raw_phone"]) - len(r["raw_phone"].split(" Durchwahl")[0].split(" ext")[0].split(" x")[0])) for e in ph)
    c["n"] += 1; c["hit"] += hit; c["full"] += full
    c["fp"] += sum(1 for e in ph if not (e["start"] < b and e["end"] > a))
    bystyle[r["format_style"]]["n"] += 1; bystyle[r["format_style"]]["hit"] += hit
    if not hit: miss.append(t)
print(f"found {c['hit']}/{c['n']} = {c['hit']/c['n']:.1%}, whole number {c['full']/c['n']:.1%}, extra phone hits {c['fp']}")
print({k: f"{v['hit']/v['n']:.0%}" for k, v in sorted(bystyle.items(), key=lambda x: x[1]['hit']/x[1]['n'])})
for m in miss[:12]: print("  MISS", m)
