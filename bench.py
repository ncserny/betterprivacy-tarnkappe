"""End-to-end check of the shipped pipeline (int8 ONNX + regex): samples, accuracy on
real test sets next to spaCy baselines, PII recall, latency.

    python bench.py            # add --spacy to also run the spaCy baselines
"""
import statistics
import sys
import time

from pii import corpora, data
from pii.detect import Detector
from pii.features import tokenize

SAMPLES = [
    "Hi, I'm Laura Becker from Nordwind GmbH. You can reach me at laura.becker@nordwind.de or +49 30 1234567.",
    "Please send the invoice to Thomas Ferreira, 221 Baker Street, London NW1 6XE.",
    "Sehr geehrter Herr Yilmaz, bitte überweisen Sie den Betrag auf DE89 3704 0044 0532 0130 00.",
    "The Windows update will be released on Monday. Restart the server afterwards.",
    "hey its mike, call me on 0171 2345678",
    "Meeting with Priya Raman (Acme Corp) moved to Friday, her card 4111 1111 1111 1111 was declined.",
    "Impressum: Bäckerei Schmitt GmbH, Inhaber Jürgen Schmitt, Hauptstraße 12, 50667 Köln, Tel. 0221 998877",
    "Posted by Aylin Demir on Tuesday: does anyone know a good dentist in Leipzig?",
]
NER = {"PERSON", "ORG", "LOCATION"}
SPACY = {"PERSON": "PERSON", "PER": "PERSON", "ORG": "ORG", "GPE": "LOCATION", "LOC": "LOCATION"}


def prf(tp, fp, fn):
    p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return p, r, 2 * p * r / max(1e-9, p + r)


def strict(predict, examples, types=NER):
    """Exact span + exact type. predict(text) -> set of (start, end, type)."""
    tp = fp = fn = 0
    per = [0, 0, 0]
    for text, spans in examples:
        got = {g for g in predict(text) if g[2] in types}
        gold = {g for g in spans if g[2] in types}
        tp += len(got & gold); fp += len(got - gold); fn += len(gold - got)
        gp, pp = {g for g in gold if g[2] == "PERSON"}, {g for g in got if g[2] == "PERSON"}
        per[0] += len(gp & pp); per[1] += len(pp - gp); per[2] += len(gp - pp)
    return prf(tp, fp, fn), prf(*per)


def caught(det, examples, gold_types=None):
    """Was each piece of PII found at all (any overlap, any type)? The number that matters for redaction."""
    hit = miss = bad = 0
    for text, spans in examples:
        ents = det.detect(text)
        for a, b, lab in spans:
            if gold_types and lab not in gold_types:
                continue
            if any(e["start"] < b and e["end"] > a for e in ents): hit += 1
            else: miss += 1
        bad += sum(not any(e["start"] < b and e["end"] > a for a, b, _ in spans) for e in ents)
    return prf(hit, bad, miss)


def main():
    det = Detector(threads=1)
    ours = lambda text: {(e["start"], e["end"], e["type"]) for e in det.detect(text, split_names=False)
                         } | set()
    ours_ner = lambda text: {(a, b, {"ORGANIZATION": "ORG"}.get(t, t)) for a, b, t in ours(text)}

    print("== samples ==")
    for s in SAMPLES:
        print(s)
        for e in det.detect(s):
            print(f"   {e['type']:<12} {e['text']!r}  {e['score']:.2f} ({e['source']})")
        print("   ->", det.redact(s))

    sets = [("ewt_test (English web text)", corpora.ewt("test"), "en"),
            ("wnut17_test (social media)", corpora.wnut("test"), "en"),
            ("germeval_test (German Wikipedia/news)", corpora.germeval("test"), "de"),
            ("pud_de (German news/Wikipedia)", corpora.pud_de(), "de")]
    systems = [("ours", lambda lang: ours_ner)]
    if "--spacy" in sys.argv:
        import spacy
        nlp = {}
        def sp(name):
            def get(lang):
                m = name.replace("LANG", {"en": "en_core_web", "de": "de_core_news"}[lang])
                if m not in nlp:
                    nlp[m] = spacy.load(m, disable=["lemmatizer"])
                return lambda text: {(e.start_char, e.end_char, SPACY[e.label_])
                                     for e in nlp[m](text).ents if e.label_ in SPACY}
            return get
        systems += [("spaCy sm", sp("LANG_sm")), ("spaCy md", sp("LANG_md"))]

    print("\n== real annotated text: strict F1 on person / organisation / location, and on person alone ==")
    for name, ex, lang in sets:
        print(name)
        for sname, get in systems:
            t = time.perf_counter()
            (p, r, f), (pp, pr, pf) = strict(get(lang), ex)
            ms = (time.perf_counter() - t) * 1000 / len(ex)
            print(f"   {sname:9s} all F1 {f:.3f} (P {p:.3f} R {r:.3f})   person F1 {pf:.3f} (P {pp:.3f} R {pr:.3f})"
                  f"   {ms:.2f} ms/sentence")

    print("\n== PII caught at all (any overlap, any label) ==")
    pres = corpora.presidio()
    core = {"PERSON", "ORGANIZATION", "STREET_ADDRESS", "GPE", "PHONE_NUMBER", "EMAIL_ADDRESS",
            "ZIP_CODE", "CREDIT_CARD", "IBAN_CODE", "IP_ADDRESS"}
    for name, ex, types in [("presidio (templates unseen in training)", pres, core),
                            ("our synthetic, new names + new phrasing", data.make("test", 2000, seed=7), None),
                            ("ewt_test persons only", corpora.ewt("test"), {"PERSON"}),
                            ("germeval_test persons only", corpora.germeval("test"), {"PERSON"})]:
        p, r, f = caught(det, ex, types)
        print(f"   {name:45s} recall {r:.3f}" + (f"  precision {p:.3f}" if types is None or types is core else ""))
    for t in sorted(core):
        _, r, _ = caught(det, pres, {t})
        print(f"      presidio {t:15s} recall {r:.3f}")

    # first / last name split, scored on synthetic text where the truth is known
    fine = data.make("test", 2000, seed=7, fine=True)
    tp = fp = fn = 0
    for text, spans in fine:
        got = {(e["start"], e["end"], e["type"]) for e in det.detect(text) if e["type"].endswith("_NAME")}
        gold = {g for g in spans if g[2].endswith("_NAME")}
        tp += len(got & gold); fp += len(got - gold); fn += len(gold - got)
    p, r, f = prf(tp, fp, fn)
    print(f"\n== first / last name split (synthetic, new names + new phrasing) ==\n   F1 {f:.3f} (P {p:.3f} R {r:.3f})")

    texts = [t for t, _ in data.make("test", 2000, seed=7)]
    n_tok = sum(len(tokenize(t)) for t in texts)
    for t in texts[:50]:
        det.detect(t)
    lat = []
    for t in texts:
        a = time.perf_counter(); det.detect(t); lat.append((time.perf_counter() - a) * 1000)
    lat.sort()
    total = sum(lat) / 1000
    print(f"\n== latency, 1 CPU thread, {len(texts)} texts, avg {n_tok / len(texts):.0f} tokens ==")
    print(f"p50 {statistics.median(lat):.2f} ms   p95 {lat[int(.95 * len(lat))]:.2f} ms   "
          f"p99 {lat[int(.99 * len(lat))]:.2f} ms")
    print(f"throughput {len(texts) / total:.0f} texts/s, {n_tok / total:.0f} tokens/s per core")
    long = "\n".join(texts[:60])
    a = time.perf_counter(); det.detect(long); ms = (time.perf_counter() - a) * 1000
    print(f"long document ({len(tokenize(long))} tokens, {len(long)} chars): {ms:.1f} ms")


if __name__ == "__main__":
    main()
