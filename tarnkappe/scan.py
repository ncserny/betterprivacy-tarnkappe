"""Scan-level company matching.

One person's scan covers many pages that mention the same few companies. A company that is
found once with a legal suffix ("appstretto UG") can then be matched by its bare name
("appstretto") on every other page, and employer names from the person's profile can be
matched directly. This needs no external company database and adds almost no false alarms,
because every name comes from that person's own results.

    from tarnkappe import Detector
    from tarnkappe.scan import detect_scan
    results = detect_scan(Detector(), page_texts, profile_companies=["compuccino"])
"""
import re

LEGAL = (r"(?:GmbH\s*&\s*Co\.?\s*KGaA|GmbH\s*&\s*Co\.?\s*KG|gGmbH|GmbH|UG\s*\(haftungsbeschränkt\)|UG|AG|KGaA|KG|OHG|SE|"
         r"e\.\s?V\.|eG|GbR|Inc\.?|LLC|Ltd\.?|Limited|LLP|PLC|plc|Corp\.?|Corporation|S\.A\.|SAS|SARL|B\.V\.|BV|N\.V\.|S\.r\.l\.|AB|Oy|ApS|A/S)")
SUFFIX = re.compile(r"[\s,]+" + LEGAL + r"(?:\s*\(haftungsbeschränkt\))?\s*$")
ACRONYM = re.compile(r"^\s*\(([A-ZÄÖÜ]{2,6})\)")
# words that are not a company name on their own
GENERIC = {"agentur", "kommunikation", "gruppe", "group", "holding", "consulting", "solutions", "systems", "services",
           "media", "digital", "software", "technologies", "partners", "ventures", "management", "company", "firma",
           "the", "die", "der", "das", "und", "and", "deutsche", "international", "berlin", "hamburg", "münchen", "köln"}


def bare_name(company):
    """'appstretto UG (haftungsbeschränkt)' -> 'appstretto'; None if there is no legal suffix or nothing distinctive is left."""
    m = SUFFIX.search(company)
    if not m:
        return None
    name = company[:m.start()].strip(" ,.-")
    words = [w for w in re.split(r"\s+", name) if w]
    if not words or len(name) < 3 or all(w.lower() in GENERIC for w in words):
        return None
    return name


def harvest(text, entities):
    """Company names worth matching elsewhere in the scan, taken from one page's detections."""
    out = set()
    for e in entities:
        if e["type"] != "ORGANIZATION":
            continue
        for line in e["text"].split("\n"):              # a detection can run over several list lines
            name = bare_name(line)
            if not name:
                continue
            out.add(name)
            last = name.split()[-1]
            # "Berliner Medienagentur compuccino GmbH": a lower-case final word is the brand itself
            if " " in name and last[0].islower() and len(last) >= 4 and last.lower() not in GENERIC:
                out.add(last)
        if bare_name(e["text"].split("\n")[-1]):
            m = ACRONYM.match(text[e["end"]:])          # "Oliver Schrott Kommunikation GmbH (OSK)"
            if m:
                out.add(m.group(1))
    return out


def company_spans(text, names):
    """Occurrences of known company names. Names of five or more letters match in any case;
    shorter ones and acronyms must match exactly. A following "'s" or "-Gründer" is allowed."""
    spans = []
    for name in sorted(set(names), key=len, reverse=True):
        if len(name) < 3:
            continue
        flags = re.IGNORECASE if len(name) >= 5 and not name.isupper() else 0
        pat = re.compile(r"(?<![\w@])" + re.escape(name).replace(r"\ ", r"\s+") + r"(?![\w])", flags)
        for m in pat.finditer(text):
            if all(m.end() <= a or m.start() >= b for a, b in spans):
                spans.append((m.start(), m.end()))
    return [{"type": "ORGANIZATION", "start": a, "end": b, "score": 1.0, "source": "list", "text": text[a:b]}
            for a, b in sorted(spans)]


def apply_known(text, entities, names):
    """Add list matches to one page's detections. A list match replaces model detections it
    overlaps, except e-mail addresses, IBANs and other pattern hits."""
    extra = company_spans(text, names)
    keep = []
    for e in entities:
        clash = [x for x in extra if x["start"] < e["end"] and x["end"] > e["start"]]
        if clash and e["source"] != "model":
            extra = [x for x in extra if x not in clash]        # pattern hits win
            keep.append(e)
        elif not clash:
            keep.append(e)
        elif e["type"] == "ORGANIZATION" and any(e["start"] <= x["start"] and e["end"] >= x["end"] for x in clash):
            extra = [x for x in extra if x not in clash]        # the model already has the longer company span
            keep.append(e)
    return sorted(keep + extra, key=lambda e: e["start"])


def detect_scan(detector, texts, profile_companies=(), **kwargs):
    """Detect PII on all pages of one scan, sharing company names between pages.
    Returns one entity list per text, in order."""
    first = [detector.detect(t, **kwargs) for t in texts]
    names = set(profile_companies)
    for t, ents in zip(texts, first):
        names |= harvest(t, ents)
    return [apply_known(t, ents, names) for t, ents in zip(texts, first)]
