"""Scan-level matching of company names and of the scanned person's own name.

One person's scan covers many pages that mention the same few companies. A company that is
found once with a legal suffix ("appstretto UG") can then be matched by its bare name
("appstretto") on every other page, and employer names from the person's profile can be
matched directly. This needs no external company database and adds almost no false alarms,
because every name comes from that person's own results.

    from tarnkappe import Detector
    from tarnkappe.scan import detect_scan
    results = detect_scan(Detector(), page_texts, profile_companies=["Nordwind Logistik"],
                          profile_names=["Jana Beispiel"])

The scanned person's own name is known before any page is read. Passing it as profile_names
makes sure it is found everywhere: alone in a headline ("Beispiel verstärkt ..."), in capitals,
or inside a domain name ("beispiel.berlin").
"""
import os
import re

LEGAL = (r"(?:GmbH\s*&\s*Co\.?\s*KGaA|GmbH\s*&\s*Co\.?\s*KG|gGmbH|GmbH|GMBH|mbH|gUG|UG\s*\(haftungsbeschränkt\)|UG|GUG|AG|KGaA|KG|OHG|SE|"
         r"e\.\s?V\.|e\.\s?K\.|eG|GbR|Inc\.?|INC\.?|LLC|L\.L\.C\.|Ltd\.?|LTD\.?|Limited|LIMITED|LLP|L\.P\.|LP|PLC|plc|Corp\.?|CORP\.?|"
         r"Corporation|CORPORATION|S\.A\.U?|SA\.?|S\.p\.A\.|SpA|SPA|S\.a\.?\s?r\.l\.|S\.à\s?r\.l\.|SAS|S\.A\.S\.|SARL|SRL|S\.r\.l\.|S\.L\.U?\.?|SLU?|"
         r"B\.V\.|BV|N\.V\.|NV|C\.V\.|AB|Oy|ApS|A/S|AS|ASA|Sp\.\s?z\s?o\.o\.)")
# "Nordwind Logistik GmbH", "Kranich Media GmbH", "BEISPIEL INTERIORS UG": up to five name words on one line, then a legal form
_W = r"[\w&+'’-]*(?:\.[\w&+'’-]+)*"                      # dots only inside a word ("blog.de"), never at its end
_NOT_FORM = r"(?!" + LEGAL + r"(?![\w-]))"                  # a name word is never itself a legal form
_CAP = _NOT_FORM + r"[A-ZÄÖÜ0-9]" + _W
_LOW = _NOT_FORM + r"[a-zäöü]" + _W
COMPANY = re.compile(r"(?<![\w@.-])((?:" + _CAP + r"[ \t]+(?:&[ \t]+)?){0,4}(?:" + _LOW + r"[ \t]+)?(?:" + _CAP + "|" + _LOW +
                     r"))[ \t]*,?[ \t]+" + LEGAL + r"(?![\w-])")
RISKY = {"AB", "SE", "SAS", "Oy", "BV", "NV", "AG", "SA", "SA.", "SL", "SLU", "LP", "AS", "SPA", "SpA"}             # short forms that are also ordinary words or model names
LEADING = {"die", "der", "das", "den", "dem", "des", "the", "bei", "von", "vom", "für", "mit", "im", "in", "an", "am", "und", "and", "at",
           "by", "for", "with", "als", "zur", "zum", "a", "an", "of", "from", "seit", "since", "auch", "laut", "ein", "eine"}
SUFFIX = re.compile(r"[\s,]+" + LEGAL + r"(?:\s*\(haftungsbeschränkt\))?\s*$")
ACRONYM = re.compile(r"^\s*\(([A-ZÄÖÜ]{2,6})\)")
# words that are not a company name on their own
GENERIC = {"agentur", "kommunikation", "gruppe", "group", "holding", "consulting", "solutions", "systems", "services",
           "media", "digital", "software", "technologies", "partners", "ventures", "management", "company", "firma",
           "the", "die", "der", "das", "und", "and", "deutsche", "international", "berlin", "hamburg", "münchen", "köln"}


def legal_form_spans(text):
    """Companies written with their legal form. This is a rule, not a guess: "<Name> GmbH" is a company."""
    out = []
    for m in COMPANY.finditer(text):
        a = m.start()
        words = m.group(1).split()
        while len(words) > 1 and words[0].lower() in LEADING:      # "Die Nordwind GmbH" -> "Nordwind GmbH"
            a = text.index(words[1], a + len(words[0]))
            words.pop(0)
        if words[0].lower() in LEADING or words[-1].lower() in LEADING or len(" ".join(words)) < 3:
            continue                                               # "der gUG", "Betrieb der gUG": no name before the form
        form = text[m.end(1):m.end()].strip(" \t,")
        if form in RISKY and (any(w[0].islower() for w in words) or words[-1].isdigit()):
            continue                                               # "He is in AB", "iPhone SE", "Fiat 500 SE"
        out.append({"type": "ORGANIZATION", "start": a, "end": m.end(), "score": 1.0, "source": "rule", "text": text[a:m.end()]})
    return out


SURE = 0.9


# Words right before a company name in CVs, bios, press and profiles. The capitalised words after them are proposed as
# a company; the company check (orgcheck.py) decides. Persons and places the tagger already found are left alone.
CUE_NOUNS = (r"Agentur|Firma|Unternehmen|Unternehmens|Startups?|Start-ups?|Beteiligungsunternehmens?|Plattform|Marke|Kunden?|"
             r"agency|company|firm|startup|brand|client|platform|employer|Arbeitgeber")
CONTEXT = re.compile(
    r"(?:(?i:\b(?:work(?:s|ed|ing)?|employed|engineer|developer|consultant|intern(?:ship)?|role|position|job|career)\s+(?:at|for|with))"
    r"|(?i:\b(?:joined|joins|joining|left|leaving|leaves|acquired|acquires|founded|co-?founded|cofounded|launched))"
    r"|(?i:\b(?:founder|co-?founder|ceo|cto|cfo|coo|director|head|manager|partner|advisor|investor|owner|president|vp)"
    r"(?:\s+[\w&-]+){0,3}?\s+(?:at|@))"
    r"|\s@"
    r"|\b(?:arbeitet(?:e)?|tätig|beschäftigt|angestellt|Praktikum|Ausbildung)\s+(?:bei|für)"
    r"|\b(?:wechselt(?:e)?|kam|kommt|ging|geht)\s+(?:er\s+|sie\s+)?(?:\w+\s+){0,4}?(?:zu|von)"
    r"|\b(?:Gesellschafter(?:in)?|Gründer(?:in)?|Mitgründer(?:in)?|Mitinhaber(?:in)?|Inhaber(?:in)?|Geschäftsführer(?:in)?|"
    r"Partner(?:in)?|Vorstand|Beirat|Berater(?:in)?)\s+(?:\w+\s+){0,3}?(?:bei|von|für)"
    r"|\b(?:" + CUE_NOUNS + r")"
    r")[ \t]+(?:(?i:a|an|the|der|die|das|den|dem|des|einer?|einem|eines)[ \t]+)?(?=[A-ZÄÖÜ0-9]|[a-zäöü]+[A-Z]|[a-z]{4})")
CUE_START = re.compile(r"(?:" + CUE_NOUNS + r")\b")
NAME_AFTER = re.compile(r"(?:[A-ZÄÖÜ0-9][\w&'’.-]*|[a-zäöü]+[A-Z][\w-]*|[a-zäöü][a-zäöü0-9-]{3,})(?:\s(?:&\s)?[A-ZÄÖÜ][\w&'’.-]*){0,3}")
NOT_NAMES = {"I", "He", "She", "We", "They", "It", "The", "A", "An", "This", "That", "My", "Our", "His", "Her", "Their",
             "Er", "Sie", "Es", "Wir", "Ich", "Der", "Die", "Das", "Den", "Dem", "Des", "Ein", "Eine", "Einer", "Mein",
             "Anfang", "Ende", "Mitte", "Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September",
             "Oktober", "November", "Dezember", "January", "February", "March", "June", "July", "October", "December",
             "Home", "Senior", "Junior", "Lead", "Head", "Chief", "Mr", "Mrs", "Ms", "Dr", "Herr", "Frau", "Prof"}


def context_company_spans(text, entities, common=None):
    """Company candidates from the words around them: 'worked at Nordwind', 'Senior Advisor bei Nordwind',
    'Head of Sales @ Nordwind'. Returned with source 'context'; pass them through the company check."""
    taken = [(e["start"], e["end"]) for e in entities]
    out = []
    for m in CONTEXT.finditer(text):
        n = NAME_AFTER.match(text, m.end())
        if not n:
            continue
        name = n.group().rstrip(".'’-")
        words = name.split()
        while words and words[-1] in NOT_NAMES:
            words.pop()
        if not words or words[0] in NOT_NAMES or CUE_START.match(words[0]) or len(name) < 3 or "\n" in name:
            continue
        if words[0].islower():                        # a lower-case brand only right after "Agentur" / "agency" ...
            if not re.search(r"(?:Agentur|agency|Firma|company)\s+$", text[max(0, n.start() - 12):n.start()]):
                continue
            st = (common or {}).get(words[0])          # ... and only if it is not a common word ("erhalten")
            if st and st[0] >= 20:
                continue
        name = " ".join(words)
        a, b = n.start(), n.start() + len(name)
        if any(x < b and y > a for x, y in taken):
            continue
        out.append({"type": "ORGANIZATION", "start": a, "end": b, "score": 0.5, "source": "context", "text": name})
        taken.append((a, b))
    return out


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


def harvest(text, entities, confident=None):
    """Company names worth matching elsewhere in the scan, taken from one page's detections.
    Names with a legal form ("appstretto UG" -> "appstretto") are returned. Names the tagger is merely
    very sure about are added to `confident` (a set) instead: detect_scan shares those only when they
    were found on at least two different pages, so one wrong guess is not spread across the scan."""
    out = set()
    for e in entities:
        if e["type"] != "ORGANIZATION":
            continue
        name = e["text"].strip()
        if (confident is not None and e.get("source") == "model" and e.get("score", 0) >= SURE and "\n" not in name
                and 4 <= len(name) <= 40 and len(name.split()) <= 4 and (name[0].isupper() or "." in name)
                and not all(w.lower() in GENERIC for w in name.split())):
            confident.add(name)
        for line in e["text"].split("\n"):              # a detection can run over several list lines
            name = bare_name(line)
            if not name:
                continue
            out.add(name)
            last = name.split()[-1]
            # "Berliner Medienagentur kranichblau GmbH": a lower-case final word is the brand itself
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


def person_spans(text, names, split_names=True):
    """Occurrences of a known person's name: the full name, and each part on its own. Parts of
    five or more letters also match in lower case (domain names); shorter parts must be
    capitalised or in capitals. Not matched right after "@" (handles are not persons)."""
    spans = []
    def scan(pattern, flags, kind):
        for m in re.finditer(r"(?<![\w@])" + pattern + r"(?![\w])", text, flags):
            if all(m.end() <= a or m.start() >= b for a, b, _ in spans):
                spans.append((m.start(), m.end(), kind))
    people = [[p for p in re.split(r"\s+", n.strip()) if p] for n in names]
    for parts in people:                                   # full names first, so they win over single parts
        if len(parts) >= 2:
            scan(r"\s+".join(re.escape(p) for p in parts), re.IGNORECASE, "full")
    for parts in people:
        for i, p in enumerate(parts):
            if len(p) < 3 or not p[0].isalpha():
                continue
            kind = "LAST_NAME" if i == len(parts) - 1 and len(parts) > 1 else "FIRST_NAME"
            scan("(?:" + re.escape(p[0].upper() + p[1:]) + "|" + re.escape(p.upper()) +
                 ("|" + re.escape(p.lower()) if len(p) >= 5 else "") + ")", 0, kind)
    out = []
    mk = lambda typ, a, b: {"type": typ, "start": a, "end": b, "score": 1.0, "source": "list", "text": text[a:b]}
    for a, b, kind in sorted(spans):
        if not split_names:
            out.append(mk("PERSON", a, b))
        elif kind == "full":
            cut = a + text[a:b].rindex(re.split(r"\s+", text[a:b])[-1])
            out += [mk("FIRST_NAME", a, a + len(text[a:cut].rstrip())), mk("LAST_NAME", cut, b)]
        else:
            out.append(mk(kind, a, b))
    return out


PERSON_TYPES = {"PERSON", "FIRST_NAME", "LAST_NAME"}


_INST = None


def institutions():
    """(name, ambiguous) pairs from model/institutions.txt, longest first."""
    global _INST
    if _INST is None:
        path = os.path.join(os.path.dirname(__file__), "model", "institutions.txt")
        out = []
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                line = line.rstrip("\n")
                if not line.strip() or line.startswith("#"):
                    continue
                amb = line.endswith(" ~")
                out.append((line[:-2] if amb else line, amb))
        _INST = sorted(out, key=lambda x: -len(x[0]))
    return _INST


def institution_spans(text, entities=()):
    """Mentions of well-known organisations from model/institutions.txt: case-sensitive, whole words, a genitive
    "s" or a hyphenated compound ("SPD-Fraktion") allowed. Ambiguous names ("Grüne") only count when another
    organisation is in the same sentence."""
    hits = []
    for name, amb in institutions():
        pat = re.compile(r"(?<![\w@/.])" + re.escape(name) + r"(?=s?(?![\w]))")
        for m in pat.finditer(text):
            if all(m.end() <= a or m.start() >= b for a, b, _ in hits):
                hits.append((m.start(), m.end(), amb))
    orgs = [(e["start"], e["end"]) for e in entities if e["type"] == "ORGANIZATION"] + \
           [(a, b) for a, b, amb in hits if not amb]
    out = []
    for a, b, amb in sorted(hits):
        if amb:
            lo = max(text.rfind(c, 0, a) for c in ".!?\n") + 1
            his = [x for x in (text.find(c, b) for c in ".!?\n") if x >= 0]
            hi = min(his) if his else len(text)
            if not any(lo <= x < hi and not (x == a and y == b) for x, y in orgs):
                continue
        out.append({"type": "ORGANIZATION", "start": a, "end": b, "score": 1.0, "source": "list", "text": text[a:b]})
    return out


def apply_known(text, entities, names, person_names=(), split_names=True, extra=()):
    """Add list matches to one page's detections. A list match replaces model detections it
    overlaps, except e-mail addresses, IBANs and other pattern hits."""
    if person_names:
        people = person_spans(text, person_names, split_names)
        keep = []
        for e in entities:
            clash = [x for x in people if x["start"] < e["end"] and x["end"] > e["start"]]
            if not clash:
                keep.append(e)
            elif e["source"] != "model" or e["type"] in PERSON_TYPES:
                people = [x for x in people if x not in clash]      # pattern hits and the model's own person spans win
                keep.append(e)
        entities = sorted(keep + people, key=lambda e: e["start"])
    extra = sorted(list(extra) + [x for x in company_spans(text, names)
                                  if all(x["end"] <= y["start"] or x["start"] >= y["end"] for y in extra)],
                   key=lambda x: x["start"])
    keep = []
    for e in entities:
        clash = [x for x in extra if x["start"] < e["end"] and x["end"] > e["start"]]
        if clash and e["source"] != "model":
            extra = [x for x in extra if x not in clash]        # pattern hits win
            keep.append(e)
        elif not clash:
            keep.append(e)
        elif e["type"] == "ORGANIZATION" and any(e["start"] <= x["start"] and e["end"] >= x["end"] for x in clash) \
                and not any(e["start"] == x["start"] and e["end"] == x["end"] for x in clash):
            extra = [x for x in extra if x not in clash]        # the model already has the longer company span
            keep.append(e)
    return sorted(keep + extra, key=lambda e: e["start"])


def _shareable(name):
    return ("\n" not in name and 4 <= len(name) <= 40 and len(name.split()) <= 4 and (name[0].isupper() or "." in name)
            and not all(w.lower() in GENERIC for w in name.split()))


def detect_scan(detector, texts, profile_companies=(), profile_names=(), **kwargs):
    """Detect PII on all pages of one scan, sharing company names between pages.
    A name with a legal form is shared from one page. Any other company name is shared when the
    company check accepted it on at least two different pages.
    profile_names: the scanned person's own name(s), matched on every page.
    Returns one entity list per text, in order."""
    first = [detector.detect(t, org_check=False, **kwargs) for t in texts]
    split = kwargs.get("split_names", True)
    cut = detector.org_cut(kwargs.get("profile")) if hasattr(detector, "org_cut") else None
    names = set(profile_companies)
    pages = {}
    for t, ents in zip(texts, first):
        names |= harvest(t, ents)
        ok = detector.check_orgs(t, ents, cut) if cut is not None else ents
        mine = {e["text"].strip() for e in ok if e["type"] == "ORGANIZATION" and e.get("source") in ("model", "context")
                and _shareable(e["text"].strip())}
        mine = {n for n in mine if cut is not None or any(e["text"].strip() == n and e.get("score", 0) >= SURE for e in ents)}
        for n in mine:
            pages.setdefault(n.lower(), [set(), 0]); pages[n.lower()][0].add(n); pages[n.lower()][1] += 1
    names |= {n for forms, k in pages.values() if k >= 2 for n in forms}
    out = []
    for t, ents in zip(texts, first):
        ents = apply_known(t, ents, names, profile_names, split)
        out.append(detector.check_orgs(t, ents, cut) if cut is not None else ents)
    return out
