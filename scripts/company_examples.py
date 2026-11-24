"""Page-like training examples built from a list of real company names (for example a register export).

    python scripts/company_examples.py companies.csv data/company_pages.jsonl --split train
    python scripts/company_examples.py companies.csv data/company_test.jsonl --split test

The CSV needs a "Name" (or "company_name") column; "Street", "Postal code" and "City" are used when present. Names are split 80/20 by a
hash of the name: --split test uses only the held-out fifth, with sentence patterns that --split train never uses,
so a test measures whether company names never seen in training are found.

Each company appears with its legal form ("Nordwind Logistik GmbH") and, more often, as the bare brand ("Nordwind
Logistik", also lower-case), because suffix-less brands are what the tagger misses most. Sole traders ("e. K.") are
skipped: their bare name is a person's name. The CSV itself is not part of this repository.
"""
import argparse
import csv
import json
import os
import random
import sys
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from tarnkappe.scan import bare_name        # noqa: E402
from build_distilled import furniture        # noqa: E402

TRAIN = [
    "Impressum\n{full}\n{street}\n{zip} {city}\nGeschäftsführer: {person}\nKontakt: {email}",
    "{person} ist seit {year} bei {bare} und verantwortet dort den Vertrieb.",
    "{bare} hat eine Finanzierungsrunde über {n} Millionen Euro abgeschlossen. Gründer {person} will damit expandieren.",
    "{person}\n{role} bei {bare}\n{city}",
    "Unsere Kunden: {bare}, {bare2} und {bare3}.",
    "{person} joined {bare} as {role_en} in {year}.",
    "{bare} announced a partnership with {bare2} on Tuesday.",
    "Speakers\n{person}, {bare}\n{person2}, {bare2}",
    "Ein Angebot der {full}, {street}, {zip} {city}.",
    "Network\n{full}\n{full2}\n{person}\npreviously\ncurrently",
    "Wir freuen uns auf die Zusammenarbeit mit {bare}! #{tag}",
    "{bare_lower} sucht ab sofort eine:n Werkstudent:in Marketing.",
]
TEST = [
    "Verantwortlich für den Inhalt: {full}, vertreten durch {person}, {street}, {zip} {city}.",
    "{person} wechselt von {bare2} zu {bare}.",
    "Referenzen\n{bare}\n{bare2}\n{bare3}",
    "Last week {bare} hired {person} to lead engineering.",
    "Mehr über {bare}: Das Team um {person} entwickelt Lösungen für den Mittelstand.",
    "{person} (Gründer, {bare})",
]
# Capitalised words and phrases that sit where names sit on web pages but are not companies: job titles, medical and
# academic fields, page buttons, technologies. Generic vocabulary, never names.
DISTRACTORS = """Ärztin für Allgemeinmedizin|Facharzt für Innere Medizin|Wahlarzt|Allgemeinmedizin|Zahnmedizin|Physiotherapie|
Kinderheilkunde|Ordination|Ordinationszeiten|Praxis|Sprechstunde|Steuerberater|Rechtsanwältin|Notar|Architekt|Heilpraktiker|
General practitioner|Internal Medicine|Dentistry|Physical therapy|Consulting Hours|Office Hours|Email & Phone Number|
Phone Number|Contact Information|View Profile|Request removal|Get valid emails|Free lookup|Show more|Read more|View all posts|
Conference Center|Student Union|Board of Directors|Advisory Board|Annual Report|Press Release|Terms of Use|Privacy Policy|
Machine Learning|Artificial Intelligence|Blockchain|Crypto|Web3|Smart Contract|Open Source|Cloud Computing|Data Science|
Software Engineering|Product Management|Digital Strategy|Online-Marketing|Projektmanagement|Unternehmensberatung|
Vertrieb|Personalwesen|Geschäftsführung|Datenschutz|Impressum|Haftung für Inhalte|Kundenservice|Öffnungszeiten|
Senior Director Digital|Head of Engineering|Chief Executive Officer|Managing Director|Founder|Co-Founder|Angel Investor|
API|SDK|PHP|Python|JavaScript|Kubernetes|Docker|Linux|HTML|JSON|CSS|SQL|Gründerszene|Startup|Mittelstand""".replace("\n", "").split("|")
NOISE_TPL = [
    "{person}\n{noise}\n{bare}\n{city}",
    "{noise}\n{noise2}\n{full}\n{street}\n{zip} {city}",
    "{person} – {noise} bei {bare}",
    "{noise} · {noise2} · {noise3}\n{person} works at {bare}.",
    "{person}, {noise}, {city}\n{noise2}: {phone}",
    "Mehr zu {noise}: {person} erklärt, wie {bare} damit arbeitet.",
    "{noise}\n{person}\n{noise2}\nemail: {email}",
]
ROLES = ["Head of Sales", "Geschäftsführer", "Projektleiterin", "Partner", "Senior Consultant", "Prokurist"]
ROLES_EN = ["CTO", "Head of Marketing", "managing director", "senior engineer", "advisor"]


FIRST = set(open(os.path.join(os.path.dirname(__file__), "..", "tarnkappe", "model", "first_names.txt"),
                encoding="utf-8").read().split("\n"))


def looks_like_person(name):
    """'Jens Bartram': a sole trader's name, i.e. a person's name. Not used as a company example."""
    words = name.split()
    return 2 <= len(words) <= 3 and words[0].lower() in FIRST and bare_name(name) is None and \
        all(w[:1].isupper() and w[1:].islower() for w in words)


def load(path, split):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    out = []
    for r in rows:
        name = (r.get("Name") or r.get("company_name") or "").strip()
        if name.endswith(" (The)"):                       # "Walt Disney Company (The)"
            name = "The " + name[:-6]
        if not name or "e. K." in name or looks_like_person(name):
            continue
        test = zlib.crc32(name.encode()) % 5 == 0
        if (split == "test") != test:
            continue
        bare = bare_name(name) or name
        out.append({"full": name, "bare": bare, "street": r.get("Street", ""), "zip": r.get("Postal code", ""),
                    "city": r.get("City", "")})
    return out


def load_all(paths, split):
    out = []
    for p in paths:
        out += load(p, split)
    return out


def render(tpl, values, labels):
    """Fill a template; return text and spans for the labelled slots."""
    text, spans, pos = "", [], 0
    import re
    for m in re.finditer(r"\{(\w+)\}", tpl):
        text += tpl[pos:m.start()]
        key = m.group(1)
        v = values[key]
        if key in labels and v:
            spans.append((len(text), len(text) + len(v), labels[key]))
        text += v
        pos = m.end()
    return text + tpl[pos:], spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="one CSV, or several joined with commas"); ap.add_argument("out")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--noise", action="store_true", help="add job titles, fields, page buttons and tech terms (not entities)")
    a = ap.parse_args()
    from faker import Faker
    rng = random.Random(21 if a.split == "train" else 22)
    fake = {"de": Faker("de_DE"), "en": Faker("en_US")}
    fake["de"].seed_instance(5); fake["en"].seed_instance(6)
    comps = load_all(a.csv.split(","), a.split)
    tpls = TRAIN if a.split == "train" else TEST
    if a.split == "train" and a.noise:
        tpls = TRAIN + NOISE_TPL * 2
    labels = {"full": "ORG", "full2": "ORG", "bare": "ORG", "bare2": "ORG", "bare3": "ORG", "bare_lower": "ORG",
              "person": "PERSON", "person2": "PERSON", "street": "STREET", "zip": "ZIP", "city": "LOCATION",
              "email": "EMAIL", "phone": "PHONE"}
    n = a.n if a.split == "train" else min(a.n, 600)
    with open(a.out, "w", encoding="utf-8") as f:
        for i in range(n):
            c, c2, c3 = rng.sample(comps, 3)
            fk = fake["en" if rng.random() < 0.35 else "de"]
            person = fk.first_name() + " " + fk.last_name()
            dom = "".join(ch for ch in c["bare"].lower() if ch.isalnum())[:20] or "firma"
            values = {"full": c["full"], "full2": c2["full"], "bare": c["bare"], "bare2": c2["bare"], "bare3": c3["bare"],
                      "bare_lower": c["bare"].lower(), "person": person, "person2": fk.first_name() + " " + fk.last_name(),
                      "street": c["street"] or fk.street_address(), "zip": c["zip"] or fk.postcode(),
                      "city": c["city"] or fk.city(),
                      "email": f"info@{dom}.de", "year": str(rng.randint(2012, 2026)), "n": str(rng.randint(2, 40)),
                      "role": rng.choice(ROLES), "role_en": rng.choice(ROLES_EN), "tag": dom[:12],
                      "noise": rng.choice(DISTRACTORS), "noise2": rng.choice(DISTRACTORS), "noise3": rng.choice(DISTRACTORS),
                      "phone": f"+49 30 {rng.randint(1000000, 9999999)}"}
            text, spans = render(rng.choice(tpls), values, labels)
            if a.split == "train" and rng.random() < 0.5:          # menu and footer lines around it
                top = furniture(rng, rng.randint(1, 6))
                head = "\n".join(top) + "\n"
                text, spans = head + text + "\n" + "\n".join(furniture(rng, rng.randint(0, 3))), \
                    [(s + len(head), e + len(head), l) for s, e, l in spans]
            f.write(json.dumps({"text": text.strip("\n") if not text.startswith("\n") else text, "spans": spans},
                               ensure_ascii=False) + "\n")
    print(f"{n} examples from {len(comps)} companies ({a.split}) -> {a.out}")


if __name__ == "__main__":
    main()
