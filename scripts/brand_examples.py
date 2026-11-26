"""Training sentences for brand names written without a legal form, in running text (v0.6).

    python scripts/brand_examples.py companies.csv data/brand_pages.jsonl --n 12000 [--names extra_names.txt]

The v0.5 tests showed what the tagger misses on real pages: well-known brands in prose ("X steigt bei Y ein"),
brands that look like surnames and are then taken for persons, lower-case startup names in lists, and organisations
other than companies (broadcasters, universities, clubs). This script writes news-, profile- and list-style sentences
in German and English where such names stand in the places persons usually stand: as subject of "said"/"sagte",
after "bei", in "Name, Firma, Stadt" lists, in genitives and compounds.

Name sources: the CSV (bare names, legal forms removed), --names (one name per line), and two invented kinds that
need no list: surname-like brands (a Faker surname, alone) and coined lower-case startup names. Persons in the same
sentences always carry a first name or a title, so a lone surname-like word next to a company cue is the brand.
Product names (a CMS, an app) appear in product contexts and are not labelled: they are not organisations
(see eval/README.md, "What counts as an organisation").
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from build_distilled import furniture          # noqa: E402
from company_examples import load_all, render  # noqa: E402

DE = [
    "{org} steigt bei {org2} ein und übernimmt die Mehrheit der Anteile.",
    "{org} beteiligt sich mit einem Minderheitsanteil an {org2}, bestätigt {person}.",
    "Interesse an einer Beteiligung zeigen {org}, {org2} und {org3}.",
    "„Wir sind sehr zufrieden“, sagt {person}, Gründer von {org}.",
    "{org} baut Stellen ab, wie {media} meldet.",
    "Investiert haben unter anderem {org}, {org2} und {org3}.",
    "{person} war zuvor bei {org} und {org2} tätig.",
    "{person} wechselt von {org} zu {org2}.",
    "Seit {year} arbeitet {person} als {role} bei {org}.",
    "{rank}. {person}, {org}, {city} {rank2}. {person2}, {org2}, {city2} {rank3}. {person3}, {org3}, {city3}",
    "Die Agentur {org} gewinnt den Etat von {org2}.",
    "Mit {org} startet {person} eine Plattform für Händler.",
    "Wie {media} berichtet, verhandelt {org} mit {org2} über eine Übernahme.",
    "{org}-Gründer {person} spricht über die Zukunft des Unternehmens.",
    "Die Studie entstand in Zusammenarbeit mit der {uni} und {org}.",
    "{person} studierte an der {uni} und gründete {year} {org}.",
    "Der Fernsehbeitrag des {media} zeigt, wie {org} entstanden ist.",
    "{org} hat {person} zum neuen Geschäftsführer ernannt.",
    "Kunden von {org} sind unter anderem {org2} und {org3}.",
    "Das Team von {org} veröffentlicht neue Funktionen für {product}.",
    "Der {club} gewann am Samstag gegen den {club2}.",
    "Bei {org} wollte man sich dazu nicht äußern.",
    "Frühere Stationen: {org} ({year}–{year2}), {org2}, {org3}.",
    "{person} ({role}, {org}) und {person2} ({org2}) diskutieren auf dem Podium.",
    "Das Start-up {org_low} aus {city} sammelt {n} Millionen Euro ein.",
    "Herr {surname} von {org} hat die Fragen beantwortet.",
    "Frau {surname} leitet bei {org} die Abteilung Kommunikation.",
]
EN = [
    "{org} announced on Tuesday that it will acquire {org2}.",
    "“We are excited about this partnership,” said {person}, chief executive of {org}.",
    "{person} previously worked at {org} and {org2}.",
    "After a short stint at {org}, {person} joined {org2} as {role_en}.",
    "{org} powers the search on sites such as {org2} and {org3}.",
    "Investors include {org}, {org2} and {org3}.",
    "According to {media}, {org} is in talks with {org2}.",
    "{org}'s new integration with {product} is a glimpse into the future, said {person}.",
    "{person} studied at the {uni_en} before co-founding {org}.",
    "The report was produced by {org} together with {org2}.",
    "{org} is a tool that lets developers test their APIs. {org} raised {n} million dollars.",
    "Clients include {org}, {org2}, {org3} and several public bodies.",
    "{org} cut {n} percent of its staff, {media} reported.",
    "Former employers: {org}, {org2}, {org3}",
    "{person} ({org}) and {person2} ({org2}) spoke on the panel.",
    "I work on the {org} API during the week and blog about {product} on weekends.",
    "{org_low} is a small startup from {city} that builds tools for retailers.",
    "Mr {surname} of {org} declined to comment.",
    "Ms {surname} has led marketing at {org} since {year}.",
    "{club_en} beat {club2_en} on Saturday.",
    "The plugin is available for {product} and {product2}; {org} maintains it.",
]
MEDIA = ["NDR", "WDR", "BR", "ZDF", "Die Welt", "die FAZ", "Handelsblatt", "der Spiegel", "die Süddeutsche Zeitung",
         "Tagesspiegel", "t3n", "Gründerszene", "Reuters", "Bloomberg", "the Financial Times", "TechCrunch", "the BBC",
         "The Guardian", "Le Monde", "heise online"]
MEDIA_LABEL = {m: m.split(" ", 1)[1] if m.split(" ", 1)[0] in ("die", "der", "the") else m for m in MEDIA}
UNIS = ["Universität Hamburg", "Freien Universität Berlin", "Universität der Künste Berlin", "TU München",
        "Universität Wien", "ETH Zürich", "Hochschule Darmstadt", "Universität zu Köln", "Uni Kiel", "RWTH Aachen"]
UNIS_EN = ["University of Manchester", "London School of Economics", "Michigan Tech", "University of Toronto",
           "Trinity College Dublin", "Stanford University", "University of Edinburgh", "Brown University"]
CLUBS = ["Hertha BSC", "FC St. Pauli", "SV Werder Bremen", "1. FC Köln", "Eintracht Frankfurt", "VfL Bochum"]
CLUBS_EN = ["Arsenal", "Leeds United", "Celtic", "Aston Villa", "Everton", "Brentford"]
PRODUCTS = ["WordPress", "Excel", "Snapchat", "Slack", "Photoshop", "Android", "iOS", "Joomla", "Postgres", "Figma",
            "Notion", "Kubernetes", "Chrome"]
ROLES = ["Geschäftsführer", "Head of Product", "Senior Advisor", "Vertriebsleiterin", "CTO", "Partnerin", "Prokurist"]
ROLES_EN = ["head of engineering", "chief executive", "senior advisor", "product manager", "partner", "CTO"]
SYL = ["mo", "ko", "no", "mix", "xt", "web", "lin", "mue", "sli", "smoo", "doo", "edel", "ight", "qy", "pe", "zu", "ra",
       "fy", "ly", "io", "pix", "tro", "va", "ne", "ta", "lo", "bu", "zz", "ki", "do"]


def coined(rng):
    """A lower-case startup name such as 'mixxt' or 'smoodoos'."""
    w = "".join(rng.choice(SYL) for _ in range(rng.randint(2, 3)))
    return (rng.choice(["my", "go", "", "", ""]) + w)[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="one CSV, or several joined with commas"); ap.add_argument("out")
    ap.add_argument("--names", default="", help="extra organisation names, one per line")
    ap.add_argument("--n", type=int, default=12000)
    a = ap.parse_args()
    from faker import Faker
    rng = random.Random(61)
    fake = {"de": Faker("de_DE"), "en": Faker("en_GB")}
    fake["de"].seed_instance(61); fake["en"].seed_instance(62)
    listed = [c["bare"] for c in load_all(a.csv.split(","), "train")]
    if a.names:
        listed += [l.strip() for l in open(a.names, encoding="utf-8") if l.strip()]
    listed = [n for n in dict.fromkeys(listed) if 2 <= len(n) <= 40]
    print(f"{len(listed)} listed names")

    def org(lang):
        r = rng.random()
        if r < 0.55:
            name = rng.choice(listed)
        elif r < 0.85:                                     # a surname used as a brand
            name = fake[lang].last_name()
            if rng.random() < 0.3:
                name += rng.choice(["-" + fake[lang].last_name(), " & " + fake[lang].last_name()])
        else:
            name = coined(rng)
        return name

    labels = {k: "ORG" for k in ["org", "org2", "org3", "org_low", "media", "uni", "uni_en", "club", "club2",
                                 "club_en", "club2_en"]}
    labels.update({"person": "PERSON", "person2": "PERSON", "person3": "PERSON", "surname": "PERSON",
                   "city": "LOCATION", "city2": "LOCATION", "city3": "LOCATION"})
    with open(a.out, "w", encoding="utf-8") as f:
        for _ in range(a.n):
            lang = "de" if rng.random() < 0.55 else "en"
            fk = fake[lang]
            media = rng.choice(MEDIA)
            values = {"org": org(lang), "org2": org(lang), "org3": org(lang), "org_low": coined(rng),
                      "person": fk.first_name() + " " + fk.last_name(), "person2": fk.first_name() + " " + fk.last_name(),
                      "person3": fk.first_name() + " " + fk.last_name(), "surname": fk.last_name(),
                      "city": fk.city(), "city2": fk.city(), "city3": fk.city(),
                      "media": media, "uni": rng.choice(UNIS), "uni_en": rng.choice(UNIS_EN),
                      "club": rng.choice(CLUBS), "club2": rng.choice(CLUBS), "club_en": rng.choice(CLUBS_EN),
                      "club2_en": rng.choice(CLUBS_EN), "product": rng.choice(PRODUCTS), "product2": rng.choice(PRODUCTS),
                      "role": rng.choice(ROLES), "role_en": rng.choice(ROLES_EN), "year": str(rng.randint(1998, 2026)),
                      "year2": str(rng.randint(1998, 2026)), "n": str(rng.randint(2, 90)),
                      "rank": str(rng.randint(1, 17)), "rank2": "", "rank3": ""}
            values["rank2"] = str(int(values["rank"]) + 1); values["rank3"] = str(int(values["rank"]) + 2)
            if rng.random() < 0.3:
                values["org"] = values["org"].lower() if " " not in values["org"] else values["org"]
            text, spans = render(rng.choice(DE if lang == "de" else EN), values, labels)
            # "die FAZ": only the name is the organisation
            fixed = []
            for s, e, l in spans:
                if text[s:e] in MEDIA_LABEL and MEDIA_LABEL[text[s:e]] != text[s:e]:
                    s = e - len(MEDIA_LABEL[text[s:e]])
                fixed.append((s, e, l))
            spans = fixed
            if rng.random() < 0.4:                            # menu and footer lines around it
                head = "\n".join(furniture(rng, rng.randint(1, 5))) + "\n"
                text = head + text + "\n" + "\n".join(furniture(rng, rng.randint(0, 3)))
                spans = [(s + len(head), e + len(head), l) for s, e, l in spans]
            f.write(json.dumps({"text": text.rstrip("\n"), "spans": spans}, ensure_ascii=False) + "\n")
    print(f"{a.n} examples -> {a.out}")


if __name__ == "__main__":
    main()
