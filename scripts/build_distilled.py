"""Turn teacher-labelled Wikipedia paragraphs into training examples that look like web pages.

    python scripts/distil_label.py data/wiki_snippets.txt data/wiki_gliner2.jsonl --teacher gliner2-plain --threshold 0.3
    python scripts/build_distilled.py data/wiki_gliner2.jsonl data/distilled_pages.jsonl
    python train.py --extra data/distilled_pages.jsonl

What it does:

1. Keeps a teacher label only when the teacher is confident (ORG 0.8, PERSON 0.7, LOCATION 0.6, others 0.5; chosen
   on the owner's first 27 hand-labelled snippets), an organisation is not a lower-case common noun, and a one-word
   person is also a person for the current tagger. Other teacher labels become "?": the text stays in the example
   but takes no part in the training loss, so the tagger is neither told that it is an entity nor that it is not.
2. Does the same wherever the current tagger sees a person, organisation or location and the teacher sees nothing.
   The teacher misses about one person in five on real pages; this keeps those misses from being taught.
3. Wraps half of the paragraphs in page furniture: menu lines above, footer lines below, separated by line breaks.
   Menu words are never entities.
4. Marks every bare number outside a kept label as "?" (the teacher rarely labels postcodes and phone numbers here).
5. Adds list-style examples built from names the teacher was very sure about (0.9): register lists, speaker lists,
   bylines, signature blocks, "follow us on" rows.
"""
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

KEEP = {"ORG": 0.8, "PERSON": 0.7, "LOCATION": 0.6, "STREET": 0.5, "ZIP": 0.5, "EMAIL": 0.5, "PHONE": 0.5}

MENU = """Home|About|About us|Pricing|Blog|News|Contact|Log in|Sign in|Sign up|Get started|Search|Menu|Products|Services|
Solutions|Careers|Jobs|Press|Team|Support|Help|FAQ|Docs|Community|Events|Resources|Partners|Customers|Platform|Features|
Overview|Follow|Followers|Following|Posts|Share|Share this article|Read more|Learn more|Subscribe|Newsletter|Download|
Privacy|Privacy policy|Terms|Terms of service|Imprint|Cookies|Cookie settings|Accept all|Back|Next|Previous|More|Watch|
Network|Location|Related people|Previous companies|Org chart|View all|Show more|Unverified|Verified|Latest|Trending|
Popular|Categories|Tags|Archive|Comments|Reply|Like|Likes|Views|min read|Listen|Save|Edit|Settings|Profile|Account|
Dashboard|Notifications|Messages|Activity|Media|Featured|Joined|Owner|Member|Admin|Sponsored|Advertisement|
Startseite|Über uns|Preise|Kontakt|Anmelden|Registrieren|Suche|Menü|Produkte|Leistungen|Dienstleistungen|Lösungen|
Karriere|Stellenangebote|Presse|Aktuelles|Nachrichten|Hilfe|Veranstaltungen|Termine|Partner|Kunden|Referenzen|Übersicht|
Folgen|Teilen|Artikel teilen|Weiterlesen|Mehr erfahren|Mehr|Abonnieren|Herunterladen|Datenschutz|Datenschutzerklärung|
Impressum|Nutzungsbedingungen|AGB|Alle akzeptieren|Einstellungen|Zurück|Weiter|Vorherige|Nächste|Beliebt|Kategorien|
Schlagwörter|Archiv|Kommentare|Antworten|Gefällt mir|Aufrufe|Lesezeit|Speichern|Bearbeiten|Profil|Konto|Benachrichtigungen|
Nachrichten|Aktivität|Medien|Beigetreten|Mitglied|Anzeige|Angaben|Verantwortlich|Haftung für Inhalte|Inhalt|Themen|
Rubriken|Politik|Wirtschaft|Kultur|Sport|Wissen|Digital|Gesellschaft|Meinung|Interviews|Startups|Investoren|Know How|
Sie befinden sich hier:|Zum Inhalt springen|Skip to content|All rights reserved|Alle Rechte vorbehalten|Powered by|
Quick search|Power search|all countries|Kontaktformular|Artikel vor/zurück|Aktuelle Artikel|Top Stories|Breaking|Live|
Deutsch|English|DE|EN|Seite drucken|Print|Sitemap|Feedback|Barrierefreiheit|Accessibility|Status|API|Changelog""".replace("\n", "").split("|")
ROLES_EN = ["CEO", "Founder", "Co-Founder", "Managing Director", "Partner", "Director of Engineering", "Head of Marketing",
            "Angel Investor", "Senior Advisor", "Editor", "Author", "Speaker", "Board Member", "CTO", "Product Manager"]
ROLES_DE = ["Geschäftsführer", "Geschäftsführerin", "Gründer", "Mitgründerin", "Gesellschafter", "Vorstand", "Partner",
            "Leiter Marketing", "Redakteurin", "Autor", "Referentin", "Beirat", "Inhaber", "Prokurist", "Projektleiterin"]
SOCIAL = ["LinkedIn", "Instagram", "Facebook", "YouTube", "Twitter", "Xing", "GitHub", "TikTok", "Mastodon", "Vimeo",
          "Pinterest", "WhatsApp", "Telegram", "Medium", "Reddit", "Twitch", "Spotify", "Flickr", "Bluesky", "Threads"]


def trusted(text, a, b, lab, score, mine):
    """Is this teacher label good enough to teach? mine: the current tagger's own spans [(start, end, TYPE)]."""
    if score < KEEP.get(lab, 0.5):
        return False
    word = text[a:b]
    if lab == "ORG" and word.islower() and "." not in word and " " not in word:
        return False                       # "company", "magazine": a common noun, not a name
    if lab == "PERSON" and " " not in word.strip():
        # a single word: only with the current tagger's agreement ("Akzeptor" is not a person)
        return any(x < b and y > a and t == "PERSON" for x, y, t in mine)
    return True


def resolve(text, spans, mine=(), extra=()):
    """Teacher spans with scores -> non-overlapping (start, end, label) with "?" for the unsure ones."""
    cand = [(s, a, b, lab if trusted(text, a, b, lab, s, mine) else "?") for a, b, lab, s in spans]
    cand.sort(key=lambda x: (x[3] == "?", -x[0]))                 # confident first, then by score
    out = []
    for _, a, b, lab in cand + [(0, a, b, "?") for a, b in extra]:
        if all(b <= x or a >= y for x, y, _ in out):
            out.append((a, b, lab))
    return sorted(out)


def furniture(rng, n):
    lines = []
    while len(lines) < n:
        w = rng.choice(MENU)
        k = rng.random()
        lines.append(w.upper() if k < 0.15 else w.lower() if k < 0.2 else w)
        if rng.random() < 0.08:
            lines.append(str(rng.randint(2, 4800)))
    return lines


def wrap(rng, text, spans):
    """Menu lines above and/or footer lines below a paragraph."""
    top = furniture(rng, rng.randint(2, 9)) if rng.random() < 0.8 else []
    bottom = furniture(rng, rng.randint(1, 5)) if rng.random() < 0.6 or not top else []
    sep = rng.choice(["\n", "\n", "\n\n"])
    head = sep.join(top) + (sep if top else "")
    return head + text + (sep + sep.join(bottom) if bottom else ""), [(a + len(head), b + len(head), l) for a, b, l in spans]


def lists(rng, orgs, persons, places, n):
    """List-shaped page parts made from names the teacher was very sure about."""
    out = []

    def build(parts):                      # parts: (text, label or None)
        text, spans = "", []
        for t, lab in parts:
            if lab:
                spans.append((len(text), len(text) + len(t), lab))
            text += t
        return text, spans

    def org():
        o = rng.choice(orgs)
        return o.lower() if rng.random() < 0.12 else o.upper() if rng.random() < 0.06 else o

    for _ in range(n):
        k = rng.random()
        de = rng.random() < 0.5
        roles = ROLES_DE if de else ROLES_EN
        parts = []
        if k < 0.2:                        # register / portfolio list: one name per line
            for w in furniture(rng, rng.randint(1, 4)):
                parts += [(w, None), ("\n", None)]
            for _ in range(rng.randint(3, 9)):
                r = rng.random()
                parts += [(org(), "ORG")] if r < 0.55 else [(rng.choice(persons), "PERSON")] if r < 0.85 else [(rng.choice(roles), None)]
                parts.append(("\n", None))
        elif k < 0.4:                      # speaker list: "Name von Firma" / "Name, Firma" / "Name Firma"
            parts += [(rng.choice(["Auf dem Podium sitzen:", "Speakers", "Referenten", "Team", "Mit dabei:", "Line-up"]), None), ("\n", None)]
            for _ in range(rng.randint(2, 6)):
                glue = rng.choice([" von ", ", ", " ", " – ", " (", " at ", " | "])
                parts += [(rng.choice(persons), "PERSON"), (glue, None), (org(), "ORG"), (")" if glue == " (" else "", None), ("\n", None)]
        elif k < 0.6:                      # signature block: name / role, company / place
            at = rng.choice([" bei ", ", ", " von ", " | "] if de else [" at ", ", ", " of ", " | ", ", Founder of "])
            parts += [(rng.choice(persons), "PERSON"), ("\n", None), (rng.choice(roles), None), (at, None), (org(), "ORG"), ("\n", None),
                      (rng.choice(places), "LOCATION"), ("\n", None)]
            parts += [(w + "\n", None) for w in furniture(rng, rng.randint(0, 3))]
        elif k < 0.75:                     # byline and credits
            by = rng.choice(["Von ", "By ", "Foto: ", "Text: ", "Autor: ", "Author: ", "Posted by ", "Ein Beitrag von "])
            parts += [(w + "\n", None) for w in furniture(rng, rng.randint(1, 4))]
            parts += [(by, None), (rng.choice(persons), "PERSON"), ("\n", None)]
            if rng.random() < 0.6:
                parts += [(rng.choice(["Quelle: ", "Source: ", "© ", "Ein Angebot von ", "A service by ", "Powered by "]), None), (org(), "ORG"), ("\n", None)]
            parts += [(rng.choice(["8 min read", "Lesezeit: 4 Minuten", "Stand: 29.10.2002", "Updated 3 days ago", "12.05.2021"]), None)]
        elif k < 0.9:                      # social and partner rows
            row = rng.sample(SOCIAL, rng.randint(2, 5))
            lead = rng.choice(["Folge uns auf ", "Follow us on ", "", "Externe Links (etwa zu ", "Find us on ", ""])
            parts.append((lead, None))
            for i, sname in enumerate(row):
                sname = sname.lower() if rng.random() < 0.1 else sname
                parts += [(sname, "ORG"), (rng.choice([", ", " ", "\n", " | ", " und ", " and "]) if i < len(row) - 1 else "", None)]
            parts += [("\n", None)] + [(w + "\n", None) for w in furniture(rng, rng.randint(1, 4))]
        else:                              # headline with a company, then furniture
            verb = rng.choice([" verstärkt ", " übernimmt ", " startet bei ", " wechselt zu ", " joins ", " acquires ", " launches at "])
            who = rng.choice(persons)
            parts += [(who.split()[-1] if rng.random() < 0.5 else who, "PERSON"), (verb, None), (org(), "ORG"), ("\n", None)]
            parts += [(w + "\n", None) for w in furniture(rng, rng.randint(1, 3))]
        out.append(build(parts))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("infile"); ap.add_argument("outfile")
    ap.add_argument("--lists", type=int, default=3000)
    ap.add_argument("--no-self-check", action="store_true", help="skip masking where the current tagger disagrees")
    a = ap.parse_args()
    rng = random.Random(11)
    rows = [json.loads(l) for l in open(a.infile, encoding="utf-8") if l.strip()]
    det = None
    if not a.no_self_check:
        from tarnkappe.detect import Detector
        det = Detector()
    examples, orgs, persons, places = [], set(), set(), set()
    kept = masked = 0
    for r in rows:
        text = r["text"]
        extra, mine = [], []
        if det:
            found = [e for e in det.detect(text, split_names=False) if e["type"] in ("PERSON", "ORGANIZATION", "LOCATION")]
            mine = [(e["start"], e["end"], e["type"]) for e in found]
            extra = [(e["start"], e["end"]) for e in found
                     if not any(a < e["end"] and b > e["start"] for a, b, _, _ in r["spans"])]
        spans = resolve(text, r["spans"], mine, extra)
        kept += sum(l != "?" for _, _, l in spans); masked += sum(l == "?" for _, _, l in spans)
        for a_, b_, lab, s in r["spans"]:
            name = text[a_:b_]
            if s >= 0.9 and 2 < len(name) < 40 and trusted(text, a_, b_, lab, s, mine) and name[0].isupper():
                {"ORG": orgs, "PERSON": persons, "LOCATION": places}.get(lab, set()).add(name)
        examples.append(wrap(rng, text, spans) if rng.random() < 0.5 else (text, spans))
    persons = [p for p in persons if " " in p]                # full names only
    examples += lists(rng, sorted(orgs), sorted(persons), sorted(places), a.lists)
    rng.shuffle(examples)
    # Numbers: the teacher rarely labels postcodes or phone numbers in encyclopedia text, so a bare number
    # outside a kept label is "unknown", not "no entity". Otherwise the tagger unlearns postcodes.
    number = re.compile(r"\+?\d[\d /().-]{1,}\d")
    for i, (text, spans) in enumerate(examples):
        extra = [(m.start(), m.end(), "?") for m in number.finditer(text)
                 if all(m.end() <= x or m.start() >= y for x, y, _ in spans)]
        examples[i] = (text, sorted(spans + extra))
    with open(a.outfile, "w", encoding="utf-8") as f:
        for text, spans in examples:
            f.write(json.dumps({"text": text, "spans": spans}, ensure_ascii=False) + "\n")
    print(f"{len(rows)} paragraphs ({kept} labels kept, {masked} masked), {a.lists} list examples, "
          f"{len(orgs)} organisation / {len(persons)} person / {len(places)} place names -> {a.outfile}")
