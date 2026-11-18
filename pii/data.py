"""Synthetic training data for the PII tagger (English + German).

Every example is (text, [(start, end, label), ...]) with character offsets.
Entity values and templates are split into disjoint train / test pools so the
evaluation measures generalisation to unseen names, companies, streets and
unseen sentence patterns, not memorisation.
"""
import os
import random
import re
import zlib

from faker import Faker

# fine-grained labels produced by the generator
LABELS = ["FIRST_NAME", "LAST_NAME", "EMAIL", "PHONE", "COMPANY", "STREET", "ZIP", "CITY", "COUNTRY"]
# what the model is trained to predict (first/last name is split afterwards by rule)
COARSE = {"FIRST_NAME": "PERSON", "LAST_NAME": "PERSON", "COMPANY": "ORG", "CITY": "LOCATION",
          "COUNTRY": "LOCATION", "STREET": "STREET", "ZIP": "ZIP", "EMAIL": "EMAIL", "PHONE": "PHONE"}
# extra locales used only as a source of more varied personal names
NAME_LOCALES = ["fr_FR", "es_ES", "it_IT", "tr_TR", "pl_PL", "nl_NL", "pt_BR", "sv_SE", "en_IN",
                "cs_CZ", "da_DK", "fi_FI", "hr_HR", "ro_RO", "en_NG"]


def coarsen(text, spans):
    """Fine spans -> model spans; 'First Last' separated by one space becomes one PERSON."""
    out = []
    for a, b, lab in sorted(spans):
        c = COARSE[lab]
        if (out and c == "PERSON" and out[-1][2] == "PERSON" and lab == "LAST_NAME"
                and out[-1][3] == "FIRST_NAME" and text[out[-1][1]:a] == " "):
            out[-1] = (out[-1][0], b, "PERSON", lab)
        elif out and c == "STREET" and out[-1][2] == "STREET" and text[out[-1][1]:a] in (" ", ", "):
            out[-1] = (out[-1][0], b, "STREET", lab)
        else:
            out.append((a, b, c, lab))
    return [(a, b, c) for a, b, c, _ in out]

LOCALES = {"en": ["en_US", "en_GB"], "de": ["de_DE", "de_AT"]}


def _is_test(value: str) -> bool:
    """Deterministic 80/20 split of entity values by hash."""
    return zlib.crc32(value.lower().encode()) % 5 == 0


# --------------------------------------------------------------------------
# Templates. {slot} placeholders; PII slots are labelled, the rest are not.
# --------------------------------------------------------------------------
EN = [
    "Hi {first}, thanks for your message.",
    "Hello {first} {last}, your order {num} has shipped.",
    "Dear Mr. {last}, we received your request on {weekday}.",
    "Dear Ms {last},",
    "My name is {first} {last} and I work at {company}.",
    "I'm {first} from {company}, nice to meet you.",
    "Please contact {first} {last} at {email} or call {phone}.",
    "You can reach me at {email}.",
    "My email is {email} and my phone number is {phone}.",
    "Call me on {phone} after 5pm.",
    "Phone: {phone}",
    "Email: {email}",
    "Tel. {phone} | {email}",
    "Ship it to {street}, {city} {zip}.",
    "Our office is located at {street}, {zip} {city}.",
    "I live at {street} in {city}.",
    "Billing address: {first} {last}, {street}, {city}, {zip}",
    "{first} {last}\n{company}\n{street}\n{city} {zip}",
    "Best regards,\n{first} {last}\n{company}",
    "Kind regards, {first}",
    "Thanks,\n{first} {last} | {company} | {phone}",
    "{first} {last} ({email}) joined the call.",
    "The invoice from {company} is overdue by {num} days.",
    "{company} announced a partnership with {company2} on {weekday}.",
    "We signed the contract with {company} last {weekday}.",
    "I spoke to {first} yesterday about the {product} rollout.",
    "{last}, {first} - employee id {num}",
    "Customer {first} {last} from {city} complained about {product}.",
    "Can you forward this to {first} {last}? Her address is {email}.",
    "Name: {first} {last}\nCompany: {company}\nPhone: {phone}\nEmail: {email}",
    "Name: {first}\nSurname: {last}\nCity: {city}\nZip: {zip}",
    "{first} moved from {city} to {city2} last year.",
    "hey its {first}, new number is {phone}",
    "Send the documents to {company}, {street}, {zip} {city}.",
    "According to {first} {last}, CEO of {company}, revenue grew {num} percent.",
    "Dr. {last} will see you on {weekday} at {num} o'clock.",
    "cc {first} {last} <{email}>",
    "From: {first} {last} <{email}>\nTo: {email2}",
    "{first}'s flight to {city} lands at {num}.",
    "I am writing on behalf of my client, {first} {last}, residing at {street}, {city}.",
    "Meeting with {last} and {last2} from {company} moved to {weekday}.",
    "If you have questions, contact our support team at {company}: {phone}.",
    "{first} {last}, {city}",
    "Please update my surname to {last}.",
    "The package for {first} {last} was returned to {company}.",
    "user {email} reset their password {num} times",
    "Mrs. {last} called from {phone} regarding her {product}.",
    "Our mailing address is {company}, {street}, {city}, {zip}.",
    # negatives: no PII at all
    "The {product} update will be released on {weekday}.",
    "Please restart the {product} server and check the logs.",
    "Total amount due: {amount}. Payment is due within {num} days.",
    "The meeting is on {weekday} in {month}.",
    "Thank you for your patience. We will get back to you shortly.",
    "Error {num}: connection timed out after {num} seconds.",
    "I love the new {product}, it works great with {product2}.",
    "Order {num} contains {num} items worth {amount}.",
    "Version {num}.{num} fixes the login bug.",
    "We ship to {country} and {country2} within {num} business days.",
    "Open the Settings page, then click Save.",
    "The quarterly report is attached. Let me know what you think.",
]

DE = [
    "Hallo {first}, danke für deine Nachricht.",
    "Guten Tag Herr {last}, Ihre Bestellung {num} wurde versendet.",
    "Sehr geehrte Frau {last},",
    "Sehr geehrter Herr Dr. {last}, wir haben Ihre Anfrage am {weekday} erhalten.",
    "Mein Name ist {first} {last} und ich arbeite bei {company}.",
    "Ich bin {first} von der {company}.",
    "Bitte kontaktieren Sie {first} {last} unter {email} oder telefonisch unter {phone}.",
    "Sie erreichen mich unter {email}.",
    "Meine E-Mail-Adresse lautet {email}, meine Telefonnummer ist {phone}.",
    "Rufen Sie mich bitte unter {phone} zurück.",
    "Tel.: {phone}",
    "E-Mail: {email}",
    "Telefon {phone} | {email}",
    "Bitte liefern Sie an {street}, {zip} {city}.",
    "Unser Büro befindet sich in der {street} in {zip} {city}.",
    "Ich wohne in der {street} in {city}.",
    "Rechnungsadresse: {first} {last}, {street}, {zip} {city}",
    "{first} {last}\n{company}\n{street}\n{zip} {city}",
    "Mit freundlichen Grüßen\n{first} {last}\n{company}",
    "Viele Grüße, {first}",
    "LG {first}",
    "Danke,\n{first} {last} | {company} | {phone}",
    "{first} {last} ({email}) hat am Termin teilgenommen.",
    "Die Rechnung der {company} ist seit {num} Tagen überfällig.",
    "Die {company} hat am {weekday} eine Partnerschaft mit {company2} bekannt gegeben.",
    "Wir haben den Vertrag mit {company} letzten {weekday} unterschrieben.",
    "Ich habe gestern mit {first} über die Einführung von {product} gesprochen.",
    "{last}, {first} - Personalnummer {num}",
    "Der Kunde {first} {last} aus {city} hat sich über {product} beschwert.",
    "Kannst du das an {first} {last} weiterleiten? Ihre Adresse ist {email}.",
    "Name: {first} {last}\nFirma: {company}\nTelefon: {phone}\nE-Mail: {email}",
    "Vorname: {first}\nNachname: {last}\nOrt: {city}\nPLZ: {zip}",
    "{first} ist letztes Jahr von {city} nach {city2} gezogen.",
    "hi hier ist {first}, meine neue nummer ist {phone}",
    "Senden Sie die Unterlagen an {company}, {street}, {zip} {city}.",
    "Laut {first} {last}, Geschäftsführer der {company}, stieg der Umsatz um {num} Prozent.",
    "Frau Dr. {last} erwartet Sie am {weekday} um {num} Uhr.",
    "Von: {first} {last} <{email}>\nAn: {email2}",
    "Ich schreibe im Namen meines Mandanten {first} {last}, wohnhaft in der {street}, {city}.",
    "Der Termin mit Herrn {last} und Frau {last2} von {company} wurde auf {weekday} verschoben.",
    "Bei Fragen wenden Sie sich an unseren Support bei {company}: {phone}.",
    "Bitte ändern Sie meinen Nachnamen in {last}.",
    "Das Paket für {first} {last} ging an {company} zurück.",
    "Benutzer {email} hat das Passwort {num} Mal zurückgesetzt",
    "Frau {last} hat von {phone} aus wegen ihres {product} angerufen.",
    "Unsere Postanschrift lautet {company}, {street}, {zip} {city}.",
    "Herr {last} wohnt jetzt in {city}.",
    # negatives
    "Das Update für {product} erscheint am {weekday}.",
    "Bitte starten Sie den {product} Server neu und prüfen Sie die Logs.",
    "Gesamtbetrag: {amount}. Zahlbar innerhalb von {num} Tagen.",
    "Die Besprechung findet am {weekday} im {month} statt.",
    "Vielen Dank für Ihre Geduld. Wir melden uns in Kürze bei Ihnen.",
    "Fehler {num}: Zeitüberschreitung nach {num} Sekunden.",
    "Die Bestellung {num} enthält {num} Artikel im Wert von {amount}.",
    "Version {num}.{num} behebt den Fehler bei der Anmeldung.",
    "Wir liefern innerhalb von {num} Werktagen nach {country} und {country2}.",
    "Öffnen Sie die Einstellungen und klicken Sie auf Speichern.",
    "Der Quartalsbericht ist angehängt. Sag mir, was du davon hältst.",
    "Die Rechnung und der Lieferschein liegen dem Paket bei.",
]

# short, varied frames: the same entity in many different contexts
EN += [
    "Regards, {first}", "Cheers, {first}", "Thanks {first}!", "Thank you, {first} {last}.", "Sincerely,\n{first} {last}",
    "{first} here.", "This is {first} {last} speaking.", "{first} said the {product} build is broken.",
    "Did {first} reply yet?", "Ask {first} or {first2}.", "{first} and {first2} are on vacation until {weekday}.",
    "I met {first} {last} at the conference in {city}.", "{first} {last} wrote:", "Posted by {first} {last} on {weekday}",
    "@{first} can you check this?", "Written by {first} {last}", "Author: {first} {last}", "Contact person: {first} {last}",
    "Owner: {last}", "Signed, {first} {last}, {company}", "Mr {last} is not available.", "Prof. {last} teaches in {city}.",
    "{first} works for {company}.", "She joined {company} in {month}.", "He left {company} after {num} years.",
    "{company} is hiring.", "{company} was founded in {city}.", "Our supplier, {company}, raised prices.",
    "I ordered it from {company}.", "Employer: {company}", "Vendor: {company}", "Thanks for choosing {company}!",
    "The team at {company} will contact you.", "{company} and {company2} merged.", "A spokesperson for {company} declined to comment.",
    "{first} {last} of {company} said so.", "Reviewed by {last} ({company}).",
    "Address: {street}, {city}", "Delivery to {street}.", "He lives in {city}.", "Greetings from {city}!",
    "They opened a store at {street} in {city}.", "{zip} {city}", "Postcode {zip}", "The event takes place in {city} on {weekday}.",
    "Mobile: {phone}", "Fax {phone}", "WhatsApp me at {phone}", "My number: {phone}", "text {phone} if you are late",
    "Write to {email}", "Contact: {email}", "{email} is my new address", "Reply to {email} by {weekday}.",
    "I bought {num} licenses of {product} for {amount}.", "See you on {weekday}!", "Happy to help.", "Sounds good, thanks.",
    "The {product} team shipped version {num}.", "Is {product} down again?", "No worries, take your time.",
    "Great news about the launch in {country}.", "Let's talk on {weekday} morning.", "Invoice {num} attached.",
]

DE += [
    "Gruß, {first}", "Beste Grüße\n{first} {last}", "Danke {first}!", "Vielen Dank, {first} {last}.", "Herzliche Grüße\n{first}",
    "Hier ist {first}.", "{first} {last} am Apparat.", "{first} sagt, der {product} Build ist kaputt.",
    "Hat {first} schon geantwortet?", "Frag {first} oder {first2}.", "{first} und {first2} sind bis {weekday} im Urlaub.",
    "Ich habe {first} {last} auf der Messe in {city} getroffen.", "{first} {last} schrieb:", "Verfasst von {first} {last} am {weekday}",
    "@{first} kannst du das prüfen?", "Autor: {first} {last}", "Ansprechpartner: {first} {last}", "Ansprechpartnerin: Frau {last}",
    "Inhaber: {first} {last}", "Geschäftsführer: {first} {last}", "Herr {last} ist nicht erreichbar.", "Prof. {last} lehrt in {city}.",
    "{first} arbeitet bei {company}.", "Sie ist seit {month} bei {company}.", "Er hat {company} nach {num} Jahren verlassen.",
    "{company} stellt ein.", "{company} wurde in {city} gegründet.", "Unser Lieferant, die {company}, hat die Preise erhöht.",
    "Ich habe es bei {company} bestellt.", "Arbeitgeber: {company}", "Lieferant: {company}", "Danke, dass Sie sich für {company} entschieden haben!",
    "Das Team von {company} meldet sich bei Ihnen.", "{company} und {company2} haben fusioniert.", "Ein Sprecher der {company} wollte sich nicht äußern.",
    "{first} {last} von {company} hat das bestätigt.", "Geprüft von {last} ({company}).", "Impressum\n{company}\n{street}\n{zip} {city}",
    "Adresse: {street}, {city}", "Lieferung an {street}.", "Er wohnt in {city}.", "Grüße aus {city}!",
    "Sie haben ein Geschäft in der {street} in {city} eröffnet.", "{zip} {city}", "PLZ {zip}", "Die Veranstaltung findet am {weekday} in {city} statt.",
    "Mobil: {phone}", "Fax {phone}", "Schreib mir per WhatsApp an {phone}", "Meine Nummer: {phone}", "ruf {phone} an wenn du später kommst",
    "Schreiben Sie an {email}", "Kontakt: {email}", "{email} ist meine neue Adresse", "Antwort bitte bis {weekday} an {email}.",
    "Ich habe {num} Lizenzen von {product} für {amount} gekauft.", "Bis {weekday}!", "Gerne.", "Klingt gut, danke.",
    "Das {product} Team hat Version {num} veröffentlicht.", "Ist {product} schon wieder down?", "Kein Problem, lass dir Zeit.",
    "Gute Nachrichten zum Start in {country}.", "Lass uns {weekday} früh telefonieren.", "Rechnung {num} im Anhang.",
]

PRODUCTS = ["Windows", "Excel", "Slack", "Outlook", "Photoshop", "Android", "Chrome", "Firefox",
            "Linux", "Kubernetes", "Postgres", "Jira", "Notion", "Teams", "Zoom", "Docker",
            "Python", "Salesforce", "WordPress", "iPhone", "Premium Plan", "Dashboard", "Router"]
WEEKDAYS = {"en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            "de": ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]}
MONTHS = {"en": ["January", "February", "March", "April", "May", "June", "July", "August",
                 "September", "October", "November", "December"],
          "de": ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August",
                 "September", "Oktober", "November", "Dezember"]}
COUNTRIES = {"en": ["Germany", "France", "Spain", "Canada", "Japan", "Brazil", "Italy", "Austria", "Poland",
                    "Turkey", "India", "Mexico", "Norway", "Ireland", "Australia", "Switzerland"],
             "de": ["Deutschland", "Frankreich", "Spanien", "Kanada", "Japan", "Italien",
                    "Österreich", "Schweden", "Polen", "Türkei", "Indien", "Schweiz",
                    "Norwegen", "Irland", "Belgien", "Dänemark"]}
CO_SUFFIX = {"en": ["Inc.", "Inc", "LLC", "Ltd", "Ltd.", "Corp.", "Group", "Holdings", "Solutions",
                    "Technologies", "Labs", "Consulting", "Partners", "& Co."],
             "de": ["GmbH", "AG", "GmbH & Co. KG", "KG", "UG", "SE", "e.V.", "OHG", "Gruppe",
                    "Systeme", "Technik", "Consulting", "Holding"]}
SYL = ["no", "va", "tec", "ri", "lo", "da", "ta", "flex", "ex", "on", "ix", "sol", "ar", "ve", "ko",
       "mi", "nex", "zen", "qua", "lum", "cor", "bit", "sys", "net", "op", "ti", "al", "pha", "be",
       "ra", "med", "fin", "log", "en", "er", "gy", "tra", "ns", "port", "data", "soft", "hub"]
MAIL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "gmx.de", "web.de", "t-online.de",
                "hotmail.com", "icloud.com", "proton.me", "posteo.de", "mail.com", "aol.com"]


def _tpl_is_test(t: str) -> bool:
    return zlib.crc32(t.encode()) % 6 == 0


class Generator:
    def __init__(self, split: str, seed: int = 0, templates: str = None):
        """split: which entity pool. templates: which template pool (defaults to split)."""
        assert split in ("train", "test")
        self.test = split == "test"
        tpl_test = (templates or split) == "test"
        self.rng = random.Random(seed + (1000 if self.test else 0))
        self.fakers = {}
        for lang, locs in LOCALES.items():
            self.fakers[lang] = []
            for loc in locs:
                f = Faker(loc)
                f.seed_instance(seed + (1000 if self.test else 0))
                self.fakers[lang].append(f)
        self.name_fakers = []
        for loc in NAME_LOCALES:
            f = Faker(loc)
            f.seed_instance(seed + (1000 if self.test else 0))
            self.name_fakers.append(f)
        self.templates = {
            "en": [t for t in EN if _tpl_is_test(t) == tpl_test],
            "de": [t for t in DE if _tpl_is_test(t) == tpl_test],
        }

    # ---- value generators (rejection-sample into the right pool) ----
    def _pool(self, fn):
        for _ in range(200):
            v = fn()
            if _is_test(v) == self.test:
                return v
        return v

    def _brand(self):
        r = self.rng
        return "".join(r.choice(SYL) for _ in range(r.randint(2, 3))).capitalize()

    def company(self, lang, f):
        r = self.rng
        k = r.random()
        if k < 0.35:
            base = f.last_name() if r.random() < 0.6 else f.last_name() + " & " + f.last_name()
            name = f"{base} {r.choice(CO_SUFFIX[lang])}"
        elif k < 0.75:
            name = self._brand()
            if r.random() < 0.6:
                name += " " + r.choice(CO_SUFFIX[lang])
        else:
            name = f.company()
        return name

    def phone(self, lang):
        r = self.rng
        d = lambda n: "".join(str(r.randint(0, 9)) for _ in range(n))
        if lang == "de":
            fmts = [f"+49 {d(2)} {d(7)}", f"+49 {d(3)} {d(3)} {d(4)}", f"0{d(3)} {d(7)}",
                    f"0{d(3)}/{d(6)}", f"0{d(2)}-{d(8)}", f"+49{d(10)}", f"01{d(2)} {d(8)}",
                    f"(0{d(2)}) {d(4)}-{d(4)}", f"0049 {d(3)} {d(7)}", f"+43 {d(1)} {d(7)}"]
        else:
            fmts = [f"({d(3)}) {d(3)}-{d(4)}", f"{d(3)}-{d(3)}-{d(4)}", f"+1 {d(3)} {d(3)} {d(4)}",
                    f"+1-{d(3)}-{d(3)}-{d(4)}", f"{d(3)}.{d(3)}.{d(4)}", f"+44 {d(2)} {d(4)} {d(4)}",
                    f"0{d(4)} {d(6)}", f"+44{d(10)}", f"{d(10)}", f"1-800-{d(3)}-{d(4)}"]
        if r.random() < 0.25:                       # other countries' formats
            return r.choice(self.name_fakers + self.fakers[lang]).phone_number()
        p = r.choice(fmts)
        if lang == "en" and r.random() < 0.08:
            p += r.choice(["x", " x", " ext. ", " ext "]) + d(r.randint(2, 5))
        return p

    def email(self, first=None, last=None, company=None):
        r = self.rng
        asc = lambda s: re.sub(r"[^a-z0-9]", "", s.lower().replace("ä", "ae").replace("ö", "oe")
                               .replace("ü", "ue").replace("ß", "ss"))
        f_, l_ = asc(first or self._brand()), asc(last or self._brand())
        local = r.choice([f"{f_}.{l_}", f"{f_}{l_}", f"{f_[:1]}.{l_}", f"{f_}_{l_}{r.randint(1, 99)}",
                          f"{l_}{r.randint(1, 999)}", f_, r.choice(["info", "kontakt", "hello", "support",
                                                                    "office", "sales"])])
        if company and r.random() < 0.4:
            dom = (asc(company.split()[0]) or "firma") + r.choice([".com", ".de", ".io", ".net", ".co.uk", ".org"])
        else:
            dom = r.choice(MAIL_DOMAINS)
        return f"{local}@{dom}"

    # ---- one example ----
    def _fill(self, tpl, lang):
        r = self.rng
        f = r.choice(self.fakers[lang])
        nf = r.choice(self.name_fakers) if r.random() < 0.3 else f
        af = r.choice(self.name_fakers) if r.random() < 0.25 else f     # addresses from other countries
        first = self._pool(nf.first_name)
        last = self._pool(nf.last_name)
        company = self._pool(lambda: self.company(lang, f))
        vals = {
            "first": (first, "FIRST_NAME"), "last": (last, "LAST_NAME"),
            "last2": (self._pool(nf.last_name), "LAST_NAME"),
            "first2": (self._pool(nf.first_name), "FIRST_NAME"),
            "company": (company, "COMPANY"),
            "company2": (self._pool(lambda: self.company(lang, f)), "COMPANY"),
            "email": (self.email(first, last, company) if r.random() < 0.6 else self.email(), "EMAIL"),
            "email2": (self.email(), "EMAIL"),
            "phone": (self.phone(lang), "PHONE"),
            "street": (self._pool(af.street_address), "STREET"),
            "city": (self._pool(af.city), "CITY"), "city2": (self._pool(af.city), "CITY"),
            "zip": (af.postcode(), "ZIP"),
            "product": (r.choice(PRODUCTS), None), "product2": (r.choice(PRODUCTS), None),
            "weekday": (r.choice(WEEKDAYS[lang]), None), "month": (r.choice(MONTHS[lang]), None),
            "country": (r.choice(COUNTRIES[lang]), "COUNTRY"), "country2": (r.choice(COUNTRIES[lang]), "COUNTRY"),
            "amount": (r.choice(["€", "$", "EUR ", "USD "]) + f"{r.randint(1, 9999)}.{r.randint(0, 99):02d}", None),
        }
        out, spans, pos = [], [], 0
        for part in re.split(r"(\{\w+\})", tpl):
            if part.startswith("{") and part.endswith("}"):
                key = part[1:-1]
                if key == "num":
                    v, lab = str(r.randint(1, 99999) if r.random() < 0.5 else r.randint(1, 30)), None
                else:
                    v, lab = vals[key]
                v = v.replace("\n", ", ")
                if lab:
                    spans.append((pos, pos + len(v), lab))
            else:
                v = part
            out.append(v)
            pos += len(v)
        return "".join(out), spans

    def example(self):
        r = self.rng
        lang = r.choice(["en", "de"])
        text, spans = "", []
        for i in range(r.choice([1, 1, 2, 2, 3])):
            t, s = self._fill(r.choice(self.templates[lang]), lang)
            if i:
                text += r.choice([" ", " ", "\n", "\n\n"])
            off = len(text)
            text += t
            spans += [(a + off, b + off, l) for a, b, l in s]
        k = r.random()
        if k < 0.08:      # sloppy chat style
            text = text.lower()
        elif k < 0.10:
            text = text.upper()
        return text, spans


NATIONS = ["German", "French", "Italian", "Polish", "Turkish", "Indian", "Brazilian", "Swedish", "Dutch", "Irish"]
PRESIDIO_TPL = os.path.join(os.path.dirname(__file__), "..", "data", "presidio-research", "templates.txt")


def make_presidio(n, seed=0, parity=0):
    """Render Microsoft Presidio's sentence templates (MIT licence) with our own values.
    Only templates whose index % 2 == parity are used; the others stay unseen for testing."""
    g = Generator("train", seed + 77)
    r = g.rng
    tpls = [t.rstrip("\n") for i, t in enumerate(open(PRESIDIO_TPL, encoding="utf-8")) if i % 2 == parity and t.strip()]
    en = g.fakers["en"][0]

    def value(key):
        f = r.choice(g.fakers["en"])
        nf = r.choice(g.name_fakers) if r.random() < 0.4 else f
        if key in ("person", "name", "name_female", "name_male"):
            return [(g._pool(nf.first_name), "FIRST_NAME"), (" ", None), (g._pool(nf.last_name), "LAST_NAME")]
        if key.startswith("first_name"):
            return [(g._pool(nf.first_name), "FIRST_NAME")]
        if key.startswith("last_name"):
            return [(g._pool(nf.last_name), "LAST_NAME")]
        if key == "organization":
            return [(g._pool(lambda: g.company("en", f)), "COMPANY")]
        if key == "city":
            return [(g._pool(nf.city), "CITY")]
        if key == "country":
            return [(r.choice(COUNTRIES["en"]), "COUNTRY")]
        if key == "address":
            sep = r.choice([", ", "\n", ", ", " "])
            parts = [(g._pool(nf.street_address).replace("\n", ", "), "STREET"), (sep, None)]
            city, z = (g._pool(nf.city), "CITY"), (nf.postcode(), "ZIP")
            parts += [z, (" ", None), city] if r.random() < 0.5 else [city, (r.choice([" ", ", "]), None), z]
            if r.random() < 0.3:
                parts += [(sep, None), (r.choice(COUNTRIES["en"]), "COUNTRY")]
            return parts
        if key == "street_name":
            return [(g._pool(nf.street_name), "STREET")]
        if key == "building_number":
            return [(en.building_number(), "STREET")]
        if key == "secondary_address":
            return [(en.secondary_address(), "STREET")]
        if key in ("postcode", "zipcode"):
            return [(nf.postcode(), "ZIP")]
        if key == "phone_number":
            return [(g.phone("en"), "PHONE")]
        if key == "email":
            return [(g.email(), "EMAIL")]
        if key == "age":
            return [(str(r.randint(18, 90)), None)]
        if key in ("nationality", "nation_woman", "nation_plural"):
            return [(r.choice(NATIONS), None)]
        if key == "ip_address":
            return [(en.ipv4(), None)]
        if key == "us_driver_license":
            return [(en.bothify("?#######").upper(), None)]
        if key in ("date_time", "date_of_birth"):
            return [(str(en.date()), None)]
        try:
            return [(str(en.format(key)), None)]
        except Exception:
            return [(key, None)]

    out = []
    for _ in range(n):
        text, spans = "", []
        for part in re.split(r"(\{\{\w+\}\})", r.choice(tpls)):
            pieces = value(part[2:-2]) if part.startswith("{{") else [(part, None)]
            for v, lab in pieces:
                if lab:
                    spans.append((len(text), len(text) + len(v), lab))
                text += v
        if r.random() < 0.06:
            text = text.lower()
        out.append((text, coarsen(text, spans)))
    return out


def make(split, n, seed=0, templates=None, fine=False):
    """fine=True keeps FIRST_NAME / LAST_NAME / COMPANY / CITY labels (for scoring the API output)."""
    g = Generator(split, seed, templates)
    ex = [g.example() for _ in range(n)]
    return ex if fine else [(t, coarsen(t, s)) for t, s in ex]


if __name__ == "__main__":
    for t, s in make("train", 5):
        print(repr(t), [(t[a:b], l) for a, b, l in s])
