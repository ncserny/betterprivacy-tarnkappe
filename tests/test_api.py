import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

from tarnkappe.api import app
from tarnkappe.detect import regex_spans

client = TestClient(app)


def types(text):
    return {(e["type"], e["text"]) for e in client.post("/v1/detect", json={"text": text}).json()["entities"]}


def test_health():
    assert client.get("/healthz").json() == {"status": "ok"}


def test_structured_identifiers():
    got = types("Mail anna@example.org, IBAN DE89 3704 0044 0532 0130 00, card 4111 1111 1111 1111, call +49 30 1234567")
    assert ("EMAIL", "anna@example.org") in got
    assert ("IBAN", "DE89 3704 0044 0532 0130 00") in got
    assert ("CREDIT_CARD", "4111 1111 1111 1111") in got
    assert ("PHONE", "+49 30 1234567") in got


def test_checksums_reject_fakes():
    kinds = {s["type"] for s in regex_spans("IBAN DE89 3704 0044 0532 0130 01 card 4111 1111 1111 1112")}
    assert "IBAN" not in kinds and "CREDIT_CARD" not in kinds


def test_redact_and_offsets():
    text = "Hello Laura Becker, your email is laura.becker@gmail.com."
    r = client.post("/v1/detect", json={"text": text, "redact": True}).json()
    assert r["has_pii"]
    for e in r["entities"]:
        assert text[e["start"]:e["end"]] == e["text"]
    kinds = {e["type"] for e in r["entities"]}
    assert "FIRST_NAME" in kinds and "LAST_NAME" in kinds
    assert "laura.becker@gmail.com" not in r["redacted"]
    assert "[EMAIL]" in r["redacted"]


def test_no_pii():
    assert client.post("/v1/detect", json={"text": "Please restart the server and check the logs."}).json()["has_pii"] is False


def test_batch_and_empty():
    r = client.post("/v1/detect/batch", json={"texts": ["", "Email: bob@web.de"]}).json()
    assert r[0]["entities"] == [] and r[1]["entities"][0]["type"] == "EMAIL"


def test_national_phone_formats():
    got = types("Ruf an: 0171 3920045 oder 030 23125 477. US office (415) 555-0132.")
    for number in ("0171 3920045", "030 23125 477", "(415) 555-0132"):
        assert ("PHONE", number) in got


def test_postcodes_need_context():
    got = types("Lieferung an Hauptstraße 12, 50667 Köln. Or 221 Baker Street, London NW1 6XE.")
    assert ("ZIP", "50667") in got and ("ZIP", "NW1 6XE") in got
    none = types("In 2019 we sold 12000 units, invoice 45402 from 12.03.2024 at 14:30.")
    assert not any(t == "ZIP" for t, _ in none)


def test_min_score_overrides_tuned_thresholds():
    text = "Posted by Aylin Demir on Tuesday."
    strict = client.post("/v1/detect", json={"text": text, "min_score": 0.999999}).json()["entities"]
    default = client.post("/v1/detect", json={"text": text}).json()["entities"]
    assert len(default) >= 2 and len(strict) <= len(default)


def test_scan_shares_company_names_between_pages():
    from tarnkappe.detect import Detector
    from tarnkappe.scan import bare_name, detect_scan
    assert bare_name("Nordwind Logistik GmbH & Co. KG") == "Nordwind Logistik"
    assert bare_name("Agentur GmbH") is None and bare_name("Nordwind") is None
    pages = ["Impressum\nflowmint UG (haftungsbeschränkt)\nHauptstraße 12, 50667 Köln",
             "the flowmint-team and flowmint's founders met in Köln. kontakt@flowmint.de"]
    res = detect_scan(Detector(), pages, profile_companies=["Pixelnest"])
    second = [(e["type"], e["text"]) for e in res[1]]
    assert second.count(("ORGANIZATION", "flowmint")) == 2              # both mentions, with or without the list
    assert ("EMAIL", "kontakt@flowmint.de") in second                   # pattern hits are not overwritten
    alone = [(e["type"], e["text"]) for e in Detector().detect("ask pixelnest about it", known_companies=["Pixelnest"])]
    assert ("ORGANIZATION", "pixelnest") in alone


def test_scan_finds_the_scanned_persons_name():
    from tarnkappe.detect import Detector
    from tarnkappe.scan import detect_scan, person_spans
    page = "Beispielova verstärkt Nordwind\nmehr auf beispielova.berlin, @beispielova oder jana@beispielova.berlin"
    res = detect_scan(Detector(), [page], profile_names=["Jana Beispielova"])[0]
    names = [(e["text"], e["start"]) for e in res if e["type"] in ("FIRST_NAME", "LAST_NAME")]
    assert ("Beispielova", 0) in names                                   # surname alone in a headline
    assert ("beispielova", page.index("beispielova.berlin")) in names    # inside a domain name
    listed = [e["start"] for e in person_spans(page, ["Jana Beispielova"])]
    assert all(a == 0 or page[a - 1] != "@" for a in listed)             # the list never matches a handle or e-mail
    assert any(e["type"] == "EMAIL" and e["text"] == "jana@beispielova.berlin" for e in res)
    assert [e["type"] for e in person_spans("JANA BEISPIELOVA", ["Jana Beispielova"])] == ["FIRST_NAME", "LAST_NAME"]
    assert [e["type"] for e in person_spans("Jana Beispielova", ["Jana Beispielova"], split_names=False)] == ["PERSON"]


def test_pages_profile_and_legal_form_rule():
    from tarnkappe.detect import Detector
    det = Detector()
    page = "Startseite\nÜber uns\nKontakt\nNordwind Logistik GmbH & Co. KG\nGeschäftsführerin: Jana Beispielova\nImpressum"
    for kw in ({}, {"profile": "pages"}):
        got = [(e["type"], e["text"]) for e in det.detect(page, **kw)]
        assert ("ORGANIZATION", "Nordwind Logistik GmbH & Co. KG") in got          # a company by rule, whatever the thresholds
        assert not any("\n" in t for _, t in got)                                  # no entity runs across a line break
    assert len(det.detect(page, profile="pages")) <= len(det.detect(page))
    r = client.post("/v1/detect", json={"text": page, "profile": "pages"})
    assert r.status_code == 200
    assert client.post("/v1/detect", json={"text": page, "profile": "nope"}).status_code == 422
    try:
        det.detect(page, profile="nope"); assert False
    except ValueError:
        pass


def test_streets_and_user_names_are_never_companies():
    from tarnkappe.detect import Detector
    det = Detector()
    for text in ("Ordination\nObere Feldzeile 12/3\n1070 Wien", "Jana Beispielova\njbeispiel-dev\nFollow", "Twitter: @jbeispiel"):
        for kw in ({}, {"profile": "pages"}):
            orgs = [e["text"] for e in det.detect(text, **kw) if e["type"] == "ORGANIZATION"]
            assert not any(("Feldzeile" in o) or ("jbeispiel" in o) for o in orgs), (text, orgs)


def test_confident_names_need_two_pages():
    from tarnkappe.scan import harvest
    ents = [{"type": "ORGANIZATION", "text": "Nordwind", "start": 0, "end": 8, "score": 0.99, "source": "model"}]
    sure = set()
    assert harvest("Nordwind", ents, sure) == set() and sure == {"Nordwind"}   # collected, not shared on its own


def test_company_check_runs_with_pages_profile():
    from tarnkappe.detect import Detector
    det = Detector()
    assert det.orgcheck.ready and det.org_cut("pages") is not None and det.org_cut(None) is None
    text = "Impressum\nNordwind Logistik GmbH\nGeschäftsführerin: Jana Beispielova"
    got = [(e["type"], e["text"]) for e in det.detect(text, profile="pages")]
    assert ("ORGANIZATION", "Nordwind Logistik GmbH") in got            # legal-form companies always pass the check
    p = det.orgcheck.probs(text, [e for e in det.detect(text) if e["type"] == "ORGANIZATION"])
    assert all(0.0 <= x <= 1.0 for x in p)
