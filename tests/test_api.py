import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

from api import app
from pii.detect import regex_spans

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
