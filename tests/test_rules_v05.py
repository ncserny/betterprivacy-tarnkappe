from tarnkappe import Detector, detect_scan

det = Detector()


def types(text, **kw):
    return {(e["type"], e["text"]) for e in det.detect(text, **kw)}


def test_phone_after_cue_words():
    assert ("PHONE", "212769108") in types("Tél. : 212769108 — service client.")
    assert ("PHONE", "060/178/732") in types("Contact: 060/178/732.")


def test_number_starting_with_zero_is_not_a_card():
    t = types("Order REF-BXSWZM; contact 00355676109737; date 2026-11-24.")
    assert not any(k == "CREDIT_CARD" for k, _ in t)
    assert ("PHONE", "00355676109737") in t


def test_dates_after_cue_are_not_phones():
    assert not any(k == "PHONE" for k, _ in types("Contact: 12/05/2023"))


def test_company_name_shared_after_two_pages():
    pages = ["Victory Park Capital and Cherry Ventures invested.", "Cherry Ventures led the round.", "Thanks to cherry ventures."]
    res = detect_scan(det, pages, profile="pages")
    assert any(e["type"] == "ORGANIZATION" for e in res[2])
