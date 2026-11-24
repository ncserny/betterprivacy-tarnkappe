# Results (v0.5.0)

Produced by `python bench.py --spacy` on 2 CPU cores, shipped int8 model + pattern layer, one thread.

How to read this:

- **Strict F1** needs the exact span and type. v0.3 lowers the decision thresholds on purpose, so strict precision
  and F1 are lower than in v0.2 while recall is higher.
- **PII caught** counts an annotated item as found if any detection overlaps it. This is the number that matters for redaction.
- Thresholds were tuned on development splits only. The EWT, GermEval, PUD, WNUT and BTC test numbers are clean.
- The model was retrained with the Broad Twitter Corpus (training sections) and casing augmentation; its test section is held out.
- The block below is v0.5.0 (same model files as v0.4.0, new rules): the v0.3 recipe plus page-like examples built from about 8,300 company names (see
  "v0.4: company detection"). It is run here with the default thresholds. Figures in the GLiNER
  comparison and in "First check on real scanned pages" further down were measured with the model before that retraining.
- The phone and postcode rules were written after looking at misses on the Presidio set and on `eval/challenge_v1.txt`,
  so those two are **not** clean evidence for phones and postcodes. v0.5 widened them again; see "v0.5: phones,
  postcodes and a larger company test" for a clean phone test.
- spaCy models were trained on other corpora with slightly different labelling rules; treat their scores as a reference point.

```
== real annotated text: strict F1 on person / organisation / location, and on person alone ==
ewt_test (English web text)
   ours      all F1 0.656 (P 0.591 R 0.737)   person F1 0.774 (P 0.716 R 0.842)   1.36 ms/sentence
/usr/local/lib/python3.13/site-packages/torch/jit/_script.py:1491: FutureWarning: `torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`.
  warnings.warn(
   spaCy sm  all F1 0.568 (P 0.558 R 0.578)   person F1 0.633 (P 0.650 R 0.617)   4.29 ms/sentence
   spaCy md  all F1 0.641 (P 0.619 R 0.664)   person F1 0.755 (P 0.751 R 0.759)   4.12 ms/sentence
wnut17_test (social media)
   ours      all F1 0.453 (P 0.381 R 0.557)   person F1 0.532 (P 0.440 R 0.671)   1.59 ms/sentence
   spaCy sm  all F1 0.248 (P 0.230 R 0.269)   person F1 0.274 (P 0.283 R 0.266)   3.61 ms/sentence
   spaCy md  all F1 0.276 (P 0.243 R 0.319)   person F1 0.356 (P 0.414 R 0.312)   3.78 ms/sentence
btc_test (tweets)
   ours      all F1 0.606 (P 0.594 R 0.619)   person F1 0.736 (P 0.733 R 0.738)   1.75 ms/sentence
   spaCy sm  all F1 0.260 (P 0.333 R 0.214)   person F1 0.221 (P 0.319 R 0.170)   3.69 ms/sentence
   spaCy md  all F1 0.309 (P 0.389 R 0.256)   person F1 0.295 (P 0.485 R 0.212)   3.97 ms/sentence
germeval_test (German Wikipedia/news)
   ours      all F1 0.690 (P 0.614 R 0.786)   person F1 0.768 (P 0.705 R 0.843)   1.61 ms/sentence
   spaCy sm  all F1 0.606 (P 0.545 R 0.682)   person F1 0.697 (P 0.672 R 0.723)   3.83 ms/sentence
   spaCy md  all F1 0.656 (P 0.601 R 0.722)   person F1 0.768 (P 0.753 R 0.783)   4.23 ms/sentence
pud_de (German news/Wikipedia)
   ours      all F1 0.709 (P 0.651 R 0.777)   person F1 0.750 (P 0.709 R 0.797)   1.67 ms/sentence
   spaCy sm  all F1 0.642 (P 0.599 R 0.691)   person F1 0.713 (P 0.710 R 0.715)   3.78 ms/sentence
   spaCy md  all F1 0.714 (P 0.675 R 0.757)   person F1 0.796 (P 0.795 R 0.797)   4.03 ms/sentence

== PII caught at all (any overlap, any label) ==
   presidio (templates unseen in training)       recall 0.964  precision 0.942
   our synthetic, new names + new phrasing       recall 0.987  precision 0.970
   ewt_test persons only                         recall 0.958
   germeval_test persons only                    recall 0.962
   wnut17_test persons only (social media)       recall 0.853
   btc_test persons only (tweets)                recall 0.832
      presidio CREDIT_CARD     recall 0.965
      presidio EMAIL_ADDRESS   recall 1.000
      presidio GPE             recall 0.948
      presidio IBAN_CODE       recall 1.000
      presidio IP_ADDRESS      recall 1.000
      presidio ORGANIZATION    recall 0.970
      presidio PERSON          recall 0.998
      presidio PHONE_NUMBER    recall 0.875
      presidio STREET_ADDRESS  recall 0.942
      presidio ZIP_CODE        recall 0.778

== release bar on challenge_v1.txt (87 records, 370 annotated items) ==
   PERSON        found  99.3% (133/134)   bar 95%   PASS
   LOCATION      found  98.9% (93/94)   bar 85%   PASS
   ORGANIZATION  found 100.0% (35/35)   bar 80%   PASS
   STREET        found  90.6% (29/32)   bar 90%   PASS
   PHONE         found 100.0% (29/29)   bar 95%   PASS
   ZIP           found  96.3% (26/27)   bar 90%   PASS
   EMAIL         found 100.0% (18/18)   bar 99%   PASS  (few examples)
   IBAN          found 100.0% (1/1)   bar 99%   PASS  (few examples)
   precision            93.1%            bar 85%   PASS
   all items found 98.4%;  by group: blog 100%, court-record 100%, data-broker 94%, de 99%, directory 98%, en 98%, forum 94%, news 100%, people-search 100%, search-engine 100%, social 100%

== first / last name split (synthetic, new names + new phrasing) ==
   F1 0.850 (P 0.858 R 0.842)

== latency, 1 CPU thread, 2000 texts, avg 17 tokens ==
p50 1.41 ms   p95 2.82 ms   p99 3.39 ms
throughput 656 texts/s, 10890 tokens/s per core
long document (1068 tokens, 4759 chars): 73.0 ms
```

## Comparison with GLiNER and GLiNER2-PII

Produced by `python scripts/compare_gliner.py --predict` on the same 2-core machine. 400 randomly drawn texts per test
set (fixed seed), the full challenge set, GLiNER threshold 0.5, labels given to GLiNER: person, organization, location,
street address, postal code, phone number, email, iban. Raw scores: `eval/gliner_scores.json`.

"Persons found" = share of annotated persons overlapped by a detection. "Correct" = share of person detections that
overlap an annotated person.

| Test set | Tarnkappe found / correct | Before retraining | gliner_multi_pii-v1 | GLiNER2-PII |
| --- | --- | --- | --- | --- |
| English web text (EWT) | 97.2% / 75.5% | 95.8% / 79.1% | 94.4% / 31.4% | 97.2% / 39.7% |
| German Wikipedia/news (GermEval) | 96.7% / 76.2% | 95.1% / 80.3% | 95.9% / 48.3% | 98.4% / 45.4% |
| German news (PUD) | 98.8% / 82.3% | 97.0% / 85.0% | 95.1% / 51.4% | 98.2% / 48.0% |
| Social media (WNUT 2017) | 84.9% / 45.4% | 56.3% / 54.7% | 92.4% / 41.6% | 95.8% / 39.6% |

| All PII, found / detections real | Tarnkappe | Before retraining | gliner_multi_pii-v1 | GLiNER2-PII |
| --- | --- | --- | --- | --- |
| Presidio, unseen templates (8 shared types) | 94.7% / 93.6% | 93.5% / 95.3% | 84.1% / 74.7% | 97.8% / 80.2% |
| Challenge set v1 | 96.8% / 92.7% | 97.8% / 93.8% | 94.6% / 92.4% | 97.0% / 91.2% |

| Speed, ms per text | Tarnkappe | gliner_multi_pii-v1 | GLiNER2-PII |
| --- | --- | --- | --- |
| All sets | 1.5 to 2.8 | 186 to 228 | 184 to 215 |

What this shows:

- On edited text Tarnkappe finds about as many persons as GLiNER2-PII (within 2 points either way) with roughly twice
  its precision, about 100 times faster.
- On social-media text the GLiNER models still find more persons (92 to 96% against 85% after retraining, 56% before),
  but only about 40% of what they flag as a person is one. They mark pronouns and common nouns ("I", "owner", "children", "er", "Menschen") and user handles.
  GLiNER2-PII's model card names this over-prediction of names as a known limitation.
- With GLiNER2-PII's native name labels (`full_name`, `first_name`, `last_name`) precision improves but recall drops
  (WNUT sample of 130 texts: 77% found, 44% correct at threshold 0.5; 40% found, 65% correct at 0.9). No setting gave both.
- GLiNER2-PII is clearly the better of the two, and strong on addresses, postcodes and organisations.
- Caveats: the corpora annotate only names, so a flagged pronoun counts as wrong even where it refers to a person; the
  WNUT figures here come from 400 texts and differ from the full-set figures above.

## First check on real scanned pages

27 snippets from 14 sites found by one BetterPrivacy scan (the repository owner's own), about half German and half English:
imprint, company-register pages, trade press, blogs, a slide transcript, a code-hosting thread. The owner corrected the
model's pre-labels for persons and companies by hand. The texts are private and not in this repository.

| | Tarnkappe | GLiNER2-PII |
| --- | --- | --- |
| Persons found (60 mentions) | 96.7% | 85.0% |
| Person detections that are correct | 85.1% | 75.8% |
| Companies found (101 mentions) | 79.2% | 89.1% |
| Company detections that are correct | 80.9% | 65.7% |

Read with care:

- One person's scan. A large share of the person mentions are the same two names, and a few company names account for
  many of the company mentions.
- The labels started from Tarnkappe's own guesses, so anything left unchanged counts for Tarnkappe and against GLiNER2-PII.
  A handful of leftover marks look like oversights.
- Almost all missed companies are brand names without a legal suffix, often in lower case.

**Scan-level company matching (`tarnkappe/scan.py`), same pages treated as one scan:**

| | Model alone | + names shared across the scan | + three profile names |
| --- | --- | --- | --- |
| Companies found | 79.2% | 84.2% | 87.1% |
| Companies labelled as companies | 68.3% | 77.2% | 80.2% |
| Company detections that are correct | 80.9% | 79.6% | 79.4% |
| Persons found | 96.7% | 96.7% | 96.7% |
| Person detections that are correct | 85.1% | 89.6% | 89.6% |

The shared names came from the pages themselves (for example a company found once with "GmbH" or "UG"); the three
profile names stand in for employers a person would list in their scan profile. These pages were also used while
building the feature, so the figures are indicative.

**Tried and not kept:** retraining with about 60 extra sentence patterns for suffix-less, lower-case and listed brand
names (invented names only). On these pages companies found rose from 79% to 82-86% depending on the threshold, but
company detections that are correct fell from 81% to 64-75%, more persons were labelled as companies (persons labelled
as persons fell from 95% to 88%), and persons found on tweets fell by about 5 points. The model in this repository is
therefore unchanged. Since these pages have now been used to judge a change, they count as development data; a
held-out set from other people's scans is still needed.

**Label review (19 November 2026):** the owner ruled on doubtful marks in seven of the 27 snippets (a project handle and
a job title are not persons; "Agentur" and similar descriptions are not companies). With the corrected labels the model
finds 96.4% of persons (54 of 56) and 79.6% of companies (98 mentions), or 83.7% with names shared across the scan.

## Held-out check: four other people's scans

38 snippets from the BetterPrivacy scans of four other people, who agreed to be scanned for this test: personal sites,
imprints, a company-register page, trade press, blogs, a people-search page; 23 German, 15 English. The model's answers
were saved before the owner labelled persons and companies, so the labels could not influence them. The texts are private.

| | Persons found | Person detections correct | Companies found | Company detections correct | ms per text |
| --- | --- | --- | --- | --- | --- |
| Tarnkappe | 93.4% | 71.8% | 67.1% | 63.8% | 5 |
| Tarnkappe, names shared across each scan | 93.4% | 72.7% | 69.6% | 65.6% | 5 |
| Tarnkappe, plus the scanned person's name | 100% | 74.7% | 69.6% | 66.7% | 5 |
| GLiNER multi PII v1 (threshold 0.5) | 90.2% | 98.2% | 75.9% | 70.4% | 285 |
| GLiNER2-PII (threshold 0.5) | 100% | 88.1% | 96.2% | 58.6% | 256 |

61 person mentions and 79 company mentions, so one miss moves a figure by 1.3 to 1.6 points.

Read with care:

- All four missed persons were the scanned person: the name on the first line of a profile page, the surname alone in a
  headline and in running text, and the surname inside a domain name. `profile_names` (see the README) was added after
  seeing these misses, so its 100% is not a clean test result.
- The labels started from Tarnkappe's own guesses. A first pass left several of its mistakes in place and put companies
  found at 80%; after the owner ruled on those cases (web sites, blogs and shops count as companies; handles and names
  inside e-mail addresses are not persons) it is 67%. The same bias may remain in smaller ways.
- Missed companies are almost all brand and site names without a legal suffix: blog.de, GitHub, LinkedIn,
  WordPress, Spreeblick.
- Wrong person detections are mostly menu words and company names. The tagger does not see line breaks, so a column of
  menu entries reads like a row of capitalised words.
- These snippets have now been used to find errors, so they count as development data from here on.

## Retraining for web pages

After the held-out check, four things changed:

1. **Line breaks are tokens.** Before, the tagger did not see them, so a menu column read like a row of capitalised words
   and detections ran across lines.
2. **Teacher-labelled text.** 11,000 German and English Wikipedia paragraphs (CC BY-SA 4.0), labelled by GLiNER2-PII
   (Apache 2.0). Only confident teacher labels are taught (organisation 0.8, person 0.7); unsure ones, bare numbers, and
   places where the previous tagger disagreed are left out of the loss. Half of the paragraphs are wrapped in menu and
   footer lines, and 3,000 list-shaped examples (register lists, speaker lists, bylines) are built from names the teacher
   was very sure about. Tweets are shown twice to keep their weight. `scripts/wiki_snippets.py`,
   `scripts/distil_label.py`, `scripts/build_distilled.py`.
3. **Rules.** "<Name> GmbH / UG / Ltd / plc ..." is a company by rule. Company names the tagger is very sure about on one
   page are matched on the other pages of the same scan.
4. **A `pages` threshold profile** (person 0.3, organisation 0.4) next to the default thresholds (0.05 and 0.1).

Both hand-labelled sets, each scan processed with `detect_scan` and the scanned person's name:

| | Persons found | Person detections correct | Companies found | Company detections correct |
| --- | --- | --- | --- | --- |
| Four other people's scans (38 snippets), before | 100% | 74.7% | 69.6% | 66.7% |
| after, `pages` profile | 100% | 86.3% | 82.3% | 86.2% |
| after, default thresholds | 100% | 82.9% | 88.6% | 63.2% |
| GLiNER2-PII (threshold 0.5), for reference | 100% | 88.1% | 96.2% | 58.6% |
| Owner's scan (27 snippets), before | 96.4% | 79.1% | 83.7% | 77.5% |
| after, `pages` profile | 96.4% | 91.7% | 83.7% | 77.8% |
| after, default thresholds | 96.4% | 84.6% | 89.8% | 60.5% |

Without scan context (each snippet alone, `pages` profile) the four-scan set gives persons 95.1% found and 85.1% correct,
companies 74.7% found and 83.1% correct. A snippet of about 400 characters takes 6 ms.

Read with care:

- **These are development figures.** The line-break change, the rules, the teacher cut-offs and the `pages` thresholds
  were all chosen while looking at these 65 snippets. A clean figure needs scans the model has never been tuned on.
- **Informal text got worse.** With the default thresholds, persons found fell from 88.2% to 79.8% on tweets (BTC test)
  and from 85.1% to 79.7% on WNUT; with the `pages` profile only about 55 to 62% of persons in tweets are found.
  Use the default thresholds for chat and social-media text. English web text and German news are unchanged
  (95.8% each).
- **Postcodes in unusual address layouts got worse** on the Presidio set (44% against 70%; 27 examples, mostly
  multi-line addresses from countries the postcode rules do not cover). German, Austrian, Swiss, UK and US postcodes
  next to a city or street are found by rule and are unaffected; the challenge set stays at 27 of 27.
- A first retraining without the number masking and without the extra weight on tweets was worse on both counts and was
  discarded.
- Still missed on pages: site and blog names with no context ("blog.de", "Spreeblick", "WordPress", "GitHub").

## Clean check: four more scans (v0.3.0)

23 snippets from the BetterPrivacy scans of four more people who agreed to be scanned (three gave usable pages, one gave a
single page, a fifth scan found nothing): doctor directories, personal blogs, a student newspaper, GitHub and two
people-search pages; 7 German, 16 English. Nothing was tuned on these before the model's answers were saved, and the owner
labelled persons and companies afterwards. `pages` profile, `detect_scan`, scanned person's name.

| | Persons found | Person detections correct | Companies found | Company detections correct |
| --- | --- | --- | --- | --- |
| Tarnkappe, answers saved before labelling | 100% (39/39) | 93.0% | 92.3% (12/13) | 34.3% |
| GLiNER2-PII, threshold 0.5 | 82.1% | 87.9% | 100% | 27.8% |
| GLiNER2-PII, threshold 0.8 | 74.4% | 96.7% | 84.6% | 47.4% |

Persons pass the release bar on clean data. Company detections do not: these pages mention few companies, and the model
flagged medical terms, a place, product names, user names, street names and page text as companies.

Two changes were then made after looking at these errors (so the figures below are development figures):

1. A company name the tagger is only very sure about is shared across a scan only when it is found on **two different
   pages**. Names with a legal form are still shared from one page.
2. A street name is never a company: such hits become STREET. A user name ("jdoe-dev", "@jdoe") is never a company.

| After the two changes (`pages` profile, `detect_scan`, person's name) | Persons found | Person detections correct | Companies found | Company detections correct |
| --- | --- | --- | --- | --- |
| Four more scans (23 snippets) | 100% | 93.0% | 84.6% | 45.8% |
| Four other people's scans (38 snippets) | 100% | 86.3% | 78.5% | 85.2% |
| Owner's scan (27 snippets) | 96.4% | 91.7% | 83.7% | 77.6% |

Company detections remain the weak point: on pages with few real companies most company hits are wrong. Persons, e-mail
addresses, phone numbers and addresses are the reliable part of v0.3.0.

## v0.5: phones, postcodes and a larger company test

v0.5 keeps the v0.4 model files and changes rules only.

**Phones.** The phone rule now accepts any 7 to 15 digit number, also with slashes, dots or brackets, when a phone
word stands right before or after it: "Tel.", "call me on", "contact", "customer care", "hotline", "Tél.", "Telefono",
"Teléfono", "reach the office at", "office", "mobile" and others. A digit string starting with 0 is no longer read as
a credit card (no card number starts with 0), so "00355…" is a phone number again.

Tested on 10,000 synthetic European phone sentences (40 countries, 20 formats, generated from libphonenumber metadata;
not in this repository). The cue words were chosen on the 8,000 training sentences only:

| Split | v0.4 | v0.5 | Text wrongly marked as a phone |
| --- | --- | --- | --- |
| test (1,000, never looked at) | 89.0% | 99.2% | 0 |
| validation (1,000) | | 99.5% | 0 |

The remaining misses are six-digit local numbers, which are left alone because too many other figures look the same.
`scripts/eval_phone_csv.py` runs this test on any CSV with the same columns. On all hand-labelled real pages the
phone and card detections are identical to v0.4.

**Postcodes.** New formats: Portuguese 1234-567, three-digit numbers after a postcode word or inside an address block,
a number on its own line in an address block, and "City 12345, DE"-style city-then-code. Years (1000–2099) are excluded.
On Presidio, postcodes found rose from 33% to 78% and phones from 63% to 88%; these were tuned on Presidio, so they are
not clean.

**Companies.** Three changes:

1. A context rule proposes names after "worked at", "joined", "Title @", "bei", "wechselt zu", "Gesellschafter des",
   "Agentur" and similar; the company check accepts them with a lower bar.
2. A company name accepted by the company check on two pages of one scan is matched on every page of that scan
   (before: only names the tagger scored 0.9 or more on two pages).
3. A one-word "person" with no title or first name in front becomes a company when the company lexicon mostly knows
   the word as a company ("Porsche", "Lufthansa", "Siemens"); surnames such as "Müller" stay persons.

The context rule was tuned on the 21 fresh snippets of v0.4, where it raised companies found from 62% to 86%. To see if
that holds, 57 new snippets from five consenting people's scans were labelled by hand (persons and companies; 102
companies), with predictions of both versions saved before labelling:

| 57 new snippets | Persons found | Person detections correct | Companies found | Company detections correct |
| --- | --- | --- | --- | --- |
| v0.4 | 99% | 85% | 66% | 79% |
| v0.5 with change 1 only (clean) | 99% | 85% | 67% | 79% |
| v0.5 as released (changes 2 and 3 tuned on these snippets) | 97% | 89% | 74% | 76% |

So the context rule does not carry over: the 86% was an effect of tuning on a small sample. Changes 2 and 3 add about
seven points on the same snippets, which are no longer clean; the next labelled batch will tell. On all 159
hand-labelled snippets together (all now development data), companies found is 69% and 78% of company detections are
correct.

What the remaining misses look like: brands the tagger never proposes (a software tool, a blogging platform, agency
and consultancy names, small startup names in a list), and brand names that look like surnames, tagged as persons. A larger change to the tagger is needed to reach the 80% bar, planned for v0.6.

## v0.4: company detection

v0.3 found most companies on real pages but flagged many other words as companies. v0.4 makes three changes:

1. **Company names in training.** 10,000 page-like examples (imprints, register lists, speaker lists, news sentences,
   signatures) built from about 8,300 real company names, with and without their legal form, and with job titles,
   medical fields, page buttons and technology words placed where names usually sit (labelled as not a company).
   Names: a CC0 sample of European and US companies from CompanyData.com's public datasets, and the STOXX Europe 600 and
   S&P 500 constituents as listed on Wikipedia. A fifth of each list is held out for testing.
2. **A second check on company detections** with the `pages` profile (`tarnkappe/orgcheck.py`): gradient-boosted trees
   (107 KB, ONNX) over 34 features: how often each word appears in company names in labelled text and how common it is,
   with capitalisation taken into account; form (length, capitals, digits, dots, "&"); surroundings (the word before it,
   whether it fills a short line, repeats on the page); the tagger's own confidence. Trained on development splits of
   the public corpora, 950 further Wikipedia paragraphs labelled by GLiNER2-PII, and the company examples. Not trained
   on hand-labelled pages.
3. **More legal forms** for the company rule: SL, SLU, SA, SRL, S.a r.l., SpA, LP, L.L.C., NV, AS, ASA, Sp. z o.o.,
   gUG in capitals, and capitalised variants.

**Fresh test.** 21 snippets from six scans of people who agreed to be scanned, taken from result pages not used before.
v0.3's answers were saved before the owner labelled persons and companies; v0.4's settings were chosen on the older
sets and run on these once. The snippets are private.

| `pages` profile, `detect_scan`, scanned person's name | Persons found | Companies found | Company detections correct |
| --- | --- | --- | --- |
| v0.3 | 100% | 62% (13/21) | 62% (13/21) |
| v0.4 tagger without the company check | 97%* | 62% | 72% (13/18) |
| **v0.4** | 97%* | 62% (13/21) | **87% (13/15)** |
| v0.4 after adding capitalised "GUG" as a legal form (a miss seen in this test) | 97%* | 67% (14/21) | 88% (14/16) |

\* 28 of 29; the one miss is a two-character mark that is most likely a labelling slip.

v0.3 made eight wrong company flags on this set (a medical field, software tool names, a user name, an abbreviation,
a greeting); v0.4 makes two (a software name and an abbreviation). The companies v0.4 still misses are mostly
well-known brands written without a legal form.

**Older hand-labelled sets** (used for tuning v0.3 and for choosing v0.4's settings):

| v0.4, same setup | Persons found | Person detections correct | Companies found | Company detections correct |
| --- | --- | --- | --- | --- |
| Owner's scan (27 snippets) | 96.4% | 85.9% | 84.7% | 80.8% |
| Four other people's scans (38 snippets) | 100% | 87.3% | 67.1% | 80.8% |
| Four more scans (23 snippets) | 97.4% | 97.4% | 84.6% | 73.3% |

**Held-out company names** (`scripts/eval_company_test.py`, 600 generated texts each, names never seen in training,
sentence patterns never used in training, `pages` profile; companies labelled as companies / of them without a legal form):

| Names held out from | v0.3 | v0.4 |
| --- | --- | --- |
| CompanyData.com sample (2,043 names) | 70.7% / 70.5% | 71.8% / 71.6% |
| STOXX 600 and S&P 500 plus a German register sample | 59.0% / 56.7% | 57.3% / 55.1% |
| German register sample only | 56.2% / 50.5% | 51.7% / 46.1% |

In these friendly sentences almost every company detection is correct for both versions, so the company check only
costs here: without it the same v0.4 tagger reaches 76.1% on the CompanyData names. Finding unfamiliar brand names is
what a model this small does not learn from more names; a fine-tuned transformer is the likely next step.

**What else moved.** Persons found with the default thresholds: tweets 79.8% → 83.2%, WNUT 79.7% → 85.3%, English
web text 95.8% (unchanged), German news 95.8% → 96.2%. Presidio: 96.1% → 94.9% caught (phone 75% → 63%, postcode
44% → 33%), 93.6% → 94.1% of detections real. Challenge set: 98.1% → 98.4%. Median latency 1.4 ms.

**Tried and not kept:** versions trained with a list of 500 German register entries and with the STOXX/S&P list only;
they were no better on real pages, and the licence of the register list was not checked.

## Transformer reference

spaCy `en_core_web_trf` (RoBERTa-base, 125M parameters, trained on OntoNotes, not on these corpora), same machine:

| Test set | All-type F1 | Person F1 | Person recall | ms per sentence |
| --- | --- | --- | --- | --- |
| English web text (EWT test) | 0.747 | 0.844 | 0.869 | 59.1 |
| Social media (WNUT 2017 test) | 0.555 | 0.646 | 0.711 | 73.3 |

Small multilingual encoders, forward pass only, one CPU core on this machine, PyTorch with int8 dynamic quantisation,
38-token text: distilbert-base-multilingual-cased 21.8 ms, Multilingual-MiniLM-L12-H384 16.8 ms, multilingual-e5-small
17.7 ms (fp32: 55, 35 and 33 ms). None has been fine-tuned for this task yet.

## v0.2 for comparison

| Measure | v0.2 | v0.3 rules and thresholds | v0.3 retrained |
| --- | --- | --- | --- |
| Persons caught, English web text (EWT test) | 90.0% | 95.3% | 95.5% |
| Persons caught, German Wikipedia/news (GermEval test) | 87.4% | 94.9% | 96.3% |
| Persons caught, social media (WNUT 2017 test) | not measured | 63.6% | 85.1% |
| Persons caught, tweets (BTC test) | not measured | 34.6% | 88.2% |
| Presidio, templates unseen in training: PII caught | 83.3% | 94.4% | 95.2% |
| Presidio: share of detections that are real PII | 97.5% | 94.8% | 92.7% |
| Challenge set v1: PII caught | 90.3% | 97.8% | 96.8% |

After the retraining for web pages (default thresholds): EWT 95.8%, GermEval 95.8%, WNUT 79.7%, BTC 79.8%,
Presidio 96.1% caught with 93.6% of detections real, challenge set 98.1% caught with 91.1% real.
| Median latency, short text | 1.4 ms | 1.6 ms | 1.6 ms |
