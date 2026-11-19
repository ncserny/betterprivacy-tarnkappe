# Results (v0.3 development, model retrained for web pages)

Produced by `python bench.py --spacy` on 2 CPU cores, shipped int8 model + pattern layer, one thread.

How to read this:

- **Strict F1** needs the exact span and type. v0.3 lowers the decision thresholds on purpose, so strict precision
  and F1 are lower than in v0.2 while recall is higher.
- **PII caught** counts an annotated item as found if any detection overlaps it. This is the number that matters for redaction.
- Thresholds were tuned on development splits only. The EWT, GermEval, PUD, WNUT and BTC test numbers are clean.
- The model was retrained with the Broad Twitter Corpus (training sections) and casing augmentation; its test section is held out.
- The block below is the current model: it reads line breaks and was also trained on 11,000 Wikipedia paragraphs labelled
  by GLiNER2-PII (see "Retraining for web pages"). It is run here with the default thresholds. Figures in the GLiNER
  comparison and in "First check on real scanned pages" further down were measured with the model before that retraining.
- The phone and postcode rules were written after looking at misses on the Presidio set and on `eval/challenge_v1.txt`,
  so those two are **not** clean evidence for phones and postcodes.
- spaCy models were trained on other corpora with slightly different labelling rules; treat their scores as a reference point.

```
== real annotated text: strict F1 on person / organisation / location, and on person alone ==
ewt_test (English web text)
   ours      all F1 0.654 (P 0.595 R 0.727)   person F1 0.752 (P 0.709 R 0.802)   1.32 ms/sentence
   spaCy sm  all F1 0.568 (P 0.558 R 0.578)   person F1 0.633 (P 0.650 R 0.617)   4.36 ms/sentence
   spaCy md  all F1 0.641 (P 0.619 R 0.664)   person F1 0.755 (P 0.751 R 0.759)   4.50 ms/sentence
wnut17_test (social media)
   ours      all F1 0.461 (P 0.390 R 0.564)   person F1 0.527 (P 0.437 R 0.664)   1.59 ms/sentence
   spaCy sm  all F1 0.248 (P 0.230 R 0.269)   person F1 0.274 (P 0.283 R 0.266)   4.47 ms/sentence
   spaCy md  all F1 0.276 (P 0.243 R 0.319)   person F1 0.356 (P 0.414 R 0.312)   4.07 ms/sentence
btc_test (tweets)
   ours      all F1 0.594 (P 0.586 R 0.601)   person F1 0.707 (P 0.712 R 0.701)   1.99 ms/sentence
   spaCy sm  all F1 0.260 (P 0.333 R 0.214)   person F1 0.221 (P 0.319 R 0.170)   4.12 ms/sentence
   spaCy md  all F1 0.309 (P 0.389 R 0.256)   person F1 0.295 (P 0.485 R 0.212)   4.17 ms/sentence
germeval_test (German Wikipedia/news)
   ours      all F1 0.704 (P 0.630 R 0.798)   person F1 0.774 (P 0.714 R 0.845)   1.72 ms/sentence
   spaCy sm  all F1 0.606 (P 0.545 R 0.682)   person F1 0.697 (P 0.672 R 0.723)   4.43 ms/sentence
   spaCy md  all F1 0.656 (P 0.601 R 0.722)   person F1 0.768 (P 0.753 R 0.783)   5.24 ms/sentence
pud_de (German news/Wikipedia)
   ours      all F1 0.702 (P 0.646 R 0.769)   person F1 0.745 (P 0.705 R 0.789)   1.79 ms/sentence
   spaCy sm  all F1 0.642 (P 0.599 R 0.691)   person F1 0.713 (P 0.710 R 0.715)   4.09 ms/sentence
   spaCy md  all F1 0.714 (P 0.675 R 0.757)   person F1 0.796 (P 0.795 R 0.797)   5.01 ms/sentence

== PII caught at all (any overlap, any label) ==
   presidio (templates unseen in training)       recall 0.961  precision 0.936
   our synthetic, new names + new phrasing       recall 0.985  precision 0.965
   ewt_test persons only                         recall 0.958
   germeval_test persons only                    recall 0.958
   wnut17_test persons only (social media)       recall 0.797
   btc_test persons only (tweets)                recall 0.798
      presidio CREDIT_CARD     recall 0.965
      presidio EMAIL_ADDRESS   recall 1.000
      presidio GPE             recall 0.943
      presidio IBAN_CODE       recall 1.000
      presidio IP_ADDRESS      recall 1.000
      presidio ORGANIZATION    recall 0.963
      presidio PERSON          recall 0.998
      presidio PHONE_NUMBER    recall 0.750
      presidio STREET_ADDRESS  recall 0.982
      presidio ZIP_CODE        recall 0.444

== release bar on challenge_v1.txt (87 records, 370 annotated items) ==
   PERSON        found  99.3% (133/134)   bar 95%   PASS
   LOCATION      found  97.9% (92/94)   bar 85%   PASS
   ORGANIZATION  found  97.1% (34/35)   bar 80%   PASS
   STREET        found  90.6% (29/32)   bar 90%   PASS
   PHONE         found 100.0% (29/29)   bar 95%   PASS
   ZIP           found 100.0% (27/27)   bar 90%   PASS
   EMAIL         found 100.0% (18/18)   bar 99%   PASS  (few examples)
   IBAN          found 100.0% (1/1)   bar 99%   PASS  (few examples)
   precision            91.1%            bar 85%   PASS
   all items found 98.1%;  by group: blog 100%, court-record 100%, data-broker 98%, de 99%, directory 95%, en 97%, forum 94%, news 100%, people-search 98%, search-engine 100%, social 100%

== first / last name split (synthetic, new names + new phrasing) ==
   F1 0.871 (P 0.852 R 0.890)

== latency, 1 CPU thread, 2000 texts, avg 17 tokens ==
p50 1.65 ms   p95 3.30 ms   p99 3.93 ms
throughput 565 texts/s, 9372 tokens/s per core
long document (1068 tokens, 4759 chars): 84.6 ms
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
