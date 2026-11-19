# Results (v0.3 development, model retrained with tweets)

Produced by `python bench.py --spacy` on 2 CPU cores, shipped int8 model + pattern layer, one thread.

How to read this:

- **Strict F1** needs the exact span and type. v0.3 lowers the decision thresholds on purpose, so strict precision
  and F1 are lower than in v0.2 while recall is higher.
- **PII caught** counts an annotated item as found if any detection overlaps it. This is the number that matters for redaction.
- Thresholds were tuned on development splits only. The EWT, GermEval, PUD, WNUT and BTC test numbers are clean.
- The model was retrained with the Broad Twitter Corpus (training sections) and casing augmentation; its test section is held out.
- The phone and postcode rules were written after looking at misses on the Presidio set and on `eval/challenge_v1.txt`,
  so those two are **not** clean evidence for phones and postcodes.
- spaCy models were trained on other corpora with slightly different labelling rules; treat their scores as a reference point.

```
== real annotated text: strict F1 on person / organisation / location, and on person alone ==
ewt_test (English web text)
   ours      all F1 0.637 (P 0.587 R 0.698)   person F1 0.728 (P 0.669 R 0.800)   1.42 ms/sentence
   spaCy sm  all F1 0.568 (P 0.558 R 0.578)   person F1 0.633 (P 0.650 R 0.617)   7.09 ms/sentence
   spaCy md  all F1 0.641 (P 0.619 R 0.664)   person F1 0.755 (P 0.751 R 0.759)   5.21 ms/sentence
wnut17_test (social media)
   ours      all F1 0.438 (P 0.367 R 0.541)   person F1 0.503 (P 0.399 R 0.681)   1.84 ms/sentence
   spaCy sm  all F1 0.248 (P 0.230 R 0.269)   person F1 0.274 (P 0.283 R 0.266)   4.28 ms/sentence
   spaCy md  all F1 0.276 (P 0.243 R 0.319)   person F1 0.356 (P 0.414 R 0.312)   4.50 ms/sentence
btc_test (tweets)
   ours      all F1 0.596 (P 0.578 R 0.616)   person F1 0.737 (P 0.701 R 0.778)   2.23 ms/sentence
   spaCy sm  all F1 0.260 (P 0.333 R 0.214)   person F1 0.221 (P 0.319 R 0.170)   4.77 ms/sentence
   spaCy md  all F1 0.309 (P 0.389 R 0.256)   person F1 0.295 (P 0.485 R 0.212)   5.03 ms/sentence
germeval_test (German Wikipedia/news)
   ours      all F1 0.702 (P 0.638 R 0.780)   person F1 0.753 (P 0.681 R 0.843)   1.89 ms/sentence
   spaCy sm  all F1 0.606 (P 0.545 R 0.682)   person F1 0.697 (P 0.672 R 0.723)   4.68 ms/sentence
   spaCy md  all F1 0.656 (P 0.601 R 0.722)   person F1 0.768 (P 0.753 R 0.783)   5.47 ms/sentence
pud_de (German news/Wikipedia)
   ours      all F1 0.700 (P 0.653 R 0.755)   person F1 0.728 (P 0.682 R 0.780)   1.89 ms/sentence
   spaCy sm  all F1 0.642 (P 0.599 R 0.691)   person F1 0.713 (P 0.710 R 0.715)   4.92 ms/sentence
   spaCy md  all F1 0.714 (P 0.675 R 0.757)   person F1 0.796 (P 0.795 R 0.797)   4.93 ms/sentence

== PII caught at all (any overlap, any label) ==
   presidio (templates unseen in training)       recall 0.952  precision 0.927
   our synthetic, new names + new phrasing       recall 0.983  precision 0.966
   ewt_test persons only                         recall 0.955
   germeval_test persons only                    recall 0.963
   wnut17_test persons only (social media)       recall 0.851
   btc_test persons only (tweets)                recall 0.882
      presidio CREDIT_CARD     recall 0.965
      presidio EMAIL_ADDRESS   recall 1.000
      presidio GPE             recall 0.962
      presidio IBAN_CODE       recall 1.000
      presidio IP_ADDRESS      recall 1.000
      presidio ORGANIZATION    recall 0.911
      presidio PERSON          recall 0.993
      presidio PHONE_NUMBER    recall 0.750
      presidio STREET_ADDRESS  recall 0.935
      presidio ZIP_CODE        recall 0.704

== release bar on challenge_v1.txt (87 records, 370 annotated items) ==
   PERSON        found  97.8% (131/134)   bar 95%   PASS
   LOCATION      found  96.8% (91/94)   bar 85%   PASS
   ORGANIZATION  found  94.3% (33/35)   bar 80%   PASS
   STREET        found  90.6% (29/32)   bar 90%   PASS
   PHONE         found 100.0% (29/29)   bar 95%   PASS
   ZIP           found  96.3% (26/27)   bar 90%   PASS
   EMAIL         found 100.0% (18/18)   bar 99%   PASS  (few examples)
   IBAN          found 100.0% (1/1)   bar 99%   PASS  (few examples)
   precision            92.3%            bar 85%   PASS
   all items found 96.8%;  by group: blog 100%, court-record 100%, data-broker 92%, de 98%, directory 95%, en 96%, forum 94%, news 97%, people-search 98%, search-engine 97%, social 100%

== first / last name split (synthetic, new names + new phrasing) ==
   F1 0.836 (P 0.794 R 0.882)

== latency, 1 CPU thread, 2000 texts, avg 16 tokens ==
p50 1.61 ms   p95 3.31 ms   p99 4.43 ms
throughput 565 texts/s, 9045 tokens/s per core
long document (964 tokens, 4763 chars): 94.2 ms
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
| Median latency, short text | 1.4 ms | 1.6 ms | 1.6 ms |
