# Results (v0.3 development)

Produced by `python bench.py --spacy` on 2 CPU cores, shipped int8 model + pattern layer, one thread.

How to read this:

- **Strict F1** needs the exact span and type. v0.3 lowers the decision thresholds on purpose, so strict precision
  and F1 are lower than in v0.2 while recall is higher.
- **PII caught** counts an annotated item as found if any detection overlaps it. This is the number that matters for redaction.
- Thresholds were tuned on development splits only. The EWT, GermEval, PUD and WNUT test numbers are clean.
- The phone and postcode rules were written after looking at misses on the Presidio set and on `eval/challenge_v1.txt`,
  so those two are **not** clean evidence for phones and postcodes.
- spaCy models were trained on other corpora with slightly different labelling rules; treat their scores as a reference point.

```
== real annotated text: strict F1 on person / organisation / location, and on person alone ==
ewt_test (English web text)
   ours      all F1 0.658 (P 0.609 R 0.715)   person F1 0.773 (P 0.707 R 0.853)   1.42 ms/sentence
   spaCy sm  all F1 0.568 (P 0.558 R 0.578)   person F1 0.633 (P 0.650 R 0.617)   4.61 ms/sentence
   spaCy md  all F1 0.641 (P 0.619 R 0.664)   person F1 0.755 (P 0.751 R 0.759)   5.01 ms/sentence
wnut17_test (social media)
   ours      all F1 0.407 (P 0.403 R 0.411)   person F1 0.496 (P 0.500 R 0.492)   1.84 ms/sentence
   spaCy sm  all F1 0.248 (P 0.230 R 0.269)   person F1 0.274 (P 0.283 R 0.266)   4.62 ms/sentence
   spaCy md  all F1 0.276 (P 0.243 R 0.319)   person F1 0.356 (P 0.414 R 0.312)   4.87 ms/sentence
germeval_test (German Wikipedia/news)
   ours      all F1 0.713 (P 0.656 R 0.781)   person F1 0.780 (P 0.728 R 0.840)   1.88 ms/sentence
   spaCy sm  all F1 0.606 (P 0.545 R 0.682)   person F1 0.697 (P 0.672 R 0.723)   4.74 ms/sentence
   spaCy md  all F1 0.656 (P 0.601 R 0.722)   person F1 0.768 (P 0.753 R 0.783)   5.55 ms/sentence
pud_de (German news/Wikipedia)
   ours      all F1 0.709 (P 0.659 R 0.766)   person F1 0.757 (P 0.714 R 0.806)   2.01 ms/sentence
   spaCy sm  all F1 0.642 (P 0.599 R 0.691)   person F1 0.713 (P 0.710 R 0.715)   4.90 ms/sentence
   spaCy md  all F1 0.714 (P 0.675 R 0.757)   person F1 0.796 (P 0.795 R 0.797)   5.72 ms/sentence

== PII caught at all (any overlap, any label) ==
   presidio (templates unseen in training)       recall 0.944  precision 0.948
   our synthetic, new names + new phrasing       recall 0.985  precision 0.988
   ewt_test persons only                         recall 0.953
   germeval_test persons only                    recall 0.949
   wnut17_test persons only (social media)       recall 0.636
      presidio CREDIT_CARD     recall 0.965
      presidio EMAIL_ADDRESS   recall 1.000
      presidio GPE             recall 0.976
      presidio IBAN_CODE       recall 1.000
      presidio IP_ADDRESS      recall 1.000
      presidio ORGANIZATION    recall 0.904
      presidio PERSON          recall 0.993
      presidio PHONE_NUMBER    recall 0.750
      presidio STREET_ADDRESS  recall 0.909
      presidio ZIP_CODE        recall 0.519

== release bar on challenge_v1.txt (87 records, 370 annotated items) ==
   PERSON        found  98.5% (132/134)   bar 95%   PASS
   LOCATION      found  96.8% (91/94)   bar 85%   PASS
   ORGANIZATION  found  97.1% (34/35)   bar 80%   PASS
   STREET        found  93.8% (30/32)   bar 90%   PASS
   PHONE         found 100.0% (29/29)   bar 95%   PASS
   ZIP           found 100.0% (27/27)   bar 90%   PASS
   EMAIL         found 100.0% (18/18)   bar 99%   PASS  (few examples)
   IBAN          found 100.0% (1/1)   bar 99%   PASS  (few examples)
   precision            93.4%            bar 85%   PASS
   all items found 97.8%;  by group: blog 100%, court-record 100%, data-broker 94%, de 99%, directory 97%, en 97%, forum 94%, news 100%, people-search 98%, search-engine 100%, social 100%

== first / last name split (synthetic, new names + new phrasing) ==
   F1 0.861 (P 0.864 R 0.858)

== latency, 1 CPU thread, 2000 texts, avg 16 tokens ==
p50 1.63 ms   p95 3.36 ms   p99 4.44 ms
throughput 557 texts/s, 8906 tokens/s per core
long document (964 tokens, 4760 chars): 92.0 ms
```

## Comparison with GLiNER and GLiNER2-PII

Produced by `python scripts/compare_gliner.py --predict` on the same 2-core machine. 400 randomly drawn texts per test
set (fixed seed), the full challenge set, GLiNER threshold 0.5, labels given to GLiNER: person, organization, location,
street address, postal code, phone number, email, iban. Raw scores: `eval/gliner_scores.json`.

"Persons found" = share of annotated persons overlapped by a detection. "Correct" = share of person detections that
overlap an annotated person.

| Test set | Tarnkappe found / correct | gliner_multi_pii-v1 found / correct | GLiNER2-PII found / correct |
| --- | --- | --- | --- |
| English web text (EWT) | 95.8% / 79.1% | 94.4% / 31.4% | 97.2% / 39.7% |
| German Wikipedia/news (GermEval) | 95.1% / 80.3% | 95.9% / 48.3% | 98.4% / 45.4% |
| German news (PUD) | 97.0% / 85.0% | 95.1% / 51.4% | 98.2% / 48.0% |
| Social media (WNUT 2017) | 56.3% / 54.7% | 92.4% / 41.6% | 95.8% / 39.6% |

| All PII, found / detections real | Tarnkappe | gliner_multi_pii-v1 | GLiNER2-PII |
| --- | --- | --- | --- |
| Presidio, unseen templates (8 shared types) | 93.5% / 95.3% | 84.1% / 74.7% | 97.8% / 80.2% |
| Challenge set v1 | 97.8% / 93.8% | 94.6% / 92.4% | 97.0% / 91.2% |

| Speed, ms per text | Tarnkappe | gliner_multi_pii-v1 | GLiNER2-PII |
| --- | --- | --- | --- |
| All sets | 1.4 to 2.6 | 186 to 228 | 184 to 215 |

What this shows:

- On edited text Tarnkappe finds about as many persons as the GLiNER models (within 1 to 3 points) with roughly twice
  their precision, about 100 times faster.
- On social-media text the GLiNER models find far more persons (92 to 96% against 56%), but only about 40% of what they
  flag as a person is one. They mark pronouns and common nouns ("I", "owner", "children", "er", "Menschen") and user handles.
  GLiNER2-PII's model card names this over-prediction of names as a known limitation.
- With GLiNER2-PII's native name labels (`full_name`, `first_name`, `last_name`) precision improves but recall drops
  (WNUT sample of 130 texts: 77% found, 44% correct at threshold 0.5; 40% found, 65% correct at 0.9). No setting gave both.
- GLiNER2-PII is clearly the better of the two, and strong on addresses, postcodes and organisations.
- Caveats: the corpora annotate only names, so a flagged pronoun counts as wrong even where it refers to a person; the
  WNUT figures here come from 400 texts and differ from the full-set figures above.

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

| Measure | v0.2 | v0.3 dev |
| --- | --- | --- |
| Presidio, templates unseen in training: PII caught | 83.3% | 94.4% |
| Presidio: share of detections that are real PII | 97.5% | 94.8% |
| Persons caught, English web text (EWT test) | 90.0% | 95.3% |
| Persons caught, German Wikipedia/news (GermEval test) | 87.4% | 94.9% |
| Challenge set v1: PII caught | 90.3% | 97.8% |
| Median latency, short text | 1.4 ms | 1.6 ms |
