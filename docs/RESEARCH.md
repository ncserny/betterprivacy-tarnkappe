# Research notes: a small, fast PII detector

Written 18 November 2026. These notes explain why Tarnkappe is built the way it is.

## Existing open models

Good open PII models exist, but those surveyed are 150M parameters or larger, so none is very fast on a CPU.

| Model | Size | Licence | Notes |
| --- | --- | --- | --- |
| [gliner_multi_pii-v1](https://huggingface.co/urchade/gliner_multi_pii-v1) | about 278M parameters (mDeBERTa-v3 base) | Apache 2.0 | Person, organisation, phone, address, email and more in six languages including German. Labels are chosen at request time. Trained on synthetic data; the model card reports no benchmark or speed numbers. |
| [GLiNER2-PII](https://github.com/fastino-ai/GLiNER2) (`fastino/gliner2-privacy-filter-PII-multi`) | 205M per its repository | Apache 2.0 | Newer generation from the same author, 42 PII types, multilingual. Its authors report higher accuracy than OpenAI's and NVIDIA's models. |
| [OpenAI Privacy Filter](https://github.com/openai/privacy-filter) | 1.5B (about 50M active) | Apache 2.0 | Primarily English. An [independent test](https://privacybridge-website.pages.dev/articles/openai-privacy-filter-tested/) found no organisation or location label and a median of 3.2 s per document on a laptop CPU. |
| [NVIDIA GLiNER-PII](https://huggingface.co/nvidia/gliner-PII) | 570M | NVIDIA Open Model License | English, 55+ types. Its card reports strict F1 of 0.87 on its own data and 0.64 on AI4Privacy. |

Three lessons from published benchmarks:

- **Data decides quality more than architecture.** Tonic's [benchmark of OpenAI's model](https://www.tonic.ai/blog/benchmarking-openai-privacy-filter-pii-detection) concludes that models trained on synthetic PII underperform on real documents.
- **Patterns still win on structured identifiers.** Serious pipelines pair a model with regular expressions and checksums for emails, IBANs and card numbers.
- **GLiNER-class speed is tens of milliseconds per call**, not one. One [ONNX build of GLiNER2 multi](https://huggingface.co/SemplificaAI/gliner2-multi-v1-onnx) reports about 21 ms on a 16-core desktop CPU.

Both GLiNER models have since been run against Tarnkappe on the same texts; see "Comparison with GLiNER" below and [RESULTS.md](../RESULTS.md).

## Design: cheap checks first

1. **Patterns and checksums** for email, phone, IBAN, card numbers, IP addresses.
2. **A small tagger** for names, organisations and addresses. This is the model in this repository.
3. **A large model as an optional accuracy tier**, for texts where the small model is unsure.

A large model such as GLiNER can also act as a *teacher*: it labels unlabelled text offline, and the small
model is trained on those labels (knowledge distillation). Tarnkappe v0.2 does not do this yet.

## Training data

Only data whose licence allows open release and commercial use was considered.

| Source | Licence | Used in v0.2 |
| --- | --- | --- |
| [UNER English EWT](https://github.com/UniversalNER/UNER_English-EWT) | CC BY-SA 4.0 | yes |
| [GermEval 2014](https://sites.google.com/site/germeval2014ner/) | CC BY 4.0 | yes |
| [WNUT 2017](https://github.com/leondz/emerging_entities_17) | CC BY 4.0 | yes |
| [Broad Twitter Corpus](https://github.com/GateNLP/broad_twitter_corpus) | CC BY 4.0 | yes, since the v0.3 retraining |
| [Presidio Research](https://github.com/microsoft/presidio-research) templates | MIT | yes, half for training, half held out for testing |
| Generated text (`pii/data.py`) | ours | yes |
| [NVIDIA Nemotron-PII](https://huggingface.co/blog/nvidia/nemotron-pii) | CC BY 4.0 | no, candidate for a later version |
| [AI4Privacy pii-masking](https://huggingface.co/datasets/ai4privacy/pii-masking-300k/blob/main/LICENSE.md) | academic and non-commercial only | no, excluded |
| [WikiNEuRal](https://github.com/Babelscape/wikineural) | CC BY-NC-SA 4.0 | no, excluded |

Because UNER English EWT is share-alike, the model weights are released under CC BY-SA 4.0.
Creative Commons' [guidance on AI training](https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/)
describes that as the conservative reading for publicly shared models.

## What two prototype rounds showed

- **v0.1, generated text only.** Strict F1 was 0.99 on familiar sentence patterns with unseen names, but 0.71 on
  unseen patterns. The model learned the templates, not the language.
- **v0.2, real labelled text plus pretrained word vectors.** Person F1 of about 0.8 on real English and German
  test sets, at 1 to 2 ms per sentence. Details in [RESULTS.md](../RESULTS.md).

The step from v0.1 to v0.2 came almost entirely from better data and pretrained input vectors, with the same small architecture.

- **v0.3 work, same weights.** Recall-favouring thresholds, libphonenumber and country postcode patterns raised the
  share of PII found on Presidio's test sentences from 83% to 94%, and of persons in English and German edited text
  from about 89% to 95%. Informal text stayed poor: 64% of persons found on WNUT 2017.
- **Transformer reference.** spaCy's RoBERTa-based English pipeline, which never saw these corpora, reaches person F1
  0.84 on English web text and 0.65 on social media, against 0.77 and 0.50 for Tarnkappe, at about 60 ms per sentence
  against 1.5 ms. The small model's gap is mostly on informal text, which is what distillation from a larger model should address.

## Comparison with GLiNER and what it means for distillation

On 400 texts per test set, GLiNER2-PII finds 96 to 98% of annotated persons and Tarnkappe 95 to 97% on edited text,
but only 56% on social-media text. The GLiNER models pay for their recall with precision: only 31 to 51% of what they
flag as a person overlaps an annotated person, against 55 to 85% for Tarnkappe. They mark pronouns, common nouns and
user handles as persons, which GLiNER2-PII's own model card lists as a known limitation. They take about 190 ms per
text on two CPU cores, against under 2 ms.

Consequences:

- **Do not distil person labels from GLiNER as they are.** A student trained on them would learn to flag "I" and "owner".
  The teacher is more useful for addresses, postcodes and organisations, where it is strong.
- **For informal text, human-labelled social-media corpora are the cleaner lever**, followed by fine-tuning a small
  pretrained transformer on gold data. spaCy's RoBERTa pipeline shows a transformer can be both more complete and more
  precise on such text (person recall 0.71 at precision 0.59, strict matching, on WNUT 2017).
- **A small transformer will not be under 10 ms on modest hardware.** Multilingual MiniLM-class encoders took 17 to 22 ms
  per short text on one core here after int8 quantisation: about ten times slower than Tarnkappe, ten times faster than GLiNER.

## Retraining with tweets

Adding the Broad Twitter Corpus (6,192 human-labelled tweets) and casing augmentation raised the share of persons
found in social-media text from 64% to 85% (WNUT 2017 test) and in held-out tweets from 35% to 88% (BTC test), with
1 to 2 points more persons found on edited text at about 3 points lower precision. Person precision on social media
fell from 55% to 45%, which is still slightly above GLiNER's. TweetNER7, Tweebank-NER and the Temporal Twitter Corpus were not used because their licences
are not clearly stated; the DFKI SmartData corpus was not used because its labelling rules differ from ours.

## Open questions

- How do the systems compare on real scanned pages, where the hand-checked test set is still missing?
- Does fine-tuning a small pretrained transformer (`next/finetune_encoder.py`) close most of that gap while staying fast enough?
- Which languages come next?
