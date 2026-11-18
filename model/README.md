# Tarnkappe PII tagger v0.3 development (model card)

A 2.5M-parameter token tagger for personal data in English and German text.
Labels: PERSON, ORG, LOCATION, STREET, ZIP, EMAIL, PHONE. The runtime (`pii/detect.py`)
adds regex detectors and splits PERSON into FIRST_NAME / LAST_NAME.

## Files and licences

| File | What | Licence |
| --- | --- | --- |
| `tagger.int8.onnx` | Quantised model used by default (5 MB) | CC BY-SA 4.0 (`LICENSE` in this folder) |
| `tagger.onnx` | Float32 model (10 MB) | CC BY-SA 4.0 |
| `vectors.npz` | Frozen input word vectors, from spaCy's `en_core_web_md` and `de_core_news_md` | CC0 vectors from MIT-licensed packages |
| `first_names.txt` | First-name list for the name-splitting rule, sampled from Faker | MIT |
| `thresholds.json` | Per-type decision thresholds, tuned for recall on development data by `scripts/tune_thresholds.py` | Apache 2.0 |
| `metrics.json` | Test scores of the float32 model written by `train.py` | Apache 2.0 |

The weights are share-alike because the training data includes UNER English EWT (CC BY-SA 4.0).
If you publish a modified version of the weights, it must carry the same licence and these credits.

## Training data

| Data | Size | Licence |
| --- | --- | --- |
| [UNER English EWT](https://github.com/UniversalNER/UNER_English-EWT), train split | 12,543 sentences of web text | CC BY-SA 4.0 |
| [GermEval 2014](https://sites.google.com/site/germeval2014ner/), train split | 24,000 German Wikipedia/news sentences | CC BY 4.0 |
| [WNUT 2017](https://github.com/leondz/emerging_entities_17), train split | 3,395 social-media sentences | CC BY 4.0 |
| [Broad Twitter Corpus](https://github.com/GateNLP/broad_twitter_corpus), sections a, b, e, g | 6,192 tweets; its @handle labels are not used | CC BY 4.0 |
| Generated sentences (`pii/data.py`) | 25,000, English and German | ours |
| [Presidio Research](https://github.com/microsoft/presidio-research) templates (even-numbered), rendered with our values | 8,000 | MIT |

No real personal data was collected for training. No non-commercial datasets were used.

## Evaluation

See [RESULTS.md](../RESULTS.md). Compared with v0.2 the model was retrained with tweets and casing augmentation
(12% of real texts lower-cased, entities lower-cased in a further 12%), is decoded with recall-favouring thresholds,
and has phone and postcode rules around it. Persons are found 95.5% of the time in English web text, 96.3% in German
Wikipedia/news, 88.2% in tweets and 85.1% in mixed social-media text.

## Limits and intended use

- A data-minimisation aid. It does not guarantee anonymisation and will miss some personal data.
- English and German only. Weakest on organisations, informal text, and phone numbers or postcodes in foreign formats.
- Next to GLiNER2-PII it finds about as many persons in edited text with about twice the precision, and fewer in social-media text (85% against 96%).
- The benchmarks include test splits of corpora whose training splits were used (EWT, GermEval, WNUT, BTC).
