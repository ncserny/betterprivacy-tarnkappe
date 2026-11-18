# Tarnkappe PII tagger v0.2 (model card)

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
| `metrics.json` | Test scores of the float32 model written by `train.py` | Apache 2.0 |

The weights are share-alike because the training data includes UNER English EWT (CC BY-SA 4.0).
If you publish a modified version of the weights, it must carry the same licence and these credits.

## Training data

| Data | Size | Licence |
| --- | --- | --- |
| [UNER English EWT](https://github.com/UniversalNER/UNER_English-EWT), train split | 12,543 sentences of web text | CC BY-SA 4.0 |
| [GermEval 2014](https://sites.google.com/site/germeval2014ner/), train split | 24,000 German Wikipedia/news sentences | CC BY 4.0 |
| [WNUT 2017](https://github.com/leondz/emerging_entities_17), train split | 3,395 social-media sentences | CC BY 4.0 |
| Generated sentences (`pii/data.py`) | 25,000, English and German | ours |
| [Presidio Research](https://github.com/microsoft/presidio-research) templates (even-numbered), rendered with our values | 8,000 | MIT |

No real personal data was collected for training. No non-commercial datasets were used.

## Evaluation

See [RESULTS.md](../RESULTS.md). In short: strict F1 for persons is 0.82 on English web text and
German Wikipedia/news, 0.77 on German news, 0.49 on social media. On Presidio test sentences from
templates unseen in training, 83% of PII is found and 97.5% of flagged spans are real PII.

## Limits and intended use

- A data-minimisation aid. It does not guarantee anonymisation and will miss some personal data.
- English and German only. Weakest on organisations, informal text, and phone numbers or postcodes in foreign formats.
- Not yet compared with GLiNER-class models, which are expected to be more accurate and slower.
- The benchmarks include test splits of two corpora whose training splits were used (EWT, GermEval).
