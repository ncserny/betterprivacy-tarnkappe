# Tarnkappe PII tagger v0.3 development (model card)

A 2.5M-parameter token tagger for personal data in English and German text.
Labels: PERSON, ORG, LOCATION, STREET, ZIP, EMAIL, PHONE. The runtime (`tarnkappe/detect.py`)
adds regex detectors and splits PERSON into FIRST_NAME / LAST_NAME.

## Files and licences

| File | What | Licence |
| --- | --- | --- |
| `tagger.int8.onnx` | Quantised model used by default (5 MB) | CC BY-SA 4.0 (`LICENSE` in this folder) |
| `tagger.onnx` | Float32 model (10 MB) | CC BY-SA 4.0 |
| `vectors.npz` | Frozen input word vectors, from spaCy's `en_core_web_md` and `de_core_news_md` | CC0 vectors from MIT-licensed packages |
| `first_names.txt` | First-name list for the name-splitting rule, sampled from Faker | MIT |
| `thresholds.json` | Per-type decision thresholds, tuned for recall on development data by `scripts/tune_thresholds.py` | Apache 2.0 |
| `profiles.json` | Named threshold sets; `pages` (person 0.3, organisation 0.4) was chosen on hand-labelled web-page snippets | Apache 2.0 |
| `metrics.json` | Test scores of the float32 model written by `train.py` | Apache 2.0 |

The weights are share-alike because the training data includes UNER English EWT and Wikipedia text (both CC BY-SA 4.0).
If you publish a modified version of the weights, it must carry the same licence and these credits.

## Training data

| Data | Size | Licence |
| --- | --- | --- |
| [UNER English EWT](https://github.com/UniversalNER/UNER_English-EWT), train split | 12,543 sentences of web text | CC BY-SA 4.0 |
| [GermEval 2014](https://sites.google.com/site/germeval2014ner/), train split | 24,000 German Wikipedia/news sentences | CC BY 4.0 |
| [WNUT 2017](https://github.com/leondz/emerging_entities_17), train split | 3,395 social-media sentences | CC BY 4.0 |
| [Broad Twitter Corpus](https://github.com/GateNLP/broad_twitter_corpus), sections a, b, e, g | 6,192 tweets; its @handle labels are not used | CC BY 4.0 |
| [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (German and English, dump of 1 November 2023), labelled by [GLiNER2-PII](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi) | 11,000 paragraphs, half of them wrapped in menu and footer lines, plus 3,000 list-shaped examples built from their names | text CC BY-SA 4.0; teacher model Apache 2.0 |
| Generated sentences (`tarnkappe/data.py`) | 25,000, English and German | ours |
| [Presidio Research](https://github.com/microsoft/presidio-research) templates (even-numbered), rendered with our values | 8,000 | MIT |

No real personal data was collected for training. No non-commercial datasets were used.

## Evaluation

See [RESULTS.md](../../RESULTS.md). The current model reads line breaks and was retrained for web pages with
teacher-labelled Wikipedia text. With the default thresholds persons are found 95.8% of the time in English web text,
95.8% in German Wikipedia/news, 79.8% in tweets and 79.7% in mixed social-media text; tweets were at 88.2% before the
retraining. On 65 hand-labelled snippets of real scanned pages (`pages` profile with scan-level name sharing) it finds
96 to 100% of persons and 82 to 84% of companies, with 78 to 92% of those detections correct; these snippets were used
for tuning. `thresholds.json` comes from `scripts/tune_thresholds.py`, except STREET, which is kept at 0.05.
