# Tarnkappe PII tagger v0.6 (model card)

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
| `profiles.json` | Named threshold sets; `pages` (person 0.3, organisation 0.4, company check 0.4) was chosen on hand-labelled web-page snippets | Apache 2.0 |
| `orgcheck.onnx` | Second check on company detections (gradient-boosted trees, 107 KB) | CC BY-SA 4.0 |
| `org_lexicon.json` | Word statistics for the company check, counted on the training corpora | CC BY-SA 4.0 |
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
| v0.6: more Wikipedia paragraphs about companies (3,500 German, 1,200 English) and 800 excerpts from 219 English Wikipedia company articles, labelled the same way (the article's own company always marked) | 5,500 paragraphs plus 2,000 list-shaped examples | text CC BY-SA 4.0 |
| v0.6: sentences built by `scripts/brand_examples.py`: organisation names without a legal form in news, profile and list sentences (company names from the CompanyData.com CC0 samples and the Wikipedia articles above, invented surname-like and lower-case startup names, broadcasters, universities, clubs) | 12,000 examples | generated |
| Page-like examples with company names: a CC0 sample of European and US companies from [CompanyData.com](https://github.com/companydatacom/public-datasets) and the STOXX Europe 600 and S&P 500 constituents (names only) | 10,000 examples, about 8,300 names (a fifth of each list held out) | CC0 1.0; names only |
| Generated sentences (`tarnkappe/data.py`) | 25,000, English and German | ours |
| [Presidio Research](https://github.com/microsoft/presidio-research) templates (even-numbered), rendered with our values | 8,000 | MIT |

No real personal data was collected for training. No non-commercial datasets were used.

## Evaluation

See [RESULTS.md](../../RESULTS.md). v0.4 adds training examples built from about 8,300 real company names and a second
check on company detections. On 21 real scanned snippets never used for tuning (`pages` profile with scan-level
sharing) it finds 100% of persons (one likely labelling slip aside) and 62% of companies, with 87% of its company
detections correct (v0.3: 62% and 62%). With the default thresholds persons are found 95.8% of the time in English web
text, 96.2% in German Wikipedia/news, 83.2% in tweets and 85.3% in mixed social-media text. `thresholds.json` comes from
`scripts/tune_thresholds.py`, except STREET, which is kept at 0.05.
