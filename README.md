# betterprivacy-tarnkappe

Fast, small PII detection for English and German text, built for [BetterPrivacy](https://betterprivacy.me).

It finds first name, last name, organisation, street, postcode, location (city/country),
email, phone, IBAN, credit card and IP address, and can redact them. It runs on one CPU core
in about 1.4 ms per short text; no GPU needed.

> Status: prototype (v0.2). Good for its size, not state of the art. See [RESULTS.md](RESULTS.md) and "Limits".

## How it works

1. **Regex + checksums** for structured identifiers (email, IBAN mod-97, card Luhn, IP, clear phone formats).
2. **A 2.5M-parameter neural tagger** (pretrained word vectors + hashed word embeddings + char-CNN + BiGRU),
   exported to a 5 MB int8 ONNX file, for names, organisations, addresses and other phone formats.
3. **A rule** splits each detected person into first and last name.

## Use it

```bash
pip install -r requirements.txt
uvicorn api:app --port 8000          # HTTP API
curl -s localhost:8000/v1/detect -H 'content-type: application/json' \
  -d '{"text": "Hi, I am Laura Becker from Nordwind GmbH, laura@nordwind.de", "redact": true}'
```

```python
from pii import Detector
d = Detector()
d.detect("Call Thomas Weber on +49 30 1234567")
# [{'type': 'FIRST_NAME', 'start': 5, 'end': 11, 'score': 1.0, 'source': 'model', 'text': 'Thomas'}, ...]
d.redact("Call Thomas Weber on +49 30 1234567")   # 'Call [FIRST_NAME] [LAST_NAME] on [PHONE]'
```

Docker: `docker build -t tarnkappe . && docker run -p 8000:8000 tarnkappe`

The runtime needs only `numpy` and `onnxruntime` plus three files in `model/`:
`tagger.int8.onnx` (5 MB), `vectors.npz` (25 MB), `first_names.txt`. Start-up takes about 1 s
and a process uses about 270 MB of memory. Texts are processed in memory and never logged.

## Results in short

Strict F1 on human-labelled test sets (full output in [RESULTS.md](RESULTS.md)):

| Test set | All types | Persons |
| --- | --- | --- |
| English web text (UNER EWT test) | 0.70 | 0.82 |
| German Wikipedia/news (GermEval 2014 test) | 0.76 | 0.82 |
| German news (UNER PUD) | 0.73 | 0.77 |
| Social media (WNUT 2017 test) | 0.42 | 0.49 |

## Train it

```bash
pip install -r requirements-train.txt
bash scripts/fetch_data.sh           # public corpora -> data/
python train.py                      # about 30 minutes on 2 CPU cores
bash scripts/fetch_data.sh && python bench.py --spacy   # accuracy, PII recall, latency (--spacy needs spaCy + its sm/md models)
python -m pytest tests
```

`model/vectors.npz` is already included; `scripts/export_vectors.py` rebuilds it from spaCy's medium models.

## Files

| Path | What |
| --- | --- |
| `pii/detect.py` | Runtime: regex layer + ONNX model + first/last name split + redaction |
| `pii/features.py` | Tokeniser, features, word vectors, BIO tag conversion (numpy only) |
| `pii/data.py` | Synthetic data generator; train/test names and templates kept disjoint |
| `pii/corpora.py` | Loaders for the public corpora |
| `train.py` | Training, evaluation, ONNX export and quantisation |
| `api.py` | FastAPI service |
| `bench.py` | Accuracy on real test sets (optionally next to spaCy), PII recall, latency |
| `model/` | Trained model files and the [model card](model/README.md) |
| `scripts/compare_gliner.py` | Same test for gliner_multi_pii-v1 and GLiNER2-PII. **Not run yet.** |
| `next/finetune_encoder.py` | Fine-tune a pretrained transformer encoder. **Not run yet.** |

## Limits

- Transformer models such as GLiNER are expected to score higher; that comparison has not been run yet.
- English and German only. Organisations, informal social-media text and foreign phone/postcode formats are the weak spots.
- A detector is a data-minimisation aid, not a guarantee of anonymisation.

## Licence

- **Code** (everything outside `model/`): [Apache License 2.0](LICENSE).
- **Model weights** (`model/tagger.onnx`, `model/tagger.int8.onnx`): [CC BY-SA 4.0](model/LICENSE),
  because part of the training data (UNER English EWT) is CC BY-SA 4.0.
- Other files in `model/` and all training-data credits: see [NOTICE](NOTICE) and the [model card](model/README.md).
