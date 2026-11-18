# betterprivacy-tarnkappe

Fast, small detection of personal data (PII) in English and German text, built for
[BetterPrivacy](https://betterprivacy.me).

*Tarnkappe* is the cloak of invisibility from German legend. This library finds the personal
data in a text so it can be flagged or hidden.

- **Fast:** about 1.4 ms per short text on one CPU core. No GPU.
- **Small:** 5 MB model plus 25 MB of word vectors. Two runtime dependencies (`numpy`, `onnxruntime`).
- **Private:** runs locally; text is processed in memory and never logged or stored.
- **Open:** code, weights, training script and benchmark are all in this repository.

> **Status: v0.2, prototype.** Good for its size, not state of the art, and not yet compared
> with larger models such as GLiNER. Read [Limits](#limits) before relying on it.

## What it detects

| Type | Found by | Example |
| --- | --- | --- |
| `FIRST_NAME`, `LAST_NAME` | model, then a splitting rule | Laura, Becker |
| `ORGANIZATION` | model | Nordwind GmbH |
| `STREET`, `ZIP`, `LOCATION` | model | Hauptstraße 12, 50667, Köln |
| `PHONE` | pattern for clear formats, model for the rest | +49 30 1234567 |
| `EMAIL` | pattern | laura@nordwind.de |
| `IBAN` | pattern + mod-97 checksum | DE89 3704 0044 0532 0130 00 |
| `CREDIT_CARD` | pattern + Luhn checksum | 4111 1111 1111 1111 |
| `IP_ADDRESS` | pattern (IPv4) | 192.168.0.12 |

## Quick start

```bash
git clone https://github.com/ncserny/betterprivacy-tarnkappe
cd betterprivacy-tarnkappe
pip install -r requirements.txt
```

### In Python

```python
from pii import Detector

detector = Detector()

detector.detect("Call Thomas Weber on +49 30 1234567")
# [{'type': 'FIRST_NAME', 'start': 5, 'end': 11, 'score': 1.0, 'source': 'model', 'text': 'Thomas'},
#  {'type': 'LAST_NAME',  'start': 12, 'end': 17, ...},
#  {'type': 'PHONE',      'start': 21, 'end': 35, 'score': 1.0, 'source': 'regex', ...}]

detector.redact("Call Thomas Weber on +49 30 1234567")
# 'Call [FIRST_NAME] [LAST_NAME] on [PHONE]'
```

`detect(text, min_score=0.5, split_names=True)`: raise `min_score` for fewer false alarms, lower it
to catch more. With `split_names=False` people are returned as one `PERSON` entity.

### As an HTTP API

```bash
uvicorn api:app --port 8000
# or: docker build -t tarnkappe . && docker run -p 8000:8000 tarnkappe
```

```bash
curl -s localhost:8000/v1/detect -H 'content-type: application/json' \
  -d '{"text": "Hi, I am Laura Becker from Nordwind GmbH, laura@nordwind.de", "redact": true}'
```

```json
{
  "has_pii": true,
  "entities": [
    {"type": "FIRST_NAME",   "text": "Laura",             "start": 9,  "end": 14, "score": 0.9998, "source": "model"},
    {"type": "LAST_NAME",    "text": "Becker",            "start": 15, "end": 21, "score": 0.9998, "source": "model"},
    {"type": "ORGANIZATION", "text": "Nordwind GmbH",     "start": 27, "end": 40, "score": 0.9484, "source": "model"},
    {"type": "EMAIL",        "text": "laura@nordwind.de", "start": 42, "end": 59, "score": 1.0,    "source": "regex"}
  ],
  "redacted": "Hi, I am [FIRST_NAME] [LAST_NAME] from [ORGANIZATION], [EMAIL]",
  "took_ms": 1.2
}
```

| Endpoint | Body | Returns |
| --- | --- | --- |
| `POST /v1/detect` | `text` (up to 100,000 characters), optional `redact`, `min_score` | entities with character offsets, optional redacted text |
| `POST /v1/detect/batch` | `texts` (up to 256), optional `redact`, `min_score` | one result per text |
| `GET /healthz` | | `{"status": "ok"}` |

Interactive API docs are served at `/docs`. The service has no authentication or rate limiting;
add both before exposing it to the internet.

## How it works

1. **Patterns and checksums** catch structured identifiers. They are near-perfect in precision and cost microseconds.
2. **A 2.5M-parameter neural tagger** reads the text once and labels each word. It combines pretrained word
   vectors, hashed word embeddings and a character CNN, followed by a two-layer BiGRU. It is exported to ONNX and quantised to int8.
3. **A rule** splits each detected person into first and last name, using titles ("Herr", "Dr."), name particles ("van der") and a first-name list.

Pattern hits win over model hits where they overlap. Background and design choices are in [docs/RESEARCH.md](docs/RESEARCH.md).

## Accuracy and speed

Measured with `python bench.py --spacy` on 2 CPU cores, one thread, shipped int8 model. Full output: [RESULTS.md](RESULTS.md).

Strict F1 (exact span and type) on human-labelled test sets:

| Test set | Tarnkappe, all types | Tarnkappe, persons | spaCy md, all types | spaCy md, persons |
| --- | --- | --- | --- | --- |
| English web text (UNER EWT) | 0.70 | 0.82 | 0.64 | 0.76 |
| German Wikipedia/news (GermEval 2014) | 0.76 | 0.82 | 0.66 | 0.77 |
| German news (UNER PUD) | 0.73 | 0.77 | 0.71 | 0.80 |
| Social media (WNUT 2017) | 0.42 | 0.49 | 0.28 | 0.36 |

The comparison favours Tarnkappe on EWT and GermEval, because it was trained on their training splits and spaCy was not. PUD is neutral ground.

On Microsoft Presidio's generated test sentences (templates not used in training), 83% of personal data is found
and 97.5% of flagged spans are real: email, IBAN and IP 100%, credit card 95%, person 94%, city/country 88%,
street address 73%, organisation 69%, phone 58%, postcode 33%.

| Speed and size | |
| --- | --- |
| Short text (16 tokens) | 1.4 ms median, 4.0 ms at p99 |
| Throughput per core | about 650 texts or 10,400 tokens per second |
| 960-token document | 78 ms |
| Memory, start-up | about 270 MB, 1 s |

## Limits

- **Not state of the art.** Transformer models such as GLiNER are expected to be more accurate and slower. `scripts/compare_gliner.py` runs that comparison but has not been run yet.
- **English and German only.**
- **Weak spots:** organisations, informal social-media text, postcodes and phone numbers in foreign formats.
- **It will miss things.** Treat it as a data-minimisation aid, not as a guarantee of anonymisation or of legal compliance.

## Train and evaluate

```bash
pip install -r requirements-train.txt
bash scripts/fetch_data.sh        # downloads the public corpora into data/
python train.py                   # about 30 minutes on 2 CPU cores; writes model/
python bench.py                   # samples, accuracy, PII recall, latency
python -m pytest tests
```

`python bench.py --spacy` adds the spaCy baselines (needs `spacy` and its `sm`/`md` models for English and German).
`model/vectors.npz` is included; `scripts/export_vectors.py` rebuilds it from spaCy's medium models.

## Repository layout

| Path | What |
| --- | --- |
| `pii/detect.py` | Runtime: patterns, ONNX model, name splitting, redaction |
| `pii/features.py` | Tokeniser, features, word vectors, tag conversion (numpy only) |
| `pii/data.py` | Synthetic data generator; train and test names and templates are kept disjoint |
| `pii/corpora.py` | Loaders for the public corpora |
| `train.py` | Training, evaluation, ONNX export and quantisation |
| `api.py` | FastAPI service |
| `bench.py` | Benchmark |
| `model/` | Trained model files and the [model card](model/README.md) |
| `docs/RESEARCH.md` | Research notes: existing models, design, data |
| `scripts/compare_gliner.py` | GLiNER and GLiNER2-PII on the same test sets. **Not run yet.** |
| `next/finetune_encoder.py` | Fine-tuning a pretrained transformer encoder. **Not run yet.** |

## Licence

- **Code** (everything outside `model/`): [Apache License 2.0](LICENSE).
- **Model weights** (`model/tagger.onnx`, `model/tagger.int8.onnx`): [CC BY-SA 4.0](model/LICENSE),
  because part of the training data (UNER English EWT) is CC BY-SA 4.0. Commercial use is allowed;
  if you publish modified weights they must carry the same licence and the credits.
- Training-data credits and the licences of the other files in `model/`: [NOTICE](NOTICE) and the [model card](model/README.md).
