# betterprivacy-tarnkappe

Fast, small detection of personal data (PII) in English and German text, built for
[BetterPrivacy](https://betterprivacy.me).

*Tarnkappe* is the cloak of invisibility from German legend. This library finds the personal
data in a text so it can be flagged or hidden.

- **Fast:** about 1.6 ms per short text on one CPU core. No GPU.
- **Small:** 5 MB model plus 25 MB of word vectors. Three runtime dependencies (`numpy`, `onnxruntime`, `phonenumbers`).
- **Private:** runs locally; text is processed in memory and never logged or stored.
- **Open:** code, weights, training script and benchmark are all in this repository.

> **Status: v0.3 in development, prototype.** Good for its size, not state of the art, not yet compared
> with larger models such as GLiNER, and not yet at its own [release bar](#release-bar). Read [Limits](#limits) before relying on it.

## What it detects

| Type | Found by | Example |
| --- | --- | --- |
| `FIRST_NAME`, `LAST_NAME` | model, then a splitting rule | Laura, Becker |
| `ORGANIZATION` | model | Nordwind GmbH |
| `STREET`, `LOCATION` | model | Hauptstraße 12, Köln |
| `ZIP` | country patterns next to an address part, plus the model | 50667, NW1 6XE |
| `PHONE` | Google's libphonenumber (`phonenumbers`), patterns, plus the model | +49 30 1234567, 0171 3920045 |
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

`detect(text, min_score=None, split_names=True)`: by default each entity type uses its own threshold from
`model/thresholds.json`, tuned to favour finding PII over avoiding false alarms. Pass `min_score` (0 to 1) to use one
threshold for all types: higher flags less, lower flags more. With `split_names=False` people are returned as one `PERSON` entity.

National phone formats are recognised for Germany, Austria, Switzerland, the UK and the US by default
(`Detector(phone_regions=("DE", "FR", ...))` changes that); international `+..` numbers always are.

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

1. **Patterns, checksums and libphonenumber** catch structured identifiers: email, IBAN, cards, IP addresses, phone numbers.
   Postcodes are matched by country pattern; plain numbers only count next to a detected street or city or after a keyword such as "PLZ".
2. **A 2.5M-parameter neural tagger** reads the text once and labels each word. It combines pretrained word
   vectors, hashed word embeddings and a character CNN, followed by a two-layer BiGRU. It is exported to ONNX and quantised to int8.
3. **A rule** splits each detected person into first and last name, using titles ("Herr", "Dr."), name particles ("van der") and a first-name list.

Pattern hits win over model hits where they overlap. Background and design choices are in [docs/RESEARCH.md](docs/RESEARCH.md).

## Accuracy and speed

Measured with `python bench.py --spacy` on 2 CPU cores, one thread, shipped int8 model. Full output and notes: [RESULTS.md](RESULTS.md).

**Share of personal data found** (an item counts as found if any detection overlaps it):

| Test | Found | Clean test? |
| --- | --- | --- |
| Persons in English web text (UNER EWT test) | 95.3% | yes |
| Persons in German Wikipedia/news (GermEval 2014 test) | 94.9% | yes |
| Persons in social-media text (WNUT 2017 test) | 63.6% | yes |
| All PII in Microsoft Presidio sentences, templates unseen in training | 94.4% (94.8% of detections are real) | not for phones and postcodes |
| All PII in the hand-written challenge set (`eval/challenge_v1.txt`) | 97.8% (93.4% of detections are real) | not for phones and postcodes |

On the Presidio set: email, IBAN and IP 100%, person 99%, city/country 98%, credit card 97%, street address 91%,
organisation 90%, phone 75%, postcode 52%.

Strict F1 (exact span and type) on persons is 0.77 (EWT), 0.78 (GermEval), 0.76 (PUD) and 0.50 (WNUT). It is lower
than in v0.2 because the thresholds now favour recall. For reference, spaCy's RoBERTa-based `en_core_web_trf`
reaches person F1 0.84 on EWT and 0.65 on WNUT at about 60 ms per sentence.

| Speed and size | |
| --- | --- |
| Short text (16 tokens) | 1.6 ms median, 4.4 ms at p99 |
| Throughput per core | about 550 texts per second |
| 960-token document | 92 ms |
| Memory, start-up | about 270 MB, 1 s |

## Release bar

`eval/release_bar.json` holds the proposed bar a version must meet on a **hand-checked test set of real pages**
before it is called production-ready: persons found 95%, email and IBAN 99%, phone 95%, street and postcode 90%,
location 85%, organisation 80%, and at least 85% of detections real.

Status: **not met.** There is no real-page test set yet, persons in informal text are far below the bar, and phone
and postcode results are only measured on sets the rules were developed against. See [eval/README.md](eval/README.md)
for how to build the test set and score against the bar (`python bench.py --testset <file> --gate`).

## Limits

- **Not state of the art.** Transformer models are more accurate, especially on informal text, and far slower. The comparison with GLiNER and GLiNER2-PII (`scripts/compare_gliner.py`) has not been run yet.
- **English and German only.**
- **Weak spots:** informal social-media text, phone numbers from countries outside the configured regions, postcodes in free text without an address around them.
- **More false alarms than v0.2,** by design: the thresholds trade precision for recall.
- **It will miss things.** Treat it as a data-minimisation aid, not as a guarantee of anonymisation or of legal compliance.

## Train and evaluate

```bash
pip install -r requirements-train.txt
bash scripts/fetch_data.sh        # downloads the public corpora into data/
python train.py                   # about 30 minutes on 2 CPU cores; writes model/
python scripts/tune_thresholds.py # re-tune per-type thresholds on dev data; writes model/thresholds.json
python bench.py                   # samples, accuracy, PII recall, release bar, latency
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
| `eval/` | Challenge test set, release bar, loader for hand-checked test sets |
| `scripts/tune_thresholds.py` | Per-type threshold tuning on development data |
| `scripts/prelabel.py` | Pre-labels raw pages for hand-checking |
| `scripts/distil_label.py` | Labels text with a teacher model for distillation. GLiNER teachers **not run yet.** |
| `scripts/compare_gliner.py` | GLiNER and GLiNER2-PII on the same test sets. **Not run yet.** |
| `next/finetune_encoder.py` | Fine-tuning a pretrained transformer encoder. **Not run yet.** |

## Licence

- **Code** (everything outside `model/`): [Apache License 2.0](LICENSE).
- **Model weights** (`model/tagger.onnx`, `model/tagger.int8.onnx`): [CC BY-SA 4.0](model/LICENSE),
  because part of the training data (UNER English EWT) is CC BY-SA 4.0. Commercial use is allowed;
  if you publish modified weights they must carry the same licence and the credits.
- Training-data credits and the licences of the other files in `model/`: [NOTICE](NOTICE) and the [model card](model/README.md).
