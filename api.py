"""HTTP API.   uvicorn api:app --host 0.0.0.0 --port 8000 --workers 2

POST /v1/detect  {"text": "...", "redact": true}
POST /v1/detect/batch  {"texts": ["...", "..."]}
Texts are processed in memory and never logged or stored.
"""
import time
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from pii.detect import Detector

app = FastAPI(title="PII Detector", version="0.1.0")
detector = Detector()


class DetectRequest(BaseModel):
    text: str = Field(..., max_length=100_000)
    redact: bool = False
    min_score: float = Field(0.5, ge=0, le=1)


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., max_length=256)
    redact: bool = False
    min_score: float = Field(0.5, ge=0, le=1)


class Entity(BaseModel):
    type: str
    text: str
    start: int
    end: int
    score: float
    source: str


class DetectResponse(BaseModel):
    has_pii: bool
    entities: list[Entity]
    redacted: Optional[str] = None
    took_ms: float


def _run(text, redact, min_score):
    t = time.perf_counter()
    ents = detector.detect(text, min_score)
    return DetectResponse(has_pii=bool(ents), entities=ents,
                          redacted=detector.redact(text, ents) if redact else None,
                          took_ms=round((time.perf_counter() - t) * 1000, 3))


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/v1/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    return _run(req.text, req.redact, req.min_score)


@app.post("/v1/detect/batch", response_model=list[DetectResponse])
def detect_batch(req: BatchRequest):
    return [_run(t, req.redact, req.min_score) for t in req.texts]
