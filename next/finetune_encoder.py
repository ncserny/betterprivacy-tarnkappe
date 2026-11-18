"""PRODUCTION PATH (NOT RUN in the prototype sandbox - it has no Hugging Face access).

Fine-tunes a small pretrained multilingual encoder as a BIO token classifier.
Feed it data labelled by a teacher (e.g. urchade/gliner_multi_pii-v1) plus
permissively licensed synthetic data (e.g. nvidia/Nemotron-PII, CC BY 4.0).

Input: JSONL, one object per line: {"text": "...", "spans": [[start, end, "LABEL"], ...]}

    pip install transformers datasets accelerate optimum[onnxruntime]
    python finetune_encoder.py train.jsonl dev.jsonl --base distilbert-base-multilingual-cased
    optimum-cli export onnx --model out/ --task token-classification onnx/   # then quantise
"""
import argparse
import json

import numpy as np
from datasets import Dataset
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer, TrainingArguments)

TYPES = ["PERSON", "ORG", "LOCATION", "STREET", "ZIP", "EMAIL", "PHONE"]
TAGS = ["O"] + [f"{p}-{t}" for t in TYPES for p in ("B", "I")]
TAG2ID = {t: i for i, t in enumerate(TAGS)}


def load(path):
    return Dataset.from_list([json.loads(l) for l in open(path, encoding="utf-8")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train"); ap.add_argument("dev")
    ap.add_argument("--base", default="distilbert-base-multilingual-cased")
    ap.add_argument("--epochs", type=int, default=3)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.base)

    def encode(ex):
        enc = tok(ex["text"], truncation=True, max_length=512, return_offsets_mapping=True)
        labels = []
        for s, e in enc["offset_mapping"]:
            if s == e:                       # special tokens
                labels.append(-100); continue
            tag = "O"
            for a_, b_, lab in ex["spans"]:
                if s >= a_ and e <= b_:
                    tag = ("B-" if s == a_ else "I-") + lab
                    break
            labels.append(TAG2ID[tag])
        enc["labels"] = labels
        enc.pop("offset_mapping")
        return enc

    train = load(a.train).map(encode, remove_columns=["text", "spans"])
    dev = load(a.dev).map(encode, remove_columns=["text", "spans"])
    model = AutoModelForTokenClassification.from_pretrained(
        a.base, num_labels=len(TAGS), id2label=dict(enumerate(TAGS)), label2id=TAG2ID)

    def metrics(p):
        pred, gold = p.predictions.argmax(-1), p.label_ids
        m = gold != -100
        pii = m & (gold != 0)
        return {"token_acc": float((pred[m] == gold[m]).mean()),
                "pii_token_recall": float((pred[pii] != 0).mean()) if pii.any() else 0.0}

    Trainer(model=model,
            args=TrainingArguments("out", learning_rate=5e-5, num_train_epochs=a.epochs,
                                   per_device_train_batch_size=32, per_device_eval_batch_size=64,
                                   eval_strategy="epoch", save_strategy="epoch", warmup_ratio=0.06,
                                   weight_decay=0.01, fp16=True, report_to="none"),
            train_dataset=train, eval_dataset=dev, compute_metrics=metrics,
            data_collator=DataCollatorForTokenClassification(tok)).train()
    model.save_pretrained("out"); tok.save_pretrained("out")


if __name__ == "__main__":
    main()
