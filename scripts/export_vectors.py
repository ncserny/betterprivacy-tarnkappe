"""Export pretrained word vectors from spaCy's medium models (MIT-licensed packages)
into one compact file the runtime can use without spaCy.

    pip install spacy
    pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl
    pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_md-3.8.0/de_core_news_md-3.8.0-py3-none-any.whl
    python scripts/export_vectors.py
"""
import numpy as np
import spacy

out = {}
for lang, name in [("en", "en_core_web_md"), ("de", "de_core_news_md")]:
    nlp = spacy.load(name, disable=["tagger", "parser", "ner", "lemmatizer", "attribute_ruler"])
    v = nlp.vocab.vectors
    words, rows = [], []
    for key, row in v.key2row.items():
        try:
            w = nlp.vocab.strings[key]
        except KeyError:
            continue
        if len(w) > 40 or any(c.isspace() for c in w):
            continue
        words.append(w); rows.append(row + 1)            # row 0 = unknown word
    table = np.vstack([np.zeros((1, v.data.shape[1]), dtype=np.float32), v.data])
    table /= np.maximum(np.linalg.norm(table, axis=1, keepdims=True), 1e-6)
    out[f"{lang}_words"] = np.frombuffer("\n".join(words).encode("utf-8"), dtype=np.uint8)
    out[f"{lang}_rows"] = np.array(rows, dtype=np.int32)
    out[f"{lang}_table"] = table.astype(np.float16)
    print(lang, "keys", len(words), "table", table.shape)
np.savez_compressed("tarnkappe/model/vectors.npz", **out)
