"""Cut Wikipedia articles into paragraph snippets for teacher labelling.

    python scripts/wiki_snippets.py data/wiki/20231101.de/train-00003-of-00020.parquet data/wiki_de.txt --n 5500

Input: a parquet file of the wikimedia/wikipedia dataset on Hugging Face (CC BY-SA 4.0).
Output: one paragraph per line (150 to 500 characters). About 60% are paragraphs that talk about companies,
products, media or web services, because suffix-less brand names are what the tagger misses most.
"""
import argparse
import random
import re

import pyarrow.parquet as pq

BUSINESS = re.compile(
    r"\b(GmbH|AG|Unternehmen|Firma|Konzern|Verlag|Agentur|Start-?up|gegründet|Gründer|Geschäftsführer|Hersteller|Marke|"
    r"Tochtergesellschaft|Anbieter|Plattform|Website|Online-|Zeitung|Zeitschrift|Magazin|Sender|Software|Blog|"
    r"company|founded|founder|startup|CEO|Inc\.|Ltd|corporation|acquired|subsidiary|brand|publisher|newspaper|magazine|"
    r"website|platform|software|agency|retailer|manufacturer|launched|investor|venture)\b")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet"); ap.add_argument("out"); ap.add_argument("--n", type=int, default=5500)
    a = ap.parse_args()
    rng = random.Random(7)
    biz, other = [], []
    pf = pq.ParquetFile(a.parquet)
    for batch in pf.iter_batches(batch_size=2000, columns=["text"]):
        for text in batch.column("text").to_pylist():
            paras = [p.strip() for p in text.split("\n") if 150 <= len(p.strip()) <= 500]
            for p in rng.sample(paras, min(2, len(paras))):       # at most two paragraphs per article
                if "|" in p or p.count("  ") > 3:
                    continue
                (biz if len(BUSINESS.findall(p)) >= 2 else other).append(p)
        if len(biz) > a.n * 3 and len(other) > a.n * 3:
            break
    picked = rng.sample(biz, min(len(biz), int(a.n * 0.6)))
    picked += rng.sample(other, min(len(other), a.n - len(picked)))
    rng.shuffle(picked)
    with open(a.out, "w", encoding="utf-8") as f:
        for p in picked:
            f.write(re.sub(r"\s+", " ", p) + "\n")
    print(f"{len(picked)} snippets ({len(biz)} business, {len(other)} other candidates) -> {a.out}")
