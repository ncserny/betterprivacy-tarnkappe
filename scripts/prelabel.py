"""Pre-label raw text files for hand-checking.

    python scripts/prelabel.py pages/ eval/real_v1.txt --category people-search --lang de

Reads every *.txt file in the folder (one page or snippet per file), runs the detector and writes
one file in the bracket markup of eval/challenge_v1.txt. Open it in a text editor, fix the
brackets by hand ([[P|..]] person, [[O|..]] organisation, [[L|..]] location, [[S|..]] street,
[[Z|..]] postcode, [[E|..]] email, [[T|..]] phone, [[I|..]] IBAN), then score it with
`python bench.py --testset eval/real_v1.txt`.

The pages contain real personal data: keep the output out of version control (see .gitignore).
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval.challenge import mark          # noqa: E402
from pii.detect import Detector          # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder"); ap.add_argument("out")
    ap.add_argument("--category", default="page"); ap.add_argument("--lang", default="en")
    ap.add_argument("--max-chars", type=int, default=4000, help="longer files are cut")
    a = ap.parse_args()
    det = Detector()
    files = sorted(glob.glob(os.path.join(a.folder, "*.txt")))
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("# Pre-labelled by the model. Correct by hand before using as a test set.\n")
        for path in files:
            text = open(path, encoding="utf-8", errors="replace").read()[:a.max_chars].strip()
            if text:
                f.write(f"---\n@ {a.category} | {a.lang}\n# source: {os.path.basename(path)}\n")
                f.write(mark(text, det.detect(text)) + "\n")
    print(f"wrote {len(files)} records to {a.out}")
