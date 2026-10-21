"""Download only the ASL Citizen clips we need, without downloading the whole 43 GB zip.

What this does, in plain words:
  1. Opens Microsoft's ASL Citizen zip file *over the internet* (zip files keep a table of
     contents at the end, so we can read single files without downloading everything).
  2. Reads the official train / val / test lists.
  3. Keeps the clips whose word is one of our 100 WLASL-100 words (e.g. BED1, BED2 -> "bed").
  4. Downloads those clips into datasets/ASL-Citizen/videos/ and writes a list (clips.csv).

Usage:
  python experiments/download_asl_citizen_subset.py                  # all clips, all splits
  python experiments/download_asl_citizen_subset.py --per-word 4 --splits test   # small sample

LICENSE: ASL Citizen (Microsoft Research) is for non-commercial research only. Do not share the
videos, put them in git, or show them in presentations. datasets/ is git-ignored for this reason.
"""
import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from remotezip import RemoteZip

URL = "https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
OUT = Path("datasets/ASL-Citizen")


def open_archive():
    # Ask for exact byte ranges; Microsoft's server sometimes rejects "last N bytes" requests.
    return RemoteZip(URL, support_suffix_range=False)


def simplify(text):
    """'BASKETBALL2' -> 'basketball2', 'NOT MIND' -> 'notmind' (letters and digits only)."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def word_for_gloss(gloss, words):
    """Return our WLASL word for an ASL Citizen gloss, or None. Numbered variants count (BED1 -> bed)."""
    if "." in gloss:  # e.g. W.H.A.T is fingerspelled, not the sign "what"
        return None
    for word in words:
        if re.fullmatch(word + r"\d*", simplify(gloss)):
            return word
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--per-word", type=int, default=0, help="max clips per word per split (0 = all)")
    args = parser.parse_args()

    words = sorted(json.loads(Path("datasets/WLASL-100/label_map.json").read_text()))
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    archive = open_archive()
    (OUT / "use.txt").write_bytes(archive.read("ASL_Citizen/use.txt"))  # keep the license next to the data

    rows = []
    for split in args.splits:
        listing = pd.read_csv(pd.io.common.BytesIO(archive.read(f"ASL_Citizen/splits/{split}.csv")))
        listing["word"] = listing["Gloss"].map(lambda g: word_for_gloss(g, words))
        listing = listing.dropna(subset=["word"]).sort_values(["word", "Participant ID", "Video file"])
        if args.per_word:
            listing = listing.drop_duplicates(["word", "Participant ID"]).groupby("word").head(args.per_word)
        for row in listing.itertuples():
            rows.append({"split": split, "label": row.word, "gloss": row.Gloss,
                         "participant": row._1, "file": row._2})
    print(f"{len(rows)} clips for {len({r['label'] for r in rows})} words")

    local = threading.local()  # each download thread keeps its own open copy of the zip

    def fetch(row):
        target = OUT / "videos" / row["file"]
        for attempt in range(5):  # the server occasionally refuses a request; just try again
            if target.is_file():
                return
            try:
                if not hasattr(local, "zf"):
                    local.zf = open_archive()
                target.write_bytes(local.zf.read("ASL_Citizen/videos/" + row["file"]))
            except Exception as error:
                print(f"retry {attempt + 1} for {row['file']}: {error}", flush=True)
                local.__dict__.pop("zf", None)
                time.sleep(2 * (attempt + 1))

    with ThreadPoolExecutor(8) as pool:  # 8 downloads at a time
        list(pool.map(fetch, rows))
    pd.DataFrame(rows).to_csv(OUT / "clips.csv", index=False)
    print(f"done -> {OUT / 'videos'}")


if __name__ == "__main__":
    main()
