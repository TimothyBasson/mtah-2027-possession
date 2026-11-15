"""Before the repo goes public: look through ALL of git history for things that must not be in it.

    python submission/audit_public_repo.py            # every branch and tag
    python submission/audit_public_repo.py --ref main # just what one ref can reach

Read-only. It lists, never deletes. Making the repo public publishes every commit any branch or tag
can reach, not only the files you see now, so a file committed once and deleted later still counts.

What it flags:
- match data and anything under data/ except data/README.md
- private ground truth (*_possession_gt.csv, *_possession_gt_meta.json, possession_by_frame.pkl, ...)
- full-match label files in evaluation/results/ other than the organisers' baseline_*.csv
  (a submission IS a full-match label file; the organisers ship that folder empty on purpose)
- video, PDFs, pickles, and any blob over --max-mb (default 5 MB)
- frame-level derived exports (*.parquet, *.feather, *.npz/*.npy outside hmm/models and deep/models)
- by CONTENT, wherever the file sits: any csv/tsv/json/txt whose header is frame,label or that has
  more than 100k lines (except the organisers' shipped baselines and public ground truth)

Model weights under hmm/models and deep/models are listed as expected, not flagged.

Exit code 1 if anything is flagged, so it can sit in a checklist or CI.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RULES = [
    ("match data", lambda p: p.startswith("data/") and p != "data/README.md"),
    ("private ground truth", lambda p: any(fnmatch.fnmatch(os.path.basename(p), g) for g in (
        "*_possession_gt.csv", "*_possession_gt_meta.json", "*_possession_gt_reference.csv",
        "possession_by_frame.pkl"))),
    ("full-match labels in results/", lambda p: p.startswith("evaluation/results/")
        and p.endswith(".csv") and not os.path.basename(p).startswith("baseline_")),
    ("submission output", lambda p: p.startswith("submission/out/")),
    ("video / container", lambda p: p.endswith((".mp4", ".mov", ".avi", ".gzip", ".pl3container"))),
    ("pdf", lambda p: p.endswith(".pdf")),
    ("pickle", lambda p: p.endswith((".pkl", ".pickle"))),
    ("array export outside model folders", lambda p: p.endswith((".npz", ".npy", ".parquet", ".feather"))
        and not p.startswith(("hmm/models/", "deep/models/"))),
]


# Label-shaped files the organisers ship on purpose (README.md, .gitignore).
SHIPPED_LABELS = ("evaluation/results/baseline_", "evaluation/gt/")
TEXT_EXTS = (".csv", ".tsv", ".json", ".txt", ".jsonl")
MAX_LINES = 100_000  # a frame-level export of this match has ~199k rows
# Committed on purpose, listed (not flagged) so that a stray one stands out.
EXPECTED = ("model weights", lambda p: p.endswith((".pt", ".pth", ".onnx", ".npz", ".joblib"))
            and p.startswith(("hmm/models/", "deep/models/")))


def label_shaped(blobs):
    """(sha -> reason) for text blobs whose CONTENTS look like frame-level data, wherever they sit.

    Paths alone miss a submission saved as hmm/out/foo.csv or a notebook dump in logs/.
    """
    if not blobs:
        return {}
    out = subprocess.run(["git", "cat-file", "--batch"], cwd=ROOT, input="\n".join(blobs).encode(),
                         capture_output=True, check=True).stdout
    found, pos = {}, 0
    while pos < len(out):
        header_end = out.index(b"\n", pos)
        sha, _, size = out[pos:header_end].decode().split()
        body = out[header_end + 1:header_end + 1 + int(size)]
        pos = header_end + 1 + int(size) + 1
        first = body.split(b"\n", 1)[0].strip().lower().replace(b" ", b"")
        if first.startswith(b"frame,label") or first.startswith(b"frame\tlabel"):
            found[sha] = "header frame,label"
        elif body.count(b"\n") > MAX_LINES:
            found[sha] = "{} lines".format(body.count(b"\n"))
    return found


def git(*args):
    return subprocess.run(["git"] + list(args), cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", help="audit only what this ref reaches (default: --all)")
    parser.add_argument("--max-mb", type=float, default=5.0)
    args = parser.parse_args()

    scope = [args.ref] if args.ref else ["--all"]
    # every (blob, path) reachable from the scope, each blob once
    listing = git("rev-list", "--objects", *scope).splitlines()
    paths = {}
    for line in listing:
        sha, _, path = line.partition(" ")
        if path:
            paths.setdefault(path, set()).add(sha)

    sizes = {}
    shas = sorted({s for group in paths.values() for s in group})
    if shas:
        out = subprocess.run(["git", "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
                             cwd=ROOT, input="\n".join(shas), capture_output=True, text=True,
                             check=True).stdout
        for row in out.splitlines():
            sha, kind, size = row.split()
            if kind == "blob":
                sizes[sha] = int(size)

    text_blobs = sorted({s for path, group in paths.items() if path.endswith(TEXT_EXTS)
                         and not path.startswith(SHIPPED_LABELS) for s in group if s in sizes})
    shaped = label_shaped(text_blobs)

    flagged, expected = {}, []
    for path, group in sorted(paths.items()):
        if not any(s in sizes for s in group):
            continue  # a tree (folder), not a file
        if EXPECTED[1](path):
            expected.append("{}  ({:.1f} MB)".format(path, max(sizes.get(s, 0) for s in group) / 1e6))
        for sha in group:
            if sha in shaped:
                flagged.setdefault("frame-level contents", []).append(
                    "{}  ({}, blob {})".format(path, shaped[sha], sha[:10]))
        for reason, rule in RULES:
            if rule(path):
                flagged.setdefault(reason, []).append(path)
        biggest = max(sizes.get(s, 0) for s in group)
        if biggest > args.max_mb * 1e6:
            flagged.setdefault("blob over {:g} MB".format(args.max_mb), []).append(
                "{}  ({:.1f} MB)".format(path, biggest / 1e6))

    print("audited {} paths reachable from {} ({} text blobs read for contents)".format(
        len(paths), " ".join(scope), len(text_blobs)))
    if expected:
        print("\nexpected, not flagged -- {} committed on purpose; check nothing here is a stray:".format(
            EXPECTED[0]))
        for item in expected:
            print("      " + item)
    if not flagged:
        print("OK    nothing flagged")
        return
    for reason, items in flagged.items():
        print("\nFLAG  {} ({})".format(reason, len(items)))
        for item in items[:50]:
            print("      " + item)
        if len(items) > 50:
            print("      ... and {} more".format(len(items) - 50))
    print("\nTo find the commits behind a path:  git log --all --oneline -- <path>")
    print("Cleaning history is a rewrite (git filter-repo) or a fresh repo with one squashed commit;")
    print("either way, decide before making the repo public, not after.")
    sys.exit(1)


if __name__ == "__main__":
    main()
