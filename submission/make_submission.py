"""Build the submission file: run the chosen method, validate it, name it, fingerprint it.

    python submission/make_submission.py                   # method and team from submission.json
    python submission/make_submission.py --method deep     # try another method without editing the json
    python submission/make_submission.py --twice           # run it twice and check the bytes match
    python submission/make_submission.py --score-public    # also score on the public 5 minutes (sanity only)
    python submission/make_submission.py --list            # show the configured methods

Writes, under submission/out/ (git-ignored, never committed):

    <team>.csv             the file to upload
    <team>.manifest.json   what produced it: method, command, commit, library versions, sha256s

Adding a method is a new entry in submission.json, not a change here. The one contract a method
has to meet: write a frame,label CSV into {workdir} under the name given in "produces".

Standard library only, so it runs under whatever interpreter the method itself needs.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONFIG = os.path.join(HERE, "submission.json")
OUT_DIR = os.path.join(HERE, "out")
PLACEHOLDER_TEAM = "TEAM_NAME_TBD"


def load_config(path):
    with open(path) as handle:
        return json.load(handle)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(team):
    """The organisers name each scoreboard row after the file, so the file is named after the team."""
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", team.strip()).strip("_")
    if not name:
        sys.exit("team_name in submission.json is empty")
    return name


SF_DATALESS = 0x40000000  # macOS: the file's bytes were evicted to iCloud and are not on disk


def unreadable_inputs(paths):
    """Input files that are missing, empty, or whose contents iCloud has evicted.

    Not hypothetical: with data/tactical/BAR-ATH_tactical_atd.txt evicted, starter/baseline.py read
    no ball at all and wrote an all-D file that still passed validation. Refuse instead. The
    eviction check is macOS-only; missing and empty are checked everywhere, and check_labels()
    catches whatever slips past both.
    """
    found = []
    for rel in paths:
        top = os.path.join(ROOT, rel)
        if not os.path.exists(top):
            found.append("{}  (missing)".format(rel))
            continue
        files = [top] if os.path.isfile(top) else [
            os.path.join(folder, name) for folder, _, names in os.walk(top, followlinks=True)
            for name in names if not name.startswith(".")]
        if not files:
            found.append("{}  (empty folder)".format(rel))
        for path in files:
            if path.endswith(".mp4"):
                continue  # video is never read by the labelling methods
            info = os.stat(path)
            if getattr(info, "st_flags", 0) & SF_DATALESS:
                found.append("{}  (offloaded to iCloud: brctl download <path>)".format(
                    os.path.relpath(path, ROOT)))
            elif info.st_size == 0:
                found.append("{}  (0 bytes)".format(os.path.relpath(path, ROOT)))
    return found


def check_labels(path, min_alive_share):
    """Fail on a file that is valid but degenerate -- the shape a silently empty input produces.

    The organisers' validator checks the format only, so an all-D or one-team file passes it. The
    public window has about 80% of graded frames alive; a real method lands far above 20%.
    """
    counts = {}
    with open(path) as handle:
        next(handle)
        for line in handle:
            label = line.rstrip("\r\n").split(",")[1].strip().upper()
            counts[label] = counts.get(label, 0) + 1
    labelled = sum(n for label, n in counts.items() if label != "X")
    alive = counts.get("H", 0) + counts.get("A", 0)
    share = alive / labelled if labelled else 0.0
    problems = []
    if share < min_alive_share:
        problems.append("only {:.1%} of labelled frames are H or A (minimum {:.0%})".format(
            share, min_alive_share))
    for team in ("H", "A"):
        if not counts.get(team):
            problems.append("no frame is labelled {}".format(team))
    return counts, problems


def run_method(method, workdir):
    cmd = [part.format(python=sys.executable, workdir=workdir) for part in method["cmd"]]
    print("$ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode:
        sys.exit("method exited with code {}".format(result.returncode))
    produced = os.path.join(workdir, method["produces"])
    if not os.path.isfile(produced):
        sys.exit("method finished but did not write {}".format(produced))
    return cmd, produced


def validate(path):
    """The organisers' own checks, run exactly as they run them."""
    result = subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"),
                             path], cwd=ROOT)
    if result.returncode:
        os.remove(path)  # never leave a rejected file where it could be uploaded
        sys.exit("validation FAILED -- this file would score nothing")


def git(*args):
    try:
        return subprocess.run(["git"] + list(args), cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def library_versions():
    versions = {"python": platform.python_version()}
    for module in ("numpy", "pandas", "scipy", "sklearn", "torch"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:  # not installed, or not needed by this method
            pass
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--method", help="override the method named in the config")
    parser.add_argument("--team", help="override the team name in the config")
    parser.add_argument("--twice", action="store_true",
                        help="run the method a second time and fail if the bytes differ (same "
                             "machine only; a clean-clone rebuild is the real test)")
    parser.add_argument("--allow-degenerate", action="store_true",
                        help="keep a valid file even if it is almost all D or has one team missing")
    parser.add_argument("--score-public", action="store_true",
                        help="score against the public 5 minutes (a pipeline check, not a tuning signal)")
    parser.add_argument("--list", action="store_true", help="list configured methods and exit")
    args = parser.parse_args()

    config = load_config(args.config)
    methods = config["methods"]
    if args.list:
        for name, method in methods.items():
            mark = "*" if name == config.get("method") else " "
            print("{} {:<10} {}".format(mark, name, method.get("description", "")))
        return

    method_name = args.method or config["method"]
    if method_name not in methods:
        sys.exit("unknown method {!r}; configured: {}".format(method_name, ", ".join(methods)))
    method = methods[method_name]
    team = args.team or config["team_name"]
    if team == PLACEHOLDER_TEAM:
        print("NOTE  team_name is still the placeholder; the file will be named {}.csv".format(team))
    name = safe_name(team)

    bad = unreadable_inputs(config.get("inputs", []) + method.get("model_files", []))
    if bad:
        sys.exit("STOP  {} input(s) cannot be read as they are:\n  {}".format(
            len(bad), "\n  ".join(bad)))

    os.makedirs(OUT_DIR, exist_ok=True)
    final_csv = os.path.join(OUT_DIR, name + ".csv")
    runs = 2 if args.twice else 1
    digests = []
    for attempt in range(runs):
        with tempfile.TemporaryDirectory(prefix="mtah-submission-") as workdir:
            cmd, produced = run_method(method, workdir)
            digests.append(sha256(produced))
            if attempt == 0:
                shutil.copyfile(produced, final_csv)
    if len(set(digests)) > 1:
        sys.exit("NOT REPRODUCIBLE: two runs gave different files\n  " + "\n  ".join(digests))

    validate(final_csv)
    min_alive = config.get("sanity", {}).get("min_alive_share", 0.2)
    counts, problems = check_labels(final_csv, min_alive)
    if problems and not args.allow_degenerate:
        os.remove(final_csv)  # never leave a rejected file where it could be uploaded
        sys.exit("DEGENERATE  the file is valid but looks like an input was read as empty:\n  "
                 + "\n  ".join(problems) + "\n(--allow-degenerate to keep it anyway)")

    commit = git("rev-parse", "HEAD")
    # untracked counts too: an unadded module or model file would not be in the public repo
    changes = (git("status", "--porcelain") or "").splitlines()
    dirty = bool(changes)
    manifest = {
        "team_name": team,
        "file": os.path.basename(final_csv),
        "sha256": digests[0],
        "reproduced_twice": args.twice,
        "reproduced_twice_scope": "same machine and interpreter only; the real test is a clean "
                                  "clone + pinned environment giving this sha256",
        "label_counts": counts,
        "method": method_name,
        "method_description": method.get("description"),
        "command": method["cmd"],
        "requirements": method.get("requirements", []),
        "model_files": {path: sha256(os.path.join(ROOT, path))
                        for path in method.get("model_files", [])},
        "git_commit": commit,
        "git_dirty": dirty,
        "git_changes": changes,
        "libraries": library_versions(),
        "platform": platform.platform(),
        "built_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path = os.path.join(OUT_DIR, name + ".manifest.json")
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    if args.score_public:
        print("\npublic 5 minutes -- a pipeline check only; do not tune on it (RULES.md):", flush=True)
        subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "run_evaluation.py"),
                        "--gt", "public", "--results", OUT_DIR], cwd=ROOT)

    print("\n-> {}".format(os.path.relpath(final_csv, ROOT)))
    print("-> {}".format(os.path.relpath(manifest_path, ROOT)))
    print("   sha256 {}".format(digests[0]))
    if dirty:
        print("WARN  {} uncommitted or untracked file(s): the repo at {} might not regenerate this "
              "file. Commit first for the real one.".format(len(changes), commit[:7] if commit else "?"))
        for line in changes[:10]:
            print("        " + line)


if __name__ == "__main__":
    main()
