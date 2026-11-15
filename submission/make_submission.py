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


def offloaded_inputs(paths):
    """Input files whose contents iCloud has evicted.

    Not hypothetical: with data/tactical/BAR-ATH_tactical_atd.txt evicted, starter/baseline.py read
    no ball at all and wrote an all-D file that still passed validation. Refuse instead.
    """
    found = []
    for rel in paths:
        top = os.path.join(ROOT, rel)
        files = [top] if os.path.isfile(top) else [
            os.path.join(folder, name) for folder, _, names in os.walk(top, followlinks=True)
            for name in names]
        for path in files:
            if path.endswith(".mp4"):
                continue  # video is never read by the labelling methods
            if getattr(os.stat(path), "st_flags", 0) & SF_DATALESS:
                found.append(os.path.relpath(path, ROOT))
    return found


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
                        help="run the method a second time and fail if the bytes differ")
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

    evicted = offloaded_inputs(config.get("inputs", []) + method.get("model_files", []))
    if evicted:
        sys.exit("STOP  {} input file(s) are offloaded to iCloud and would read as empty:\n  {}\n"
                 "Download them first, e.g.  brctl download <path>".format(
                     len(evicted), "\n  ".join(evicted)))

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

    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    manifest = {
        "team_name": team,
        "file": os.path.basename(final_csv),
        "sha256": digests[0],
        "reproduced_twice": args.twice,
        "method": method_name,
        "method_description": method.get("description"),
        "command": method["cmd"],
        "requirements": method.get("requirements", []),
        "model_files": {path: sha256(os.path.join(ROOT, path))
                        for path in method.get("model_files", [])},
        "git_commit": commit,
        "git_dirty": dirty,
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
        print("WARN  uncommitted changes to tracked files: the repo at {} would not regenerate this "
              "file. Commit first for the real one.".format(commit[:7] if commit else "?"))


if __name__ == "__main__":
    main()
