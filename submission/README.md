# submission/

Everything that turns "a method we like" into the three deliverables in [SUBMISSION.md](../SUBMISSION.md).
Nothing here is final. The method, the team name and the document can all change until the form closes
(**17 November 2026**).

| file | what it is | change it when |
|---|---|---|
| `submission.json` | team name, which method, and the command for each method | anything changes; this is the only place the choice lives |
| `make_submission.py` | checks inputs exist and are not empty or iCloud-offloaded, runs the method, validates with the organisers' checker, refuses a degenerate file (almost all D, or a team missing), writes `out/<team>.csv` + a manifest | rarely; add methods in the json instead |
| `audit_public_repo.py` | searches all of git history for data, ground truth and label files | run before making the repo public |
| `TECHNICAL_DOC.md` | draft of the technical document, six required sections | as decisions close |
| `out/` | built files. Git-ignored: a submission is a full-match label file | never commit |

## Use

```bash
python submission/make_submission.py --list            # the configured methods, * = current choice
python submission/make_submission.py                   # build the current choice
python submission/make_submission.py --method deep     # try another without editing the json
python submission/make_submission.py --twice           # prove a rerun gives the same bytes
python submission/audit_public_repo.py                 # history leak check
```

To add a method, add an entry under `methods` in `submission.json`. A command can use `{python}` (the
interpreter running the script) and `{workdir}` (an empty scratch folder), and must write its CSV into
`{workdir}` under the name in `produces`.

## Checklist before the form

Team and accounts
- [x] Team name set in `submission.json`: **Gambtt**
- [ ] Confirm it matches the registration form exactly (spelling and case)
- [ ] Whoever uploads has a Google account and has opened the form link (sent by email)
- [ ] Team has 3 to 5 members (needed for the Innovation Challenge and the Final Pitch)

Method
- [ ] `method` set in `submission.json`
- [ ] Choice justified on the outside matches, **not** on the public 5 minutes (RULES.md, EVALUATION.md)

Reproducibility
- [ ] One environment pinned: Python version + requirement files for the chosen method (README/doc say which)
- [ ] All code and model files committed; `make_submission.py` prints no "uncommitted changes" warning
- [ ] `python submission/make_submission.py --twice` passes (same machine only: necessary, not sufficient)
- [ ] Fresh clone in a fresh venv with the pinned versions + data bundle + documented commands give the same sha256 as the manifest; that sha256 goes in the doc
- [ ] `deep` only: torch version and thread count pinned (CPU kernels can change the bytes)

Public repository
- [ ] `git fetch origin` then `python submission/audit_public_repo.py` is clean (it only sees branches fetched locally), or history has been cleaned / squashed into a fresh repo
- [ ] Repo and data out of iCloud's reach (Desktop is synced; offloaded inputs read as empty -- `make_submission.py` stops on them, other scripts do not)
- [ ] Root README explains this is our submission and how to reproduce it (it is still the organisers' README)
- [ ] Repo made public, URL opened in a logged-out browser

Upload
- [ ] Validate the **exact** file being uploaded: `python evaluation/validate_submission.py submission/out/<team>.csv`
- [ ] Files named `<team>.csv` and `<team>.pdf` (or `.md`)
- [ ] Technical document's TODOs gone; repo URL and commit filled in
- [ ] Form submitted; one response per team (it can be edited until the deadline)
