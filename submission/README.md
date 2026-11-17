# submission/

Everything that turns "a method we like" into the three deliverables in [SUBMISSION.md](../SUBMISSION.md).
Nothing here is final. The method, the team name and the document can all change until the form closes
(**17 November 2026**).

| file | what it is | change it when |
|---|---|---|
| `submission.json` | team name, which method, and the command for each method | anything changes; this is the only place the choice lives |
| `make_submission.py` | checks inputs exist and are not empty or iCloud-offloaded, runs the method, validates with the organisers' checker, refuses a degenerate file (almost all D, or a team missing), writes `out/<team>.csv` + a manifest | rarely; add methods in the json instead |
| `out/` | built files. Git-ignored: a submission is a full-match label file | never commit |

## Use

```bash
python submission/make_submission.py --list            # the configured methods, * = current choice
python submission/make_submission.py                   # build the current choice
python submission/make_submission.py --method deep     # try another without editing the json
python submission/make_submission.py --twice           # prove a rerun gives the same bytes
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
- [x] `method` set in `submission.json`: **hmm** = hmm_v9 (`hmm_v8` kept as a no-torch fallback)
- [x] Choice justified on the outside matches, **not** on the public 5 minutes (held out 0.677, fair 0.664; external/README.md)

Reproducibility
- [x] Environment named: Python 3.11, `requirements.txt` + `requirements-deep.txt` (torch 2.15.0); `uv.lock` pins the rest
- [ ] All code and model files committed; `make_submission.py` prints no "uncommitted changes" warning
- [ ] `python submission/make_submission.py --twice` passes (same machine only: necessary, not sufficient)
- [ ] Fresh clone in a fresh venv with the pinned versions + data bundle + documented commands give the same sha256 as the manifest; that sha256 goes in the doc
- [x] torch version and thread count pinned (hmm.decode runs the nets on one thread); not yet checked on a second CPU

Public repository
- [x] Git history audited for data, ground truth and label files across all 84 branches on GitHub: clean (2026-11-17). Re-check if new branches are pushed
- [ ] Repo and data out of iCloud's reach (Desktop is synced; offloaded inputs read as empty -- `make_submission.py` stops on them, other scripts do not)
- [ ] Root README explains this is our submission and how to reproduce it (it is still the organisers' README)
- [ ] Repo made public, URL opened in a logged-out browser

Upload
- [ ] Validate the **exact** file being uploaded: `python evaluation/validate_submission.py submission/out/<team>.csv`
- [ ] Files named `<team>.csv` and `<team>.pdf` (or `.md`)
- [ ] Technical document's TODOs gone; repo URL and commit filled in
- [ ] Technical document exported from the Claude Doc as `<team>.pdf` (uploaded through the form, not committed)
- [ ] Form submitted; one response per team (it can be edited until the deadline)
