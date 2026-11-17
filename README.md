# Gambtt — More than a Hack 2027, Phase 1: Ball Possession

This is team **Gambtt**'s submission to [More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027)
(Metrica Sports, FC Barcelona and the Mobile World Capital Barcelona). For every frame of the match it
says who has the ball: `H` (home), `A` (away), `D` (dead ball) or `X` (not scored).

The method is a sticky hidden Markov model over the tracking data. It combines tracking clues, a learned
control classifier and two small players-only networks, decodes each half with Viterbi, and adds rules for
restarts and goalkeeper holds. The code is in [`hmm/`](hmm/), and the technical document is
[`submission/doc/Gambtt.pdf`](submission/doc/Gambtt.pdf).

## Reproduce the submission

It runs on a CPU. No GPU is needed.

1. **Data:** unpack the organisers' data bundle into `data/` (see [DATA.md](DATA.md)). The data is not in
   this repository and must not be redistributed.
2. **Environment:** Python 3.11, then install the pinned dependencies (`uv.lock` pins every version):
   ```bash
   pip install -r requirements.txt -r requirements-deep.txt
   ```
3. **Build:**
   ```bash
   python submission/make_submission.py --twice
   ```
   This runs `python -m hmm.decode` with the committed models, checks the file with the organisers'
   validator, checks that two runs give identical bytes, and writes `submission/out/Gambtt.csv` with a
   manifest of its sha256.

More detail on the build is in [submission/README.md](submission/README.md). The organisers' challenge
description is in [OVERVIEW.md](OVERVIEW.md), [SUBMISSION.md](SUBMISSION.md) and [RULES.md](RULES.md).
