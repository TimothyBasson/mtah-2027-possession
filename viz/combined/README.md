# Match viewer

The whole tactical video with the tracking drawn on it, a synced top-down pitch beside it, and
possession underneath, all driven by the video's clock. It combines [viz/overlay](../overlay/README.md)
and [viz/replay](../replay/README.md) so you can see what the data says against what actually happened.

Needs the match data in `data/` (see [DATA.md](../../DATA.md)). Standard library only, no `pip install`.

```bash
python viz/combined/make_combined.py --open                    # truth only, opens at kickoff
python viz/combined/make_combined.py --pred evaluation/results/baseline_smooth50.csv --open
python viz/combined/make_combined.py --pred my_submission.csv --start 1:20:05
```

| option | |
|---|---|
| `--match` | an outside match from `data/external/` (`idsse_J03WMX`, `metrica_1`, `skillcorner_2017461`, ...; see [external/README.md](../../external/README.md)), or `all` for BAR-ATH plus every outside match, each page with a game picker by the title: a button per set (BAR-ATH, IDSSE, Metrica, SkillCorner), then the set's matches with ◀ ▶ to step through them. Outside matches have no video: a timer drives the page, the pitch takes the video's place, and truth is the provider's label, not the challenge rule. SkillCorner's broadcast cutaways (replays, close-ups) show an empty pitch and no truth. Their chunks go to `data/viz/match_<name>/` (~45 MB each, written once). Needs numpy: run with `uv run` |
| `--pred` | a `frame,label` submission CSV; its label is shown next to the truth |
| `--start` | where the page opens: `7:27.4` (default, kickoff), `1:20:05`, `453.4` or `f11185`. Added to the URL as `#f<frame>`, which you can edit or share |
| `--gt` | ground truth CSV, default the public 5 minutes; `none` to hide it |
| `--explain` | show `possession/`'s labels and how it got them, in three levels you switch under the strips. **Simple** (default): the scores and a whole-match strip of mistakes to click through. **Why**: around the playhead, the mistakes, whether the model saw the ball (on outside matches after knocking it out to 59%, as `external/score.py --seen 0.59` does), and all clues together. **Every clue**: each clue's own lean and the guess before smoothing. Leans are log-odds, up for home or dead, down for away or in play. Pages are written as `data/viz/match[_<name>]_explain.html`. Needs numpy |
| `--deep` | show the learned model in `deep/` at work, stage by stage (below). Pages are written as `data/viz/match[_<name>]_deep.html`. Reads `data/cache/deep/v2/viz/<match>.npz`, which `python deep/explain.py [--match all]` writes where torch is installed; the page itself then only needs numpy |
| `--out` | output path, default `data/viz/match[_<pred>].html` |
| `--rebuild` | rewrite the tracking chunks (see below) |
| `--open` | open the page in your default browser when done |

## What the page shows

- **On the video:** a ring at the feet of every tracked slot, in team colours, numbered with the
  slot ID, and a yellow ring on the ball. They are projected onto the picture through that frame's
  homography. A ring off a player, or a player with no ring, is the tracking being wrong there.
  A dashed ring has no speed, so its position is probably held or interpolated.
- **Label box on the video:** true possession, your estimate, and whether the tracking has the ball.
- **Under the video, side by side:** the same frame as a top-down pitch (as in viz/replay; the
  shaded area is the part of the pitch the camera can see), the read-outs, and the legend.
- **Team spread** (checkbox by the play controls, remembered): each team's centre, a line to every
  player, and a dashed circle at the average distance from the centre, leaving out the keeper
  (ringed), with both numbers on the pitch. The team your estimate says has the ball is shaded. It
  is the who-has-the-ball clue in `possession/model.py` for this one frame; the model averages it
  over 2 s either side.
- **Dead-ball model** (checkbox by the play controls, remembered, `--explain` pages): a panel on
  the pitch with the stoppage model in `possession/model.py` for this frame. Each of its three
  player-speed and acceleration signals shows its value and how far it pushes towards dead (right, grey) or in
  play (left, green). They add up, with the starting lean, into P(dead), with the 0.5 line that
  decides. Under it are the label after the blip clean-up and P(dead) over 30 s either side.
- **Two strips under the video.** Click either one to jump, or drag it to scrub:
  - *Whole match*: possession for the entire video, for finding your way around. Dragging moves
    the playhead with the pointer.
  - *Two minutes around the playhead*: the same at a scale where single possessions are visible.
    It stays centred on the playhead, so dragging pans time like a map: drag left to go forward.

  Truth is the top row only where the ground truth has labels, which for the public file is video
  7:27.4 to 12:27.4. Elsewhere the strip shows your estimate alone.

Keyboard: space plays or pauses, ← / → step one frame, shift+← / → step one second, N notes the
moment.

## In-viewer notes

Press **N** (or **+ Note**) to pause and annotate the current frame: pick a tag from the chips
(missed stoppage, false stoppage, wrong team, change too late or early, missed short possession,
truth looks wrong, tracking problem, other) or type your own, and add a description. Tags you
type join the chips. Where the frame has ground truth that disagrees with the estimate, the form
suggests a tag from the two labels.

Notes show as yellow ticks on both strips and in the label box on the video within 2 s of their
frame. **Notes** opens the list: how many notes of each kind (click a kind to filter), and every
note, which you can click to jump to, edit or delete. **Export JSON** downloads them.

Served by `viz/combined/serve.py` (below), notes are saved to `data/viz/notes/<match>.json`, one
file per match (`BAR-ATH`, `idsse_J03WMX`, ...) shared by every page of that match, so notes taken
on one prediction's page show on the next one's. Each note keeps the frame, video time and game
clock, the prediction it was taken against (`model`), and the truth and estimate at that frame
(`H` / `A` / `D`, `X` for none). Opened from disk or another server, notes stay in that browser
(localStorage) and move to the file the next time the page is served by `serve.py`.

The page fits the window without scrolling: the video takes whatever height the strips, controls
and panel leave. Narrow or very short windows (under 900 px wide or 480 px tall) switch to a
scrolling column instead, so the video never shrinks to a thumbnail.

## The learned model at work (`--deep`)

```bash
python deep/explain.py --match all                               # once, needs torch: about a minute a match
uv run python viz/combined/make_combined.py --match all --deep --open
```

The page's estimate is the learned model's answer, and the model's six steps are on the page in
the order it works, each with what it has at the playhead:

- **Beside the pitch**, in place of the legend, the steps as a list: (1) how many players it is
  given, (2) that they are read as a group, (3) how far it looks across time, (4) its score for
  this frame as home, dead ball or away, (5) the label at each stage of the smoothing, (6) the
  possession the playhead is in, how far into it and how long it is. A last line has the ball
  guess, which the model only learned as a second task in training. Hover a step for what it does.
- **Model view** (checkbox by the play controls, remembered, on by default): the pitch as the model
  reads it. No slot numbers; an arrow on each player for his velocity (one second of travel); a
  green ring where he is speeding up and a red one where he is braking; and a dashed ring, labelled
  *ball guess*, where the model thinks the ball is, with a line to the tracked ball where the
  tracking has one. On BAR-ATH the ball guess is drawn on the video too, in pink. It is faint
  during a stoppage, because the model only learned the ball from live play.
- **Under the strips**, three levels. **Result**: the scores and a whole-match strip of mistakes to
  click through. **Steps** (default): *4 · Score per frame*, the three probabilities stacked (home
  from the bottom, away from the top, dead ball between), with a bracket for how far the
  convolutions reach either side of the playhead; *5a* the best guess for each frame alone, *5b*
  after the switching cost, *5c* after runs under a second are absorbed, which is the answer; and
  *6 · Possessions*, the answer cut into the spells that spellF1 scores. **Extras**: the mistakes
  around the playhead, the model's chance of a change of label nearby, and how far the ball guess
  is from the tracked ball.

BAR-ATH is labelled by the network trained on all 30 outside matches. An outside match is labelled
by the network that never saw its provider, so what you watch there is held out.

## How it is stored

Tracking for 199k frames is too big for one page, so the first run writes it as one-minute chunk
files under `data/viz/match/` (133 files, ~70 MB, a few seconds). Every page shares them and
loads the minutes around the playhead as you play or seek. A page itself holds only the labels,
~400 KB, so a page per submission is cheap. Chunks are only rewritten with `--rebuild`, which you
need only if the tracking data changes.

## Opening it

**From disk:** `--open`, or double-click the page. It finds the video and the chunks by paths
relative to itself, so keep it where it was written or regenerate it with `--out`.

**On localhost**, for the Claude app's built-in browser or any viewer that blocks local files:
`.claude/launch.json` has a `viz` config that runs `viz/combined/serve.py`, which serves `data/`
at `http://localhost:8765` with the range requests the video needs for seeking, and keeps the
notes (above). In the Claude app, ask Claude to start the `viz` preview, but for a long session
ask it to run the server in a Terminal tab instead: the app stops preview servers after about half
an hour, and the page then fails on the next tracking chunk it needs ("Couldn't load tracking
chunk"). Anywhere else, run it yourself (standard library only, any Python 3):

```bash
python3 viz/combined/serve.py
```

The config names `/opt/homebrew/bin/python3` because the app can't start Xcode's `python3` on a
script under `~/Desktop` (macOS privacy blocks it with "Operation not permitted"); change it if your
Python lives elsewhere. Any static server works for watching (`npx --yes http-server@14.1.1 data -p
8765 -c-1`), but then notes stay in the browser.

The generator prints the page's URL, e.g. `http://localhost:8765/viz/match_baseline_smooth50.html#f11185`.

"Couldn't load tracking chunk" with the chunk file still on disk usually means iCloud has offloaded
it (the repo under `~/Desktop` syncs to iCloud, and macOS offloads files when the disk is short):
`ls -lO data/viz/match` shows them as `dataless`, and the server can't read them back. Bring them
back with `for f in data/viz/match/*.js; do brctl download "$f"; done`, or in Finder right-click
`data/viz` and choose Keep Downloaded.
The overlay and replay pages are written to the same folder, so they open from the same server.

## The output contains match data

The chunks embed tracking positions. Everything is written under `data/`, which git ignores. Don't
commit it or share it outside the team, because the data terms cover derived data too.
