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
| `--pred` | a `frame,label` submission CSV; its label is shown next to the truth |
| `--start` | where the page opens: `7:27.4` (default, kickoff), `1:20:05`, `453.4` or `f11185`. Added to the URL as `#f<frame>`, which you can edit or share |
| `--gt` | ground truth CSV, default the public 5 minutes; `none` to hide it |
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
- **Two strips under the video**, both click-to-seek:
  - *Whole match*: possession for the entire video, for finding your way around.
  - *Two minutes around the playhead*: the same at a scale where single possessions are visible.

  Truth is the top row only where the ground truth has labels, which for the public file is video
  7:27.4 to 12:27.4. Elsewhere the strip shows your estimate alone.

Keyboard: space plays or pauses, ← / → step one frame, shift+← / → step one second.

The page fits the window without scrolling: the video takes whatever height the strips, controls
and panel leave. Narrow or very short windows (under 900 px wide or 480 px tall) switch to a
scrolling column instead, so the video never shrinks to a thumbnail.

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
`.claude/launch.json` has a `viz` config that serves `data/` at `http://localhost:8765`, with the
range requests the video needs for seeking. In the Claude app, ask Claude to start the `viz`
preview. Anywhere else, run the same command yourself (needs Node):

```bash
npx --yes http-server@14.1.1 data -p 8765 -c-1
```

The generator prints the page's URL, e.g. `http://localhost:8765/viz/match_baseline_smooth50.html#f11185`.
The overlay and replay pages are written to the same folder, so they open from the same server.

## The output contains match data

The chunks embed tracking positions. Everything is written under `data/`, which git ignores. Don't
commit it or share it outside the team, because the data terms cover derived data too.
