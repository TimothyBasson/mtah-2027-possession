# Tracking replay

Turns any stretch of the match into an animated, top-down replay of the tracking data, as a single
HTML page you open in a browser. Use it to watch the tracking alongside the tactical video and to
see where a submission agrees or disagrees with the truth.

Needs the match data in `data/` (see [DATA.md](../DATA.md)). Standard library only, no `pip install`.

```bash
python replay/make_replay.py --start 7:33.4 --duration 40 --open
python replay/make_replay.py --start 7:33.4 --pred evaluation/results/baseline_smooth50.csv --open
```

| option | |
|---|---|
| `--start` | tactical **video** time to start at: `7:33.4`, `1:20:05`, `453.4` (seconds) or `f11335` (a video frame) |
| `--duration` | seconds to include, default 40, up to 15 minutes |
| `--pred` | a `frame,label` submission CSV; its label is shown next to the truth at every moment |
| `--gt` | ground truth CSV, default the public 5 minutes; `none` to hide it |
| `--step` | keep every n-th frame (default 3; the page interpolates in between). `1` is frame-exact |
| `--out` | output path, default `data/viz/replay_<start>_<duration>s[_<pred>].html` |
| `--open` | open the page in your default browser when done |

## What the page shows

- **Pitch:** every tracking slot as a circle, numbered with its slot ID, in team colours, with the
  ball as a small dot. Slot IDs are tracks, not players: see [DATA.md](../DATA.md).
  - **Hollow circle:** the slot has a position but no speed. Those positions are usually held or
    interpolated rather than observed. Watch the far keeper.
  - **Green ring / dashed red ring:** a slot just appeared / was just lost.
  - **Dashed grey line:** a lost slot and a same-team slot that appeared nearby shortly after,
    probably the same player under a new ID.
- **Video time:** where to seek the tactical video to watch along.
- **Game clock:** counted from the kickoff anchors in `games.py`, with stoppage time as `45+` /
  `90+`. Shows *Pre-match*, *Half-time* and *Full time* outside play.
- **True possession:** from the ground truth. Only the public 5 minutes (video 7:27.4 to 12:27.4)
  has one, so elsewhere it says *No ground truth here*.
- **Your estimate:** the label in your `--pred` file at that frame, marked ✓/✗ against the truth.
- **Strip under the pitch:** possession across the whole clip, with truth on top and your file
  below. Click it to jump.

Keyboard: space to play or pause.

## The output contains match data

The page embeds tracking positions. It is written under `data/`, which git ignores. Don't commit
it or share it outside the team, because the data terms cover derived data too.

Pages open straight from disk in a normal browser. If a page shows only a blank pitch in an
embedded viewer that blocks scripts on local files, serve the folder instead:

```bash
python3 -m http.server 8765 --directory data/viz
```
