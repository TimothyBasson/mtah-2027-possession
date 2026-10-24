# Match viewer

The tactical video with the tracking drawn on it, a synced top-down pitch beside it, and possession
underneath, all driven by the video's clock. It combines [viz/overlay](../overlay/README.md) and
[viz/replay](../replay/README.md) so you can see what the data says against what actually happened.

Needs the match data in `data/` (see [DATA.md](../../DATA.md)). Standard library only, no `pip install`.

```bash
python viz/combined/make_combined.py --open                    # public 5 minutes, first 60 s
python viz/combined/make_combined.py --start 7:27.4 --duration 300 --pred evaluation/results/baseline_smooth50.csv --open
```

| option | |
|---|---|
| `--start` | tactical **video** time to start at: `7:27.4` (default, the start of the public ground truth), `1:20:05`, `453.4` or `f11185` |
| `--duration` | seconds to include, default 60, up to 10 minutes. Every frame is kept |
| `--pred` | a `frame,label` submission CSV; its label is shown next to the truth |
| `--gt` | ground truth CSV, default the public 5 minutes; `none` to hide it |
| `--out` | output path, default `data/viz/combined_<start>_<duration>s[_<pred>].html` |
| `--open` | open the page in your default browser when done |

## What the page shows

- **On the video:** a ring at the feet of every tracked slot, in team colours, numbered with the
  slot ID, and a yellow ring on the ball. They are projected onto the picture through that frame's
  homography. A ring off a player, or a player with no ring, is the tracking being wrong there.
  A dashed ring has no speed, so its position is probably held or interpolated.
- **Label box on the video:** true possession, your estimate, and whether the tracking has the ball.
- **Top-down pitch:** the same frame from above, as in viz/replay. The shaded area is the part of the
  pitch the camera can see.
- **Strip under the video:** possession across the clip, truth on top and your file below. Click it
  to seek.

Keyboard: space plays or pauses, ← / → step one frame, shift+← / → step one second. The page keeps
playback inside the clip.

## Opening it

Pages open straight from disk in a normal browser. The page finds the video by a path relative to
itself, so keep it where it was written or regenerate it with `--out`. If your viewer blocks local
files, serve `data/` with a server that supports range requests (for seeking the video):

```bash
npx http-server data -p 8765
```

then open `http://localhost:8765/viz/<page>.html`.

## The output contains match data

The page embeds tracking positions. It is written under `data/`, which git ignores. Don't commit it
or share it outside the team, because the data terms cover derived data too.
