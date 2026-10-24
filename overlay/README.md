# Possession overlay

Plays the tactical video with a small label box over it, showing the possession your submission
gives for the frame on screen and the true possession where the ground truth covers it. Nothing is
re-encoded: the page plays `data/tactical/BAR-ATH_tactical.mp4` straight from disk.

Needs the match data in `data/` (see [DATA.md](../DATA.md)). Standard library only, no `pip install`.

```bash
python overlay/make_overlay.py --pred evaluation/results/baseline_smooth50.csv --open
```

| option | |
|---|---|
| `--pred` | your `frame,label` submission CSV |
| `--gt` | ground truth CSV, default the public 5 minutes; `none` to hide it |
| `--out` | output path, default `data/viz/overlay_<pred>.html` |
| `--open` | open the page in your default browser when done |

Use the normal video controls to play, pause, seek and change speed. The labels follow the frame on
screen. Use the page's **Full screen** button rather than the video's own, because the video's
button would hide the labels.

- **Truth** only exists for the public 5 minutes, video 7:27.4 to 12:27.4. Seek to 7:27 to compare
  the two. Everywhere else it reads *No ground truth*.
- **Estimate** reads *No answer (X)* where your file says `X`.

The page finds the video by a path relative to itself, so keep it where it was written or
regenerate it with `--out`. It contains labels only, no tracking, and it is written under `data/`,
which git ignores.
