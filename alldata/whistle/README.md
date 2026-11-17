# Referee whistles from match audio: not adopted

Question (2026-11-17): can the referee's whistle in BAR-ATH's audio serve as dead-ball evidence
in the submission's decode? **No.** Two generic detectors find no whistle signal at the
submission's own stoppages, so there is nothing to feed the decode. The submission
(`alldata/learned/run.py`, sha 4979aff0) is unchanged; nothing here is wired into it.

## Data

- Both BAR-ATH videos carry a stereo 48 kHz AAC track (tactical 7,972.7 s, broadcast 6,579.5 s).
  Pulled with ffmpeg on the Mac (no re-encode of video), mono, to the cloud. The tactical track
  sits on the submission's own timeline (frame = seconds x 25). The broadcast one maps with the
  video offsets (+4014 frames first half, +26920 second); cross-correlating the two tracks'
  loudness gives +4000 / +26914, so the audio is within about 0.6 s of the video sync.
- The two tracks are different mixes (loudness correlation only 0.17 / 0.28 at the best lag).
- **No outside match has audio** (the 30 training matches and SoccerTrack are tracking only), so
  nothing can be scored held out. The only reference is the submission's own decoded stoppages
  (116 starts of 2 s or more, 115 restarts), read without ground truth: if whistles were there,
  detections should crowd around those moments.

## Detectors (generic; no setting tied to this match or to frames)

1. `spec.py` + `events.py`: tone prominence. Per 10 ms, the strongest bin in 1.8-4.6 kHz against
   the median of its flanks (150-600 Hz either side); events = runs >= 0.15 s above a threshold.
2. `tag.py`: a pretrained AudioSet tagger, EfficientAT `mn10_as` (MobileNetV3, mAP 0.471, weights
   from the authors' GitHub release), class "Whistle", on 1 s windows every 0.5 s. The tagger
   returns nonsense on clips under about 5 s (every class near 1), so each 1 s window is tiled
   five times; on EfficientAT's own example clip that gives sensible tags (speech, train).
   About 4 minutes per track on 4 CPU cores.

## Results

`agree.py`: detections near a decoded stoppage start (-2 to +3 s), against random graded frames (0.09).

| detector | track | detections | near a stoppage start | stoppages with one |
|---|---|---|---|---|
| tone, 14 dB, 0.15 s | tactical | 3,998 | 0.09 (chance) | 0.94 (fires everywhere) |
| tone, 20 dB, 0.15 s | tactical | 859 | 0.08 | 0.35 |
| tone, 24 dB, 0.15 s | broadcast | 209 | 0.15 | 0.28 |
| tone, 28 dB, 0.15 s | broadcast | 46 | 0.22 | 0.11 |
| Whistle >= 0.005 | tactical | 104 | 0.10 | 0.09 |
| Whistle >= 0.02 | tactical | 17 | 0.18 | 0.03 |
| Whistle >= 0.005 | broadcast | 178 | 0.10 | 0.15 |
| Whistle >= 0.02 | broadcast | 10 | 0.30 | 0.03 |

Where a threshold reaches 2-3x chance it fires at only a handful of stoppages (3-11%), so it
could change at most a few stoppage edges, and with 10-46 detections that ratio is noise.

`bump.py`: averaged around all 116 stoppage starts and 115 restarts, from -6 s to +6 s, the share
of tonal hops, the strongest tone and the Whistle probability are flat on both tracks (no
half-second bin stands clear of the rest; tactical Whistle x1000 peaks at 1.37 at -1 s against
0.65 overall, one bin, and the broadcast shows nothing). A whistle at a real share of stoppages
would show as a clear bump near 0 s.

Reading: on these mixes the whistle is buried under crowd and commentary, or the referee's
whistle is not picked up, and what stands out in 2-4.6 kHz is other tones. Rival entries that
used whistles may have had a different detector or a different reference; nothing here
supports it on this audio.

Not tried (would need more than a quick run): a whistle detector trained on labelled whistles
from other matches' audio (we have none), and source separation before detection.

## Files

Features and tagger outputs (no audio) in `/mnt/project-files/alldata/whistle/`: `tac.npz`,
`bro.npz` (spec.py), `tac_tags.npz`, `bro_tags.npz` (tag.py, all 527 classes, float16).

    python alldata/whistle/spec.py AUDIO16k.wav OUT.npz
    python alldata/whistle/tag.py AUDIO OUT.npz --eat ../EfficientAT
    python alldata/whistle/agree.py TAGS.npz [--offset0 4014 --offset1 26920]
    python alldata/whistle/bump.py SPEC.npz TAGS.npz [--offset0 4014 --offset1 26920]

`tag.py` needs torchvision (EfficientAT's model code imports it), which the deep group does not
install; `uv pip install torchvision` matching the installed torch.
