"""Run a stock COCO detector (Ultralytics YOLO11s, class 32 "sports ball", no fine-tuning) over
the tactical video at the frames targets.py lists. Writes every candidate box per frame to
data/cache/video/det_<chunk>.npz: frame, cx, cy, w, h, conf (image pixels).

    python alldata/video/detect.py VIDEO.mp4 [VIDEO.mp4 ...]

The video can be the whole tactical file or pieces of it cut with `ffmpeg -c copy -copyts`
(timestamps kept), so the frame index is round(pts * 25)."""
import json, os, subprocess, sys
import numpy as np
import cv2
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
CACHE = os.path.join(ROOT, "data", "cache", "video")
MODEL, IMGSZ, CONF, FPS = "yolo11s.pt", 1920, 0.05, 25


def first_frame(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+#1",
                          "-show_entries", "frame=pts_time", "-of", "json", path], capture_output=True, text=True).stdout
    return int(round(float(json.loads(out)["frames"][0]["pts_time"]) * FPS))


def run(path, model, want):
    tag = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(CACHE, f"det_{tag}.npz")
    if os.path.exists(out):
        return
    f0 = first_frame(path)
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    todo = set(int(f) for f in want if f0 <= f < f0 + n)
    rows, last = [], max(todo) if todo else -1
    f = f0
    while todo and f <= last:
        if f in todo:
            ok, img = cap.read()
            if not ok:
                break
            r = model.predict(img, imgsz=IMGSZ, conf=CONF, classes=[32], verbose=False)[0]
            for b, c in zip(r.boxes.xywh.tolist(), r.boxes.conf.tolist()):
                rows.append([f, *b, c])
            rows.append([f, np.nan, np.nan, np.nan, np.nan, np.nan]) if len(r.boxes) == 0 else None
        elif not cap.grab():
            break
        f += 1
    np.savez(out, det=np.array(rows, np.float32).reshape(-1, 6), first=f0, n=n)
    print(tag, f0, n, len(todo), "frames scanned", flush=True)


def main():
    t = np.load(os.path.join(CACHE, "targets.npz"))
    want = np.union1d(t["gap"], t["val"])
    model = YOLO(MODEL)
    for p in sys.argv[1:]:
        run(p, model, want)


if __name__ == "__main__":
    main()
