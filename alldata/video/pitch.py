"""Detections (image pixels) to pitch metres through the tactical homography, which maps the ATD's
0-1 pitch coordinates to image pixels (checked against the tracker's ball on the check frames)."""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
H_CSV = os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical_homography.csv")


def homographies():
    d = pd.read_csv(H_CSV)
    H = d[[f"h{i}{j}" for i in range(3) for j in range(3)]].to_numpy(np.float64).reshape(-1, 3, 3)
    return H, d["valid"].to_numpy().astype(bool)


def to_image(H, frames, xy01):
    p = np.einsum("nij,nj->ni", H[frames], np.c_[xy01, np.ones(len(xy01))])
    return p[:, :2] / p[:, 2:]


def to_pitch(H, frames, uv):
    Hi = np.linalg.inv(H[frames])
    p = np.einsum("nij,nj->ni", Hi, np.c_[uv, np.ones(len(uv))])
    return p[:, :2] / p[:, 2:]
