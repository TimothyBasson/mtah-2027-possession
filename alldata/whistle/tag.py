"""Run a pretrained AudioSet tagger (EfficientAT mn10_as, MobileNetV3, mAP 0.471) over a match's
audio in 1 s windows every 0.5 s and save all 527 class probabilities (float16).

    git clone --depth 1 https://github.com/fschmid56/EfficientAT ../EfficientAT
    curl -L -o ../EfficientAT/resources/mn10_as_mAP_471.pt \
        https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn10_as_mAP_471.pt
    python alldata/whistle/tag.py AUDIO OUT.npz [--eat ../EfficientAT]

AUDIO is anything ffmpeg reads (resampled to 32 kHz mono). The tagger was trained on AudioSet
(YouTube clips); nothing here is fitted to a match. The mel front end is EfficientAT's
AugmentMelSTFT in eval mode, rewritten without torchaudio (Kaldi mel banks). The
tagger gives nonsense on inputs under about 5 s (every class near 1 on a 1 s clip), so each 1 s
window is tiled 5 times; on EfficientAT's own example clip that gives sensible tags.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import torch

SR, WIN, HOP_W, TILE = 32000, 1.0, 0.5, 5
WHISTLE, WHISTLING = 402, 40


def load(path):
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def mel_banks(n_mels=128, n_fft=1024, sr=SR, fmin=0.0, fmax=SR // 2 - 1000):
    def mel(f):
        return 1127.0 * np.log(1 + np.asarray(f, float) / 700.0)
    lo, hi = mel(fmin), mel(fmax)
    d = (hi - lo) / (n_mels + 1)
    b = np.arange(n_mels)[:, None]
    left, center, right = lo + b * d, lo + (b + 1) * d, lo + (b + 2) * d
    m = mel(sr / n_fft * np.arange(n_fft // 2))[None]
    banks = np.maximum(0, np.minimum((m - left) / (center - left), (right - m) / (right - center)))
    return torch.tensor(np.pad(banks, ((0, 0), (0, 1))), dtype=torch.float32)


class Mel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(800, periodic=False))
        self.register_buffer("pre", torch.tensor([[[-0.97, 1.0]]]))
        self.register_buffer("banks", mel_banks())

    def forward(self, x):
        x = torch.nn.functional.conv1d(x.unsqueeze(1), self.pre).squeeze(1)
        x = torch.stft(x, 1024, hop_length=320, win_length=800, center=True, window=self.window,
                       return_complex=True).abs() ** 2
        return ((self.banks @ x + 1e-5).log() + 4.5) / 5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("out")
    ap.add_argument("--eat", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "EfficientAT"))
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    eat = os.path.abspath(a.eat)
    sys.path.insert(0, eat)
    cwd = os.getcwd()
    os.chdir(eat)  # its loader looks for resources/ relative to the working directory
    from models.mn.model import get_model
    model = get_model(width_mult=1.0, pretrained_name="mn10_as").eval()
    os.chdir(cwd)
    torch.set_num_threads(os.cpu_count())
    x = torch.from_numpy(load(a.audio).copy())
    w, h = int(WIN * SR), int(HOP_W * SR)
    n = (len(x) - w) // h + 1
    mel = Mel()
    out = np.zeros((n, 527), np.float16)
    with torch.no_grad():
        for s in range(0, n, a.batch):
            idx = range(s, min(n, s + a.batch))
            seg = torch.stack([x[i * h:i * h + w].repeat(TILE) for i in idx])
            p, _ = model(mel(seg).unsqueeze(1))
            out[s:s + len(idx)] = torch.sigmoid(p).numpy()
            if s % (a.batch * 50) == 0:
                print(f"{s}/{n}", flush=True)
    np.savez_compressed(a.out, p=out, start=np.arange(n) * HOP_W, win=WIN)
    print(a.out, n, "whistle p50/p99/p99.9", np.percentile(out[:, WHISTLE].astype(float), [50, 99, 99.9]))


if __name__ == "__main__":
    main()
