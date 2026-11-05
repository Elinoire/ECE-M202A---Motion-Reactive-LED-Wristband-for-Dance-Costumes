"""Shared constants and feature code for export_model.py and verify_export.py.

features() must match compute_features() in src/wb_features.h, and the
constants must match train.py and live.py.
"""
import glob
import sys

import numpy as np
import pandas as pd

SAMPLE_HZ = 89
WIN = 50
STRIDE = 25
TRIM = 100


def features(w):
    """w: (WIN, 6) array of ax ay az gx gy gz -> 56 features."""
    acc_mag = np.linalg.norm(w[:, :3], axis=1)
    gyro_mag = np.linalg.norm(w[:, 3:], axis=1)
    x = np.column_stack([w, acc_mag, gyro_mag])
    spec = np.abs(np.fft.rfft(x - x.mean(0), axis=0))
    dom_freq = (spec[1:].argmax(0) + 1) * SAMPLE_HZ / len(x)
    return np.concatenate([
        x.mean(0),
        x.std(0),
        x.min(0),
        x.max(0),
        np.abs(np.diff(x, axis=0)).mean(0),
        (x ** 2).mean(0),
        dom_freq,
    ])


def load_windows(pattern="data/*.csv"):
    """Return raw windows (N, WIN, 6), labels (N,), sessions (N,)."""
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit("No CSVs found in data/. Run this from the repo root after recording data.")
    raw, labels, sessions = [], [], []
    for path in files:
        df = pd.read_csv(path)
        arr = df[["ax", "ay", "az", "gx", "gy", "gz"]].to_numpy()[TRIM:-TRIM]
        label, session = df["label"].iloc[0], df["session"].iloc[0]
        for s in range(0, len(arr) - WIN + 1, STRIDE):
            raw.append(arr[s:s + WIN])
            labels.append(label)
            sessions.append(session)
    return np.array(raw), np.array(labels), np.array(sessions)


def feature_matrix(raw):
    return np.array([features(w) for w in raw])
