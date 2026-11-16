"""Compatibility shim. The feature code now lives in src/train/train.py (single source of truth),
so scripts that still import wbcommon (live.py, verify_export.py, ...) see the same features,
window size and exclusions as training and export_model.py.

features(w) now returns the FULL 90-feature vector; select columns with train.group_columns()
if a script needs only the recipe's subset.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "train"))
import train as _t  # noqa: E402

SAMPLE_HZ, WIN, STRIDE, TRIM = _t.SAMPLE_HZ, _t.WIN, _t.STRIDE, _t.TRIM
features = _t.window_features


def load_windows(pattern="data/*.csv"):
    """Return raw windows (N, WIN, 6), MERGED labels (N,), sessions (N,); honours EXCLUDE."""
    files = [p for p in sorted(glob.glob(pattern)) if os.path.basename(p) not in _t.EXCLUDE]
    if not files:
        sys.exit("No CSVs found in data/. Run this from the repo root after recording data.")
    raw, labels, sessions = [], [], []
    for path in files:
        df = pd.read_csv(path)
        arr = df[["ax", "ay", "az", "gx", "gy", "gz"]].to_numpy()[TRIM:-TRIM]
        label = _t.MERGE.get(df["label"].iloc[0], df["label"].iloc[0])
        session = df["session"].iloc[0]
        for s in range(0, len(arr) - WIN + 1, STRIDE):
            raw.append(arr[s:s + WIN])
            labels.append(label)
            sessions.append(session)
    return np.array(raw), np.array(labels), np.array(sessions)


def feature_matrix(raw):
    return np.array([features(w) for w in raw])
