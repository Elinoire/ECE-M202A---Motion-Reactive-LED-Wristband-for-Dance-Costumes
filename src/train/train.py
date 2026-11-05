"""Window the collected data, extract features, and test whether a simple
classifier separates the movements.

Usage:  python train.py
Needs:  pip install pandas numpy scikit-learn
"""
import glob

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

SAMPLE_HZ = 89
WIN = 50          # 0.5 s
STRIDE = 25       # 50% overlap
TRIM = 100        # drop first and last 1 s of each recording (start/stop fumbling)


def features(w):
    """w: (WIN, 6) array of ax ay az gx gy gz."""
    acc_mag = np.linalg.norm(w[:, :3], axis=1)
    gyro_mag = np.linalg.norm(w[:, 3:], axis=1)
    x = np.column_stack([w, acc_mag, gyro_mag])          # 8 channels
    spec = np.abs(np.fft.rfft(x - x.mean(0), axis=0))
    dom_freq = (spec[1:].argmax(0) + 1) * SAMPLE_HZ / len(x)
    return np.concatenate([
        x.mean(0),
        x.std(0),
        x.min(0),
        x.max(0),
        np.abs(np.diff(x, axis=0)).mean(0),              # roughness / jerk-like
        (x ** 2).mean(0),                                # energy
        dom_freq,
    ])


X, y, groups = [], [], []
files = sorted(glob.glob("data/*.csv"))
if not files:
    raise SystemExit("No CSVs in data/. Run collect.py first.")

for path in files:
    df = pd.read_csv(path)
    arr = df[["ax", "ay", "az", "gx", "gy", "gz"]].to_numpy()[TRIM:-TRIM]
    label, session = df["label"].iloc[0], df["session"].iloc[0]
    for start in range(0, len(arr) - WIN + 1, STRIDE):
        X.append(features(arr[start:start + WIN]))
        y.append(label)
        groups.append(session)

X, y, groups = np.array(X), np.array(y), np.array(groups)
print(f"{len(X)} windows from {len(files)} files")
print("Windows per class:", dict(zip(*np.unique(y, return_counts=True))))
print("Sessions:", sorted(set(groups)))

if len(set(groups)) < 2:
    raise SystemExit("Need at least 2 sessions (--session s1, --session s2) for an honest test.")

# Leave-one-session-out: train on some sessions, test on a session the model never saw.
# (Random splits would leak: neighboring overlapping windows end up in both train and test.)
clf = RandomForestClassifier(n_estimators=100, random_state=0)
pred = cross_val_predict(clf, X, y, groups=groups, cv=LeaveOneGroupOut())

labels = sorted(set(y))
print("\n", classification_report(y, pred, digits=3))
print("Confusion matrix (rows = true, cols = predicted):")
print("labels:", labels)
print(confusion_matrix(y, pred, labels=labels))

clf.fit(X, y)
order = np.argsort(clf.feature_importances_)[::-1][:8]
print("\nTop feature indices by importance:", order.tolist())
