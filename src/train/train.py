"""Train ONE model on all orientations pooled and measure how well it generalizes,
comparing several feature / training recipes.

Evaluation: leave-one-session-out (LOSO). Train on all sessions but one, test on the
held-out one. This approximates an orientation / wearer the model never saw, so it is
the number that matters for a wearable. Every recipe is reported twice:
  * raw       - one forest prediction per window
  * soft vote - what the board shows: EMA of tree-vote fractions with a switch margin,
                a hold time and a fast switch (SOFT_* below; keep in sync with main.cpp)

Recipes (CONFIGS below):
  base+shape            base + shape features
  +gravity              adds gravity-frame features
  +gravity+aug          same, training data also gets rotated copies
  +gravity+phase+aug    also adds phase features (needs extra DFT work on the board)

Diagnostics printed for the best recipe (by raw LOSO):
  * which ORIGINAL label (leftright vs frontback) is confused with circle_x
  * the weakest sessions' confusion matrices and the worst individual recordings
  * FLICKER: majority vote vs soft vote, as accuracy and flips/min. Each recording
    holds ONE gesture, so every label change is flicker.
  * what accuracy would be if horizontal and circle_x were ONE class
  * top feature importances

Labels leftright + frontback are merged into "horizontal" (MERGE) for training, but the
original names are kept for the diagnostics.

Other scripts import from this file: tools/export_model.py (CONFIGS, group_columns,
load_data, make_model, FEATURE_NAMES, SMOOTH, ...) and tools/wbcommon.py (window_features,
EXCLUDE, MERGE, ...). Keep those names.

Usage (repo root):  python3 src/train/train.py
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------------- settings
SAMPLE_HZ = 89
WIN = 80
STRIDE = 25
DOM_BINS = 12
TRIM = 100
SMOOTH = 5                       # old majority-vote size: baseline in the flicker table, read by export_model.py
N_WEAK_SESSIONS = 3
N_WORST_FILES = 10
TRAIN_GLOB = "data/*.csv"
EPS = 1e-9

# Soft vote settings (must match the constants in main.cpp).
# HOLD is counted in predictions: one prediction = STRIDE / SAMPLE_HZ = 0.28 s.
SOFT_ALPHA = 0.6
SOFT_MARGIN = 0.3
SOFT_MIN_SCORE = 0.35
SOFT_HOLD = 2
# Fast switch: if the new class is already clearly ahead, switch without waiting
# for the hold or the full margin. None disables it (the sweep compares both).
SOFT_FAST_SCORE = 0.55
SOFT_FAST_MARGIN = 0.15

# Augmentation. Yaw is kept SMALL on purpose: circle_x vs circle_y vs diagonal
# may depend on direction in the horizontal plane, and a full 360 deg yaw
# rotation could blur them. Raise YAW_DEG only if LOSO says it helps.
AUG_COPIES = 2
YAW_DEG = 30
TILT_DEG = 15
SEED = 0

# Labels to merge. Anything not listed keeps its own name.
MERGE = {"leftright": "horizontal", "frontback": "horizontal"}

# Session regrouping (used only for LOSO grouping and the per-session reports,
# never as a feature). Horizontal data exists only in s1-s6, so this shifts every
# frontback file forward by 6 session numbers: leftright covers s1-s6, frontback
# covers s7-s12, and every session has one file per class. {} = sessions as recorded.
SESSION_SHIFT = {"frontback": 6}

# Recordings to leave out entirely (file names, not paths). These four were
# 82-99% misclassified in LOSO and look mislabeled or recorded differently.
# Re-record rather than quietly dropping data. Set to set() to use every file.
EXCLUDE = {
    "circle_x_s11_1794710971.csv",
    "frontback_s6_1794701141.csv",
    "updown_s6_1794701021.csv",
    "frontback_s4_1794698684.csv",
}

# Feature column groups (sizes are asserted at load time).
GROUP_SIZES = {"base": 56, "shape": 13, "grav": 13, "phase": 8}
DROP_PREFIXES = ("domfreq_",)   # features left out of every recipe; () brings them back
CONFIGS = [
    # name,                  feature groups,                       augment?
    ("base+shape",           ("base", "shape"),                    False),
    ("+gravity",             ("base", "shape", "grav"),            False),
    ("+gravity+aug",         ("base", "shape", "grav"),            True),
    ("+gravity+phase+aug",   ("base", "shape", "grav", "phase"),   True),
]


def make_model():
    # Same size as the exported model, so these numbers match the board.
    return RandomForestClassifier(n_estimators=50, max_depth=12, min_samples_leaf=2,
                                  class_weight="balanced", random_state=0, n_jobs=-1)


def vote_frac(clf, X):
    """Fraction of trees voting for each class (what the firmware computes).

    Each tree casts one hard vote, like treeVotes() in main.cpp. This differs
    slightly from sklearn's predict_proba, which averages leaf probabilities.
    """
    votes = np.stack([e.predict(X) for e in clf.estimators_]).astype(int)  # (trees, n)
    return np.stack([(votes == c).mean(0) for c in range(len(clf.classes_))], axis=1)


# ---------------------------------------------------------------- features
def base_features(w):
    """56 features: mean/std/min/max/roughness/energy/dominant freq on 8 channels."""
    acc_mag = np.linalg.norm(w[:, :3], axis=1)
    gyro_mag = np.linalg.norm(w[:, 3:], axis=1)
    x = np.column_stack([w, acc_mag, gyro_mag])
    spec = np.abs(np.fft.rfft(x - x.mean(0), axis=0))
    dom_freq = (spec[1:DOM_BINS + 1].argmax(0) + 1) * SAMPLE_HZ / len(x)
    return np.concatenate([
        x.mean(0), x.std(0), x.min(0), x.max(0),
        np.abs(np.diff(x, axis=0)).mean(0),
        (x ** 2).mean(0),
        dom_freq,
    ])


def shape_features(w):
    """13 cross-axis features describing the SHAPE of the motion.

    Normalized covariance eigenvalues (one dominant = line, two similar =
    circle), pairwise axis correlations, and |cos| between the main accel
    axis and gravity.
    """
    a = w[:, :3] - w[:, :3].mean(0)
    g = w[:, 3:] - w[:, 3:].mean(0)
    feats = []
    for s in (a, g):
        cov = np.cov(s.T) + EPS * np.eye(3)
        evals = np.linalg.eigvalsh(cov)[::-1]            # largest first
        feats += list(evals / evals.sum())
        std = np.sqrt(np.diag(cov))
        c = cov / np.outer(std, std)
        feats += [c[0, 1], c[0, 2], c[1, 2]]
    cov_a = np.cov(a.T) + EPS * np.eye(3)
    _, evecs = np.linalg.eigh(cov_a)
    grav = w[:, :3].mean(0)
    grav = grav / (np.linalg.norm(grav) + EPS)
    feats.append(abs(evecs[:, -1] @ grav))
    return np.array(feats)


def _gravity_frame(w):
    """Split motion into a vertical part and a horizontal-plane part.

    gravity direction = direction of the window's mean accel.
    Returns: vertical accel (demeaned), accel along the main horizontal axis,
    horizontal covariance eigenvalues (ascending), gyro spin about gravity,
    gyro magnitude perpendicular to gravity.
    """
    a, g = w[:, :3], w[:, 3:]
    gh = a.mean(0)
    gh = gh / (np.linalg.norm(gh) + EPS)
    av = a @ gh
    av_d = av - av.mean()
    ah = a - np.outer(av, gh)                       # horizontal-plane accel
    ah_d = ah - ah.mean(0)
    evals, evecs = np.linalg.eigh(ah_d.T @ ah_d / len(ah_d))
    ap = ah_d @ evecs[:, -1]                        # main horizontal direction
    spin = g @ gh
    gperp = np.linalg.norm(g - np.outer(spin, gh), axis=1)
    return av_d, ap, evals, spin, gperp


def gravity_features(w):
    """13 gravity-frame features (invariant to rotation about gravity)."""
    av_d, ap, evals, spin, gperp = _gravity_frame(w)
    e1, e2 = max(evals[-1], 0.0), max(evals[-2], 0.0)
    var_v = av_d.var()
    corr_vh = abs(av_d @ ap) / (np.linalg.norm(av_d) * np.linalg.norm(ap) + EPS)
    return np.array([
        np.sqrt(var_v),                      # std of vertical accel
        np.sqrt(e1 + e2),                    # std of horizontal accel
        var_v / (var_v + e1 + e2 + EPS),     # fraction of motion that is vertical
        e1 / (e1 + e2 + EPS),                # horizontal linearity (1 = line)
        spin.mean(), spin.std(), np.abs(spin).mean(),
        gperp.mean(), gperp.std(),
        corr_vh,                             # vertical/horizontal in-phase-ness
        np.ptp(av_d), np.ptp(ap),
        np.abs(np.diff(av_d)).mean(),
    ])


def _phase(x, y):
    """|sin| and |cos| of the phase difference at the shared dominant frequency."""
    X, Y = np.fft.rfft(x), np.fft.rfft(y)
    k = (np.abs(X) ** 2 + np.abs(Y) ** 2)[1:DOM_BINS + 1].argmax() + 1
    d = np.angle(X[k] * np.conj(Y[k]))
    return [abs(np.sin(d)), abs(np.cos(d))]


def phase_features(w):
    """8 phase / periodicity features (|sin| ~ 1 means circle, ~ 0 means line)."""
    a_d = w[:, :3] - w[:, :3].mean(0)
    _, evecs = np.linalg.eigh(a_d.T @ a_d / len(a_d))
    s1, s2 = a_d @ evecs[:, -1], a_d @ evecs[:, -2]   # top two principal axes
    av_d, ap, *_ = _gravity_frame(w)
    f = _phase(s1, s2) + _phase(av_d, ap)
    p = np.abs(np.fft.rfft(s1)) ** 2
    k = p[1:DOM_BINS + 1].argmax() + 1
    f.append(p[k] / (p[1:].sum() + EPS))              # power in the main peak
    for x in (av_d, ap, s1):                          # zero-crossing rates
        f.append(np.mean(np.diff(np.signbit(x).astype(int)) != 0))
    return np.array(f)


def window_features(w):
    return np.concatenate([base_features(w), shape_features(w),
                           gravity_features(w), phase_features(w)])


def _names():
    stats = ["mean", "std", "min", "max", "rough", "energy", "domfreq"]
    chans = ["ax", "ay", "az", "gx", "gy", "gz", "amag", "gmag"]
    n = [f"{s}_{c}" for s in stats for c in chans]
    for pre in ("acc", "gyro"):
        n += [f"{pre}_eig1", f"{pre}_eig2", f"{pre}_eig3",
              f"{pre}_corr_xy", f"{pre}_corr_xz", f"{pre}_corr_yz"]
    n += ["acc_axis_vs_grav"]
    n += ["g_std_vert", "g_std_horiz", "g_vert_frac", "g_horiz_linearity",
          "g_spin_mean", "g_spin_std", "g_spin_absmean", "g_gyro_perp_mean",
          "g_gyro_perp_std", "g_corr_vert_horiz", "g_ptp_vert", "g_ptp_horiz",
          "g_rough_vert"]
    n += ["ph_pc_sin", "ph_pc_cos", "ph_vh_sin", "ph_vh_cos", "ph_peak_frac",
          "zc_vert", "zc_horiz", "zc_pc1"]
    return n


FEATURE_NAMES = _names()


def group_columns(groups):
    start, cols = 0, {}
    for name, size in GROUP_SIZES.items():
        cols[name] = np.arange(start, start + size)
        start += size
    keep = np.concatenate([cols[g] for g in groups])
    return np.array([j for j in keep if not FEATURE_NAMES[j].startswith(DROP_PREFIXES)])


# ------------------------------------------------------------ augmentation
def _rot(axis, angle):
    axis = axis / (np.linalg.norm(axis) + EPS)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def rotate_window(w, rng):
    """Small yaw about gravity plus small tilt, applied to accel AND gyro."""
    gh = w[:, :3].mean(0)
    gh = gh / (np.linalg.norm(gh) + EPS)
    yaw = np.deg2rad(rng.uniform(-YAW_DEG, YAW_DEG))
    r = rng.normal(size=3)
    r -= (r @ gh) * gh                                # axis perpendicular to gravity
    tilt = np.deg2rad(rng.uniform(-TILT_DEG, TILT_DEG))
    R = _rot(r, tilt) @ _rot(gh, yaw)
    return np.hstack([w[:, :3] @ R.T, w[:, 3:] @ R.T])


# -------------------------------------------------------------------- data
def load_data():
    rng = np.random.default_rng(SEED)
    files = sorted(glob.glob(TRAIN_GLOB))
    skipped = [p for p in files if os.path.basename(p) in EXCLUDE]
    files = [p for p in files if os.path.basename(p) not in EXCLUDE]
    if EXCLUDE:
        print(f"  excluding {len(skipped)} of {len(EXCLUDE)} listed files: "
              f"{[os.path.basename(p) for p in skipped]}")
    data = []
    for i, path in enumerate(files, 1):
        df = pd.read_csv(path)
        arr = df[["ax", "ay", "az", "gx", "gy", "gz"]].to_numpy()[TRIM:-TRIM]
        raw = df["label"].iloc[0]
        session = str(df["session"].iloc[0])
        if raw in SESSION_SHIFT:
            session = f"s{int(session.lstrip('s')) + SESSION_SHIFT[raw]}"
        wins = [arr[s:s + WIN] for s in range(0, len(arr) - WIN + 1, STRIDE)]
        if not wins:
            print(f"  skipping {path}: too short")
            continue
        F = np.array([window_features(w) for w in wins])
        Fa = np.array([[window_features(rotate_window(w, rng))
                        for _ in range(AUG_COPIES)] for w in wins])
        assert F.shape[1] == sum(GROUP_SIZES.values()) == len(FEATURE_NAMES), F.shape
        data.append({"F": F, "Faug": Fa, "raw": raw, "label": MERGE.get(raw, raw),
                     "session": session, "path": path})
        print(f"  [{i}/{len(files)}] {path}: {len(wins)} windows", end="\r")
    print(" " * 80, end="\r")
    return data


# -------------------------------------------------------------- evaluation
def _train_matrix(F, Fa, y, cols, use_aug):
    X, Y = F[:, cols], np.asarray(y)
    if use_aug and len(Fa):
        c = Fa.shape[1]
        X = np.vstack([X, Fa.reshape(-1, Fa.shape[-1])[:, cols]])
        Y = np.concatenate([Y, np.repeat(Y, c)])
    return X, Y


def loso(data, cols, use_aug):
    """Leave-one-session-out: how well does it handle an unseen orientation?

    Output order follows the file order of `data` (segment id = file index).
    Returns (y, pred, groups, segments, frac) where frac[i, c] is the fraction
    of trees voting for class c (classes in sorted order) on window i.
    """
    F = np.vstack([d["F"] for d in data])
    Fa = np.concatenate([d["Faug"] for d in data])
    y = np.concatenate([[d["label"]] * len(d["F"]) for d in data])
    g = np.concatenate([[d["session"]] * len(d["F"]) for d in data])
    seg = np.concatenate([[i] * len(d["F"]) for i, d in enumerate(data)])
    classes = np.array(sorted(set(y)))
    frac = np.zeros((len(y), len(classes)))
    for s in sorted(set(g)):
        te = g == s
        X, Y = _train_matrix(F[~te], Fa[~te], y[~te], cols, use_aug)
        clf = make_model().fit(X, Y)
        # assumes every class is in the training fold (true while each session has all classes)
        frac[te] = vote_frac(clf, F[te][:, cols])
    pred = classes[frac.argmax(1)]          # hard vote, ties -> lowest index, like the board
    return y, pred, g, seg, frac


def smooth(pred, segs, k=SMOOTH):
    """Causal majority vote over the last k predictions within each segment."""
    out = pred.copy()
    for s in np.unique(segs):
        idx = np.where(segs == s)[0]                  # contiguous, in time order
        p = pred[idx]
        for i in range(len(p)):
            vals, counts = np.unique(p[max(0, i - k + 1):i + 1], return_counts=True)
            out[idx[i]] = vals[counts.argmax()]
    return out


def soft_vote(frac, segs, alpha=SOFT_ALPHA, margin=SOFT_MARGIN,
              min_score=SOFT_MIN_SCORE, hold=SOFT_HOLD,
              fast_score=SOFT_FAST_SCORE, fast_margin=SOFT_FAST_MARGIN):
    """Same logic as vote() in main.cpp: EMA of tree vote fractions.

    Slow path: switch if the new class leads the current one by `margin`, has at
    least `min_score`, and `hold` predictions passed since the last switch.
    Fast path (if fast_score is not None): switch right away, ignoring the hold,
    if the new class has at least `fast_score` and leads by `fast_margin`.
    Returns class INDICES (into the sorted class list)."""
    out = np.empty(len(frac), int)
    for s in np.unique(segs):
        idx = np.where(segs == s)[0]
        score, cur, since = None, -1, 0
        for i in idx:
            since += 1
            score = frac[i].copy() if score is None else alpha * score + (1 - alpha) * frac[i]
            best = int(score.argmax())
            if cur < 0:
                cur, since = best, 0
            elif best != cur:
                fast = (fast_score is not None and score[best] >= fast_score
                        and score[best] >= score[cur] + fast_margin)
                slow = (since >= hold and score[best] >= min_score
                        and score[best] >= score[cur] + margin)
                if fast or slow:
                    cur, since = best, 0
            out[i] = cur
    return out


def flips_per_min(p, segs):
    """Label changes per minute. Each recording is one gesture, so all are flicker."""
    flips = sum(int((p[segs == s][1:] != p[segs == s][:-1]).sum()) for s in np.unique(segs))
    minutes = len(p) * STRIDE / SAMPLE_HZ / 60
    return flips / minutes


# ------------------------------------------------------------------ reports
def report(name, y_true, y_pred, groups, classes):
    print(f"\n--- {name} ---")
    for s in sorted(set(groups)):
        m = groups == s
        print(f"  {s:<14} acc {accuracy_score(y_true[m], y_pred[m]):.3f}")
    print(f"  overall        acc {accuracy_score(y_true, y_pred):.3f}")
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted):")
    print("labels:", [str(c) for c in classes])
    print(confusion_matrix(y_true, y_pred, labels=classes))


def diagnose_merged(raw, p, g):
    """Which original label (e.g. leftright vs frontback) is getting confused?"""
    raws = sorted(r for r in MERGE if (raw == r).any())
    if not raws:
        return
    print("\n--- merged-class diagnostic (LOSO, raw predictions) ---")
    tops = {}
    for r in raws:
        m = raw == r
        vals, counts = np.unique(p[m], return_counts=True)
        order = sorted(zip(vals, counts), key=lambda t: -t[1])
        print(f"  {r:<10} (n={m.sum()}) predicted as: "
              + ", ".join(f"{v} {c / m.sum():.0%}" for v, c in order))
        wrong = [(v, c) for v, c in order if v != MERGE[r]]
        tops[r] = wrong[0][0] if wrong else None
    print("\n  per session: % predicted correctly, and % predicted as its top wrong class")
    head = f"  {'session':<8}"
    for r in raws:
        head += f"{r + ' ok':>16}{'-> ' + (tops[r] or '-'):>16}"
    print(head)
    for s in sorted(set(g)):
        row = f"  {s:<8}"
        for r in raws:
            m = (raw == r) & (g == s)
            if not m.any():
                row += f"{'n/a':>16}{'n/a':>16}"
                continue
            ok = (p[m] == MERGE[r]).mean()
            wr = (p[m] == tops[r]).mean() if tops[r] is not None else 0.0
            row += f"{ok:>16.0%}{wr:>16.0%}"
        print(row)


def weak_sessions(y, p, g, classes):
    accs = {s: accuracy_score(y[g == s], p[g == s]) for s in sorted(set(g))}
    worst = sorted(accs, key=accs.get)[:N_WEAK_SESSIONS]
    print("\n--- weakest sessions (LOSO, raw predictions) ---")
    print("labels:", [str(c) for c in classes])
    for s in worst:
        m = g == s
        print(f"\n  {s}: acc {accs[s]:.3f}  (rows = true, cols = predicted)")
        print(confusion_matrix(y[m], p[m], labels=classes))


def worst_files(data, y, p, seg):
    rows = []
    for i, d in enumerate(data):
        m = seg == i
        wrong = p[m][p[m] != y[m]]
        if len(wrong):
            vals, counts = np.unique(wrong, return_counts=True)
            top = f"{vals[counts.argmax()]} ({counts.max() / m.sum():.0%})"
        else:
            top = "-"
        rows.append(((p[m] == y[m]).mean(), d["path"], d["raw"], d["session"], top))
    rows.sort(key=lambda r: r[0])
    print(f"\n--- {N_WORST_FILES} worst recordings (LOSO, raw predictions) ---")
    print(f"  {'acc':>5}  {'session':<8}{'true label':<12}{'mostly predicted as':<24}file")
    for acc, path, raw, session, top in rows[:N_WORST_FILES]:
        print(f"  {acc:>5.2f}  {session:<8}{raw:<12}{top:<24}{path}")


def soft_vote_sweep(y, p, frac, seg, classes):
    """Flicker (flips/min) and accuracy: raw, majority vote, soft vote variants."""
    print("\n--- flicker: majority vote vs soft vote (LOSO) ---")
    print(f"  {'method':<34}{'accuracy':>9}{'flips/min':>11}")
    print(f"  {'raw':<34}{accuracy_score(y, p):>9.3f}{flips_per_min(p, seg):>11.1f}")
    pm = smooth(p, seg)
    print(f"  {f'majority vote (k={SMOOTH}, old)':<34}"
          f"{accuracy_score(y, pm):>9.3f}{flips_per_min(pm, seg):>11.1f}")
    q = classes[soft_vote(frac, seg)]
    print(f"  {'soft vote (firmware settings)':<34}"
          f"{accuracy_score(y, q):>9.3f}{flips_per_min(q, seg):>11.1f}")
    for alpha in (0.5, 0.6, 0.7):
        for margin in (0.12, 0.2, 0.3):
            q = classes[soft_vote(frac, seg, alpha=alpha, margin=margin, fast_score=None)]
            print(f"  {f'soft a={alpha} margin={margin}':<34}"
                  f"{accuracy_score(y, q):>9.3f}{flips_per_min(q, seg):>11.1f}")
    print("  (higher alpha / margin = fewer flips but slower to react; "
          f"hold = {SOFT_HOLD} predictions, min_score = {SOFT_MIN_SCORE})")

    # Fast switch: should switch EARLIER at about the same accuracy and flips/min.
    print("\n--- fast switch on top of soft vote (LOSO) ---")
    print(f"  {'method':<34}{'accuracy':>9}{'flips/min':>11}")
    for alpha in (0.6, 0.7):
        for fs in (None, 0.65, 0.55, 0.5):
            q = classes[soft_vote(frac, seg, alpha=alpha, margin=0.3, fast_score=fs)]
            label = f"a={alpha} m=0.3 fast={fs}"
            print(f"  {label:<34}{accuracy_score(y, q):>9.3f}{flips_per_min(q, seg):>11.1f}")
    print(f"  (fast margin = {SOFT_FAST_MARGIN}; lower fast_score = earlier switches. Pick the lowest "
          "value whose accuracy and flips/min stay close to the fast=None row.)")


def merged_what_if(y, p, ps):
    f = lambda a: np.where(np.isin(a, ["horizontal", "circle_x"]), "horizontal_or_circle_x", a)
    print("\n--- what if horizontal and circle_x were ONE class? (LOSO) ---")
    print(f"  raw accuracy        {accuracy_score(f(y), f(p)):.3f}   (now {accuracy_score(y, p):.3f})")
    print(f"  soft-vote accuracy  {accuracy_score(f(y), f(ps)):.3f}   (now {accuracy_score(y, ps):.3f})")
    print("  (evaluation-only: a model trained with them merged could do slightly better)")


def main():
    if not glob.glob(TRAIN_GLOB):
        raise SystemExit("No CSVs in data/. Run collect.py first.")

    print("Computing features (including augmented copies)...")
    data = load_data()
    classes = np.array(sorted({d["label"] for d in data}))
    n_win = sum(len(d["F"]) for d in data)
    print(f"{len(data)} files | {n_win} windows | "
          f"{sum(GROUP_SIZES.values())} candidate features | classes: {list(classes)}")

    results = {}
    for name, groups, use_aug in CONFIGS:
        cols = group_columns(groups)
        print(f"\nRunning '{name}' ({len(cols)} features, aug={use_aug}) ...")
        y, p, g, seg, frac = loso(data, cols, use_aug)
        results[name] = (y, p, g, seg, frac, classes[soft_vote(frac, seg)])

    # ------------------------------------------------------------ summary
    print("\n" + "=" * 70)
    print(f"{'recipe':<22}{'LOSO':>8}{'LOSO+soft':>11}{'flips/min raw':>15}{'-> soft':>9}")
    for name, _, _ in CONFIGS:
        y, p, _, seg, _, ps = results[name]
        print(f"{name:<22}{accuracy_score(y, p):>8.3f}{accuracy_score(y, ps):>11.3f}"
              f"{flips_per_min(p, seg):>15.1f}{flips_per_min(ps, seg):>9.1f}")
    print("=" * 70)

    bname, bgroups, baug = max(CONFIGS, key=lambda c: accuracy_score(results[c[0]][0], results[c[0]][1]))
    print(f"\nBest by raw LOSO: '{bname}'. Detailed results follow.")
    y, p, g, seg, frac, ps = results[bname]
    report(f"LOSO, '{bname}'", y, p, g, classes)
    report(f"LOSO + soft vote (firmware settings), '{bname}'", y, ps, g, classes)

    # -------------------------------------------------------- diagnostics
    raw = np.concatenate([[d["raw"]] * len(d["F"]) for d in data])
    diagnose_merged(raw, p, g)
    weak_sessions(y, p, g, classes)
    worst_files(data, y, p, seg)
    soft_vote_sweep(y, p, frac, seg, classes)
    merged_what_if(y, p, ps)

    # --------------------------------------------------- feature importance
    cols = group_columns(bgroups)
    F = np.vstack([d["F"] for d in data])
    Fa = np.concatenate([d["Faug"] for d in data])
    yall = np.concatenate([[d["label"]] * len(d["F"]) for d in data])
    X, Y = _train_matrix(F, Fa, yall, cols, baug)
    imp = make_model().fit(X, Y).feature_importances_
    print(f"\nTop 15 features for '{bname}':")
    for j in np.argsort(imp)[::-1][:15]:
        print(f"  {FEATURE_NAMES[cols[j]]:<22}{imp[j]:.4f}")

    print("\nOnly port a feature group to the firmware if it helps LOSO by a clear "
          "margin (several points).")


if __name__ == "__main__":
    main()