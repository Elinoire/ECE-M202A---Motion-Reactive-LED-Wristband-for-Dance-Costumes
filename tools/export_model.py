"""Train the final classifier with the SAME recipe as src/train/train.py and
export it as C++ for the board.

Usage (repo root, venv active):  python3 tools/export_model.py

Writes:
  src/model.h           trees + class names + default LED patterns + constants
  src/model_selftest.h  one real window per class with the expected features and
                        prediction, so the firmware can prove its C++ feature code
                        matches Python (set SELFTEST = False to skip)

Re-run whenever recordings, the EXCLUDE list, or features in train.py change,
then re-upload the firmware.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src", "train"))
import train as T  # noqa: E402  (importing is safe: main() is guarded)

OUT = "src/model.h"
OUT_TEST = "src/model_selftest.h"
SELFTEST = True

# Which recipe from T.CONFIGS to ship. Pick the one that won on LOSO.
RECIPE = "+gravity+aug"

# Solid color (r, g, b) per class at BRIGHTNESS (0-255). Change freely; the web
# app can still override them over BLE. Class names must match your labels.
BRIGHTNESS = 30
DEFAULTS = {
    "horizontal": (0, 0, 255),    # blue
    "updown":     (0, 255, 0),    # green
    "circle_x":   (255, 0, 255),  # magenta
    "circle_y":   (0, 255, 255),  # cyan
    "diagonal":   (255, 160, 0),  # orange
    "still":      (255, 255, 0),  # yellow
}
FALLBACK_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255),
                   (255, 160, 0), (0, 255, 255)]


def c_floats(values):
    out = []
    for v in values:
        s = f"{float(np.float32(v)):.9g}"
        if not any(ch in s for ch in ".en"):
            s += ".0"
        out.append(s + "f")
    return out


def c_array(name, ctype, values, per_line=10):
    lines = [" " + ", ".join(str(v) for v in values[i:i + per_line])
             for i in range(0, len(values), per_line)]
    return f"static const {ctype} {name}[{len(values)}] = {{\n" + ",\n".join(lines) + "\n};\n"


# ------------------------------------------------------------------ train
recipe = {c[0]: c for c in T.CONFIGS}.get(RECIPE)
if recipe is None:
    sys.exit(f"RECIPE {RECIPE!r} not in train.CONFIGS: {[c[0] for c in T.CONFIGS]}")
_, groups, use_aug = recipe
cols = T.group_columns(groups)

print(f"Recipe '{RECIPE}': {len(cols)} features, aug={use_aug}")
data = T.load_data()          # applies EXCLUDE, MERGE, TRIM exactly like training
F = np.vstack([d["F"] for d in data])
Fa = np.concatenate([d["Faug"] for d in data])
y = np.concatenate([[d["label"]] * len(d["F"]) for d in data])
X, Y = T._train_matrix(F, Fa, y, cols, use_aug)

clf = T.make_model().fit(X, Y)
classes = [str(c) for c in clf.classes_]
print(f"Trained on {len(X)} windows. Classes (firmware order): {classes}")

# ----------------------------------------------------------- flatten trees
# TREE_FEATURE stores the index into the FULL feature vector (T.FEATURE_NAMES),
# not the column within the selected subset. The firmware therefore always
# builds the vector in canonical order; unused groups can be left at 0.
roots, feat, thr, left, right = [], [], [], [], []
offset = 0
for est in clf.estimators_:
    t = est.tree_
    roots.append(offset)
    for i in range(t.node_count):
        if t.children_left[i] == -1:                 # leaf: class index in "left"
            feat.append(-1)
            thr.append(0.0)
            left.append(int(np.argmax(t.value[i][0])))
            right.append(0)
        else:
            feat.append(int(cols[t.feature[i]]))
            thr.append(t.threshold[i])
            left.append(offset + int(t.children_left[i]))
            right.append(offset + int(t.children_right[i]))
    offset += t.node_count
n_nodes = offset
if n_nodes >= 65536:
    sys.exit("Too many nodes for 16-bit indices. Use fewer trees or a lower max_depth.")

# ------------------- check: firmware-style hard vote vs sklearn on train data
thr32 = np.float32(thr)


def predict_like_firmware(x):
    votes = np.zeros(len(classes), int)
    x = np.float32(x)
    for r in roots:
        n = r
        while feat[n] >= 0:
            n = left[n] if x[feat[n]] <= thr32[n] else right[n]
        votes[left[n]] += 1
    return int(votes.argmax())                       # ties -> lowest index


Ffull = F[np.random.default_rng(0).choice(len(F), min(2000, len(F)), replace=False)]
fw = np.array([predict_like_firmware(r) for r in Ffull])
sk = np.searchsorted(classes, clf.predict(Ffull[:, cols]))
print(f"Hard-vote C++ logic agrees with sklearn on {np.mean(fw == sk):.1%} of "
      f"{len(Ffull)} sampled windows (small gap is expected: sklearn averages probabilities).")

# ------------------------------------------------------------- model.h
defaults = []
for i, name in enumerate(classes):
    rgb = DEFAULTS.get(name)
    if rgb is None:
        rgb = FALLBACK_COLORS[i % len(FALLBACK_COLORS)]
        print(f"  note: no DEFAULTS color for '{name}', using a fallback")
    defaults.append(f"{{0, {rgb[0]}, {rgb[1]}, {rgb[2]}, {BRIGHTNESS}}}")
names = ", ".join(f'"{c}"' for c in classes)

with open(OUT, "w") as f:
    f.write("// Generated by tools/export_model.py. Do not edit by hand.\n")
    f.write(f"// Recipe: {RECIPE}\n")
    f.write(f"// Movement order (BLE movement index): {', '.join(f'{i}={c}' for i, c in enumerate(classes))}\n")
    f.write("// Feature vector order (index: name):\n")
    for i, n in enumerate(T.FEATURE_NAMES):
        f.write(f"//   {i}: {n}{'' if i in set(cols.tolist()) else '   (unused by this recipe)'}\n")
    f.write("#pragma once\n#include <stdint.h>\n\nnamespace wb {\n\n")
    f.write(f"constexpr float MODEL_SAMPLE_HZ = {T.SAMPLE_HZ}.0f;\n")
    f.write(f"constexpr int MODEL_WIN = {T.WIN};        // samples per window\n")
    f.write(f"constexpr int MODEL_STRIDE = {T.STRIDE};     // new prediction every N samples\n")
    f.write(f"constexpr int MODEL_SMOOTH = {T.SMOOTH};      // causal majority vote over last N predictions\n")
    f.write(f"constexpr int DOM_BINS = {T.DOM_BINS};\n")
    f.write(f"constexpr bool MODEL_USES_PHASE = {'true' if any(x >= 82 for x in feat) else 'false'};  // phase features 82..89 needed?\n")
    f.write(f"constexpr bool MODEL_USES_DOMFREQ = {'true' if any(48 <= x < 56 for x in feat) else 'false'};  // dominant-frequency features 48..55 needed?\n")
    f.write(f"constexpr int NUM_FEATURES = {len(T.FEATURE_NAMES)};  // full vector; tree indices refer to it\n")
    f.write(f"constexpr int NUM_CLASSES = {len(classes)};\n")
    f.write(f"constexpr int NUM_TREES = {len(roots)};\n")
    f.write(f"constexpr int NUM_NODES = {n_nodes};\n\n")
    f.write(f"static const char* const CLASS_NAMES[NUM_CLASSES] = {{{names}}};\n\n")
    f.write("struct DefaultPattern { uint8_t pattern, r, g, b, brightness; };\n")
    f.write(f"static const DefaultPattern DEFAULT_PATTERNS[NUM_CLASSES] = {{{', '.join(defaults)}}};\n\n")
    f.write(c_array("TREE_ROOT", "uint16_t", roots))
    f.write(c_array("TREE_FEATURE", "int8_t", feat, 20))
    f.write(c_array("TREE_THRESH", "float", c_floats(thr), 8))
    f.write(c_array("TREE_LEFT", "uint16_t", left))
    f.write(c_array("TREE_RIGHT", "uint16_t", right))
    f.write("\n} // namespace wb\n")
print(f"Wrote {OUT}: {len(roots)} trees, {n_nodes} nodes, about {n_nodes * 9 / 1024:.0f} KB of flash.")

# ------------------------------------------------------ model_selftest.h
if SELFTEST:
    wins, expected = [], []
    for cls in classes:
        for p in sorted(glob.glob(T.TRAIN_GLOB)):
            if os.path.basename(p) in T.EXCLUDE:
                continue
            df = pd.read_csv(p)
            if T.MERGE.get(df["label"].iloc[0], df["label"].iloc[0]) != cls:
                continue
            arr = df[["ax", "ay", "az", "gx", "gy", "gz"]].to_numpy()[T.TRIM:-T.TRIM]
            wins.append(arr[10 * T.STRIDE:10 * T.STRIDE + T.WIN])
            break
    wins = np.array(wins)
    feats = np.array([T.window_features(w) for w in wins])
    expected = [predict_like_firmware(r) for r in feats]
    with open(OUT_TEST, "w") as f:
        f.write("// Generated by tools/export_model.py. One real window per class.\n")
        f.write("// Firmware self-test: compute features from SELFTEST_WINDOWS[i], compare to\n")
        f.write("// SELFTEST_FEATURES[i] (tolerance ~1e-3 relative), and check the tree vote\n")
        f.write("// equals SELFTEST_EXPECTED[i]. Raw columns: ax ay az gx gy gz, same units as the CSVs.\n")
        f.write("#pragma once\nnamespace wb {\n\n")
        f.write(f"constexpr int SELFTEST_N = {len(wins)};\n")
        f.write(c_array("SELFTEST_EXPECTED", "int", expected, 20))
        f.write(c_array("SELFTEST_WINDOWS", "float", c_floats(wins.ravel()), 6))
        f.write(c_array("SELFTEST_FEATURES", "float", c_floats(feats.ravel()), 8))
        f.write("\n} // namespace wb\n")
    print(f"Wrote {OUT_TEST}: {len(wins)} test windows.")

print("Upload the firmware to the board.")