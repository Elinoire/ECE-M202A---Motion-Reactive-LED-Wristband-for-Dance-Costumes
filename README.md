# Motion-Reactive LED Wristband for Dance Costumes

An Arduino Nano 33 BLE Sense Rev2 worn on the wrist reads its IMU, a random-forest classifier running on the board recognizes six movements, and an APA102 LED strip changes color for each one.

| movement | what it is | default LED color |
|---|---|---|
| `still` | arm at rest | green |
| `updown` | vertical motion | cyan |
| `horizontal` | side-to-side or front-to-back (recorded as `leftright` and `frontback`, merged for training) | orange |
| `diagonal` | diagonal sweep | magenta |
| `circle_x` | circle in the horizontal plane | yellow |
| `circle_y` | circle in the vertical plane | blue |

Note: yellow and orange look essentially the same, this was my quick "hack" to merge the two categories for now. If we change the categories, these colors should be changed to be more distinct.

**Status:** data collection, offline evaluation, model export and fully on-board classification with LED output all work. The board needs only power once flashed. Accuracy isn't great, and there is a fading effect to transition between LED colors. 

---

## Repo layout

| path | what it does |
|---|---|
| `src/main.cpp` | firmware: IMU sampling, feature extraction, forest, smoothing, LED output |
| `src/model.h` | **generated** by `tools/export_model.py` (trees, class names, default colors, window constants). Never edit by hand. Commit it after exporting |
| `src/wb_features.h` | C++ port of the feature code in `train.py`. Must compute the same numbers as Python |
| `src/patterns.h`, `src/apa102.h` | LED pattern rendering and APA102 strip driver |
| `src/train/train.py` | feature definitions, training recipes, evaluation (and imported by the export script) |
| `src/train/collect.py` | records one labeled 30 s IMU recording to `data/` |
| `tools/export_model.py` | trains the final model and writes `src/model.h` |
| `data/*.csv` | recordings, one file per movement per session |

---

## Setup (once)

1. Install VS Code and Python 3.
2. Open this folder in VS Code and install the recommended extensions (PlatformIO IDE, Python).
3. Create the Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

In each new terminal, run `source venv/bin/activate` from the repo folder.

---

## Workflow at a glance

```
record data  ->  train.py (evaluate)  ->  export_model.py  ->  flash firmware
collect.py       pick RECIPE, EXCLUDE      writes src/model.h   board runs on its own
```

Re-run the last three steps whenever you add recordings, change `EXCLUDE`, change features, or change the window settings.

---

## 1. Record data

Flash the **logger** build (it streams raw IMU data over USB), click the plug icon in VS Code to check that numbers are streaming, then **close the Serial Monitor** so the next step can use the port.

One command records one movement:

```bash
python3 src/train/collect.py still --session s13
python3 src/train/collect.py updown --session s13
python3 src/train/collect.py circle_x --session s13
```

- Each run counts down 5 seconds, then records 30 seconds. Do the movement continuously until it says `Saved`.
- Use a **new `--session` tag every time the band is taken off and put back on**. Sessions stand in for different orientations or wearers, and the evaluation depends on them.
- Keep the board mounted the same way every time within a session.
- Try to record **every movement in every session**.
- Horizontal motion is recorded as `leftright` and `frontback`. Training merges both into `horizontal`.

### Data format

Each file in `data/` is a CSV with columns `ax, ay, az, gx, gy, gz, label, session` (acceleration in g, gyro in deg/s, about 89 samples per second). File names look like `circle_x_s11_1794710971.csv` (label, session, timestamp), so recording the same movement twice in a session adds a file instead of overwriting.

The first and last 100 samples (`TRIM`) of every recording are dropped, so you can start the movement a little early or stop a little late.

### Replacing or adding data

- **Add data:** record with a new session tag, then re-run training and export.
- **Replace everything:** move or delete the old CSVs in `data/`, record the new ones, then re-run training and export. Check `SESSION_SHIFT` (below) because it is specific to the current dataset.
- **Replace one bad recording:** record it again (same label and session tag), then either delete the bad CSV or add its file name to `EXCLUDE`.

---

## 2. Train and evaluate

```bash
python3 src/train/train.py
```

It reads every CSV in `data/`, cuts it into overlapping windows, computes features, and prints a table:

```
recipe                  blocked     LOSO
base+shape                0.xxx    0.xxx
+gravity                  0.xxx    0.xxx
+gravity+aug              0.xxx    0.xxx
+gravity+phase+aug        0.xxx    0.xxx
```

### What the numbers mean

| column | what it tests | how to read it |
|---|---|---|
| `blocked` | each recording is cut into 5 time blocks; the model is tested on one block after training on the others (windows next to the test block are dropped so overlapping windows can't leak) | optimistic. Measures how well it knows movements it has seen in the same session |
| `LOSO` | leave-one-session-out: train on all sessions but one, test on the held-out one | **the number that matters.** Approximates a new orientation or wearer the model never saw |

Accuracy is per window (about 0.9 s each), before the board's smoothing. A wide gap between `blocked` and `LOSO` means the model works for sessions it has seen but doesn't generalize to new ones, which usually means more varied data is needed.

### Recipes

| recipe | features |
|---|---|
| `base+shape` | 56 basic statistics + 13 motion-shape features |
| `+gravity` | adds 13 gravity-frame features (rotation-invariant about gravity) |
| `+gravity+aug` | same features, training also uses randomly rotated copies of each window |
| `+gravity+phase+aug` | adds 8 phase features (needs an FFT on the board) |

Choose the recipe with the best LOSO, but only accept a bigger feature group if it helps by several points. `RECIPE` in `tools/export_model.py` is set to `+gravity+aug`: the phase features gave only about +1.5 points and the firmware does not compute them, so don't switch to a phase recipe unless `wb_features.h` implements them.

### Reference results (from earlier runs, for comparison)

With a 100-sample window the best recipe scored about 0.81 LOSO per window (about 0.85 with 5-vote smoothing), and a 64-sample window dropped it to about 0.76. The main confusions are:
- `circle_x` vs `horizontal` (the biggest one)
- `diagonal` vs `updown` and `circle_y`

If these numbers change a lot after new data, look at the data first (mislabeled or oddly recorded files).

### `EXCLUDE`: leaving out bad recordings

`EXCLUDE` in `train.py` is a set of file names (not paths) that are ignored for both training and evaluation. The files stay in `data/`.

```python
EXCLUDE = {
    "circle_x_s11_1794710971.csv",
    "frontback_s6_1794701141.csv",
    "updown_s6_1794701021.csv",
    "frontback_s4_1794698684.csv",
}
```

These four were 82 to 99 percent misclassified when held out and look mislabeled or recorded differently. Treat it as a stopgap: re-record them instead of quietly dropping data, then remove the names. Set it to `set()` to use every file. Changing `EXCLUDE` changes the model, so re-run `export_model.py`.

### Other settings in `train.py`

| setting | meaning |
|---|---|
| `WIN`, `STRIDE` | window length and step in samples (currently `WIN = 80`, `STRIDE = 25`; about 89 samples per second). **Must match the firmware**, see below |
| `MERGE` | labels combined into one class (`leftright`, `frontback` -> `horizontal`) |
| `SESSION_SHIFT` | renumbers `frontback` sessions by +6 (s1 -> s7 ...) so each session has one file per class and LOSO can test `horizontal` in every session. It only affects grouping, never features. Revisit it when you add sessions, and set it to `{}` if every session already has all classes |
| `AUG_COPIES`, `YAW_DEG`, `TILT_DEG` | augmentation for the `+aug` recipes. Yaw is kept small on purpose because circles and diagonals can depend on direction in the horizontal plane |
| `make_model()` | random forest, 50 trees, depth 12 |

---

## 3. Export to the board

```bash
python3 tools/export_model.py
```

It trains the final model on all (non-excluded) data with the `RECIPE` set at the top of the file, flattens the trees into arrays, and writes `src/model.h`. That file contains the trees, the class names (in alphabetical order: `circle_x, circle_y, diagonal, horizontal, still, updown`), the default LED colors, and the window constants (`MODEL_WIN`, `MODEL_STRIDE`, `MODEL_SAMPLE_HZ`, ...). Then upload the classifier firmware. **Commit `src/model.h` after exporting.**

### LED color mapping

Colors live in `DEFAULTS` in `tools/export_model.py` and are baked into `model.h` at export time:

| movement | RGB | color |
|---|---|---|
| `still` | (0, 255, 0) | green |
| `updown` | (0, 255, 255) | cyan |
| `horizontal` | (255, 160, 0) | orange |
| `diagonal` | (255, 0, 255) | magenta |
| `circle_x` | (255, 255, 0) | yellow |
| `circle_y` | (0, 0, 255) | blue |

`BRIGHTNESS = 30` (0 to 255) sets the default brightness for all of them. The firmware also caps brightness at `MAX_BRIGHTNESS = 160` in `main.cpp`. Every movement currently uses pattern type `0` (solid); the other types in `patterns.h` are `1` pulse, `2` chase, `3` strobe and `4` rainbow. To change a pattern type, edit the leading `0` in the line that builds `defaults` in `export_model.py`.

To change colors or brightness, edit `export_model.py`, export, and upload again.

---

## 4. How the firmware works (`src/main.cpp`)

1. **Sampling.** The IMU is read at about 89 Hz (`MODEL_SAMPLE_HZ`), paced by a timer so the rate matches the recordings. The I2C bus runs at 400 kHz.
2. **Window.** The latest `WIN` samples sit in a ring buffer. Every `STRIDE` samples (25, about 0.28 s) the window is classified.
3. **Features.** `wb::compute_features()` (in `wb_features.h`) computes the same features as `train.py`.
4. **Forest.** `treeVotes()` walks all 50 trees; the result is the fraction of trees voting for each movement.
5. **Soft vote.** The fractions are averaged over time (an exponential moving average, `score[]`). The shown movement changes only when a new movement is clearly ahead:
   - *slow path:* leads the current movement by `SWITCH_MARGIN`, has at least `MIN_SCORE`, and `HOLD_PREDS` predictions have passed since the last switch;
   - *fast path:* has at least `FAST_SCORE` and leads by `FAST_MARGIN`, with no hold. It lets clear changes through sooner.
6. **LED color.** A second, faster-smoothed score (`cscore[]`) drives the color, as a mix of all movement colors weighted by their scores. The strip starts fading toward a new color before the official switch, which hides some of the latency. The pattern type and brightness follow the official movement. About 60 frames per second.
7. **Failure.** If the IMU does not start, the strip turns solid red and stops.

### Tunable constants (top of `main.cpp`)

| constant | value | effect |
|---|---|---|
| `NUM_LEDS` | 60 | set to your strip's length |
| `MAX_BRIGHTNESS` | 160 | brightness cap to protect the battery and regulator |
| `EMA_ALPHA` | 0.6 | higher = smoother, more delay |
| `SWITCH_MARGIN` | 0.3 | higher = fewer false switches, more delay |
| `MIN_SCORE` | 0.35 | minimum support for a slow-path switch |
| `HOLD_PREDS` | 2 | predictions between switches (2 = about 0.56 s) |
| `FAST_SCORE`, `FAST_MARGIN` | 0.55, 0.15 | confident switches skip the hold. Set `FAST_SCORE` above 1.0 to disable |
| `COLOR_ALPHA` | 0.25 | color smoothing (lower = faster color, more jitter) |
| `COLOR_EASE` | 0.4 | per-frame fade speed of the color |

Reducing flicker and reducing delay pull against each other. If the LEDs flicker between movements, raise `EMA_ALPHA` or `SWITCH_MARGIN`. If switching feels slow, lower them or `HOLD_PREDS`. These constants only need a firmware upload, not a re-export. Your earlier flicker evaluation on recorded data showed that this soft vote cut label flips per minute by roughly 4 times compared with a plain 5-vote majority.

### Keeping Python and C++ in sync

- **`WIN` in `train.py` must equal `WIN` in `wb_features.h`**, and the export writes the same value into `model.h`. `main.cpp` has `static_assert`s that stop the build if they disagree. After changing `WIN` or `STRIDE`, edit `wb_features.h` if needed, re-run `export_model.py`, and re-upload. (The comments in `main.cpp` still mention a 100-sample window and a 0.28 s stride; they're only comments.)
- **If you change a feature in `train.py`, change it identically in `wb_features.h`.** The `static_assert`s only check sizes, not values. The old boot-time self-test that compared C++ features to Python was removed along with the other debug code, so a mismatch would now show up only as bad accuracy on the board. If this happens, check `git history` for the self-test and bring it back temporarily.

---

## Hardware setup

Solder the two VBUS pads on either the top or the bottom of the Nano 33 BLE Sense Rev2 together.

Connect the APA102 LED wires to the Nano 33 BLE Sense Rev2:

- CLK to D13
- Data to D11
- Ground to Ground
- Input voltage to 5V out

To check the LEDs on their own: install the Arduino IDE, select the board, install the Adafruit DotStar library (Tools > Manage Libraries), and run Files > Examples > Adafruit DotStar > onboard.

---

## Troubleshooting

- **No serial port, or it hangs:** close the Serial Monitor, or unplug and replug the board. If upload fails, double-tap the reset button and retry.
- **Python or pip is very slow:** the venv or repo is in an iCloud-synced folder. Move it.
- **Build error `model.h is missing`:** run `python3 tools/export_model.py`.
- **Build error from a `static_assert`:** `wb_features.h` and `model.h` disagree on window length, feature count or `DOM_BINS`. Fix `wb_features.h` and re-export.
- **Strip turns solid red:** the IMU did not initialize. Unplug and replug the board.
- **LEDs flicker or react slowly:** see the tunable constants above.
- **Accuracy is fine in `train.py` but poor on the wrist:** the band is mounted differently from the training sessions, or the movements differ from how they were recorded. Record a new session in the way the band is actually worn.

---

## Next steps

- Collect new training data with specific movements in mind (either for specific dance moves or for particular motions within each class). Once this is done we can edit/remove the EXCLUDE files in train.py
- Make it wireless (powered by a battery pack & portable)