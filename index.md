---
layout: default
title: Motion-Reactive LED Wristband for Dance Costumes
---

# ECE-M202A---Motion-Reactive-LED-Wristband-for-Dance-Costumes

# Motion-Reactive LED Wristband

An Arduino Nano 33 BLE Sense Rev2 worn on the wrist streams IMU data, and a classifier tells four movements apart: `still`, `leftright`, `updown`, `rotate`, and lights an APA102 LED strip in a different color for each movement.

**Status:** data collection and offline classification work (99.1% accuracy on unseen sessions), and on-board classification with LED output work. BLE, the web app, and more sophisticated inference are not built yet.

## Setup (once)

1. Install VS Code and Python 3.
2. Open this folder in VS Code and install the recommended extensions (PlatformIO IDE, Python).
3. Create the Python virtual environment. 
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

In each new terminal, run `source venv/bin/activate` from the repo folder.

## Flash the board

There are two firmware builds. Pick one in VS Code's bottom status bar (default: wireless), then click **Upload** (→). If it fails, double-tap the reset button and retry.

- `logger` streams raw IMU data over USB. Use it to record data.
- `wireless` is the on-board classifier with LED output.

To record data, upload `logger`, click the **plug icon** to check that numbers are streaming, then **close the Serial Monitor** so the next step can use the port.

## Record data

One command records one movement. Use a new `--session` tag each time the band is taken off and put back on.

```bash
python3 src/train/collect.py still --session sN
python3 src/train/collect.py leftright --session sN
python3 src/train/collect.py updown --session sN
python3 src/train/collect.py rotate --session sN
```

Each run counts down 5 seconds, then records 30 seconds. Do the movement continuously until it says `Saved`. Keep the board mounted the same way every time. N corresponds to the session number.

## Test the classifier

```bash
python3 tools/train.py
```

It trains on some sessions and tests on one it hasn't seen, so it needs at least two sessions.

## Put the classifier on the board

```bash
python3 tools/export_model.py
```

Then upload the `wireless` build. The board now classifies and lights the LEDs by itself, so you can unplug USB and power it from a battery or power bank. Re-run this and upload again whenever you record new data. Commit `src/model.h` after exporting.

To change the LEDs: set `NUM_LEDS` in `src/main.cpp` to your strip's length, and edit the colors and `BRIGHTNESS` in `tools/export_model.py` (then export and upload again).

Currently: Brightness = 30, leftright = blue, updown = green, rotate = magenta, still = yellow

## Troubleshooting

- **No serial port, or it hangs:** close the Serial Monitor, or unplug and replug the board.
- **Python or pip is very slow:** the venv or repo is in an iCloud-synced folder. Move it.


## Hardware Setup

Solder the two VBUS pads on either the top or the bottom of the Nano 33 BLE Sense Rev2 together.
Connect APA102 LED wires to Nano 33 BLE Sense Rev2 accordingly:

   Connect CLK to D13, Data to D11, Ground to Ground, and Input voltage to 5V out.

Download Arduino IDE

Connect Nano 33 BLE Sense Rev2 to laptop and selected board from dropdown list on Arduino IDE.

Go to Tools > Manage Libraries > Type in Adafruit Dotstar and install the library along with dependent libraries.

Go to Files > Examples > Adafruit Dotstar > onboard. Compile and run the code by clicking the (->) icon in the task bar at top.

