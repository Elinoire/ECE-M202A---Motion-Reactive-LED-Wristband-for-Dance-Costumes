# ECE-M202A---Motion-Reactive-LED-Wristband-for-Dance-Costumes

# Motion-Reactive LED Wristband

An Arduino Nano 33 BLE Sense Rev2 worn on the wrist streams IMU data, and a classifier tells four movements apart: `still`, `leftright`, `updown`, `rotate`.

**Status:** data collection and offline classification work (99.1% accuracy on unseen sessions). LEDs, BLE, the web app, and on-device inference are not built yet.

## Setup (once)

1. Install VS Code and Python 3.
2. Open this folder in VS Code and install the recommended extensions (PlatformIO IDE, Python).
3. Create the Python virtual environment. 
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

In each new terminal, run `source venv/bin/activate` from the repo folder..

## Flash the board

1. Plug in the board and click **Upload** (→) in VS Code's bottom status bar. If it fails, double-tap the reset button and retry.
2. Click the **plug icon** to check that numbers are streaming, then **close the Serial Monitor** so the next step can use the port.

## Record data

One command records one movement. Use a new `--session` tag each time the band is taken off and put back on.

```bash
python3 tools/collect.py still --session sN
python3 tools/collect.py leftright --session sN
python3 tools/collect.py updown --session sN
python3 tools/collect.py rotate --session sN
```

Each run counts down 5 seconds, then records 30 seconds. Do the movement continuously until it says `Saved`. Keep the board mounted the same way every time. N corresponds to the the session number.

## Test the classifier

```bash
python3 tools/train.py
```

It trains on some sessions and tests on one it hasn't seen, so it needs at least two sessions.

## Troubleshooting

- **No serial port, or it hangs:** close the Serial Monitor, or unplug and replug the board.
- **Python or pip is very slow:** the venv or repo is in an iCloud-synced folder. Move it.
