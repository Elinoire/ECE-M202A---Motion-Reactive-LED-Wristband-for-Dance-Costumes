"""Record labeled IMU data from the wristband over serial.

Usage:
    python3 tools/collect.py sweep --session s1 --seconds 30
    python3 tools/collect.py still --session s1 --port /dev/cu.usbmodem101

Close the PlatformIO serial monitor first (only one program can hold the port).
Output: data/<label>_<session>_<timestamp>.csv
"""
import argparse
import csv
import glob
import os
import sys
import time

import serial

ap = argparse.ArgumentParser()
ap.add_argument("label", help="movement name, e.g. still / sweep / spin / shake")
ap.add_argument("--session", default="s1", help="tag for a recording session (used for honest evaluation)")
ap.add_argument("--seconds", type=int, default=30)
ap.add_argument("--delay", type=int, default=5, help="countdown seconds before recording starts")
ap.add_argument("--port", default=None)
ap.add_argument("--baud", type=int, default=115200)
args = ap.parse_args()

port = args.port
if port is None:
    candidates = (glob.glob("/dev/cu.usbserial*") + glob.glob("/dev/cu.usbmodem*")
                  + glob.glob("/dev/cu.SLAB*") + glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    if not candidates:
        sys.exit("No serial port found. Plug in the board or pass --port.")
    port = candidates[0]
print(f"Using port {port}")

os.makedirs("data", exist_ok=True)
path = f"data/{args.label}_{args.session}_{int(time.time())}.csv"

ser = serial.Serial(port, args.baud, timeout=1)


def drain(seconds):
    """Keep reading and discarding so no backlog builds up in the USB/serial buffers."""
    end = time.time() + seconds
    count = 0
    while time.time() < end:
        if ser.readline():
            count += 1
    return count


if drain(2) == 0:             # let the connection settle, and check the board is streaming
    ser.close()
    sys.exit("The board isn't sending data. Close the serial monitor, unplug and replug "
             "the board, and try again.")

for n in range(args.delay, 0, -1):
    print(f"Get ready to do '{args.label}' in {n}...")
    drain(1)                  # keep draining during the countdown (a plain sleep lets data pile up)
print("GO")

rows = []
t_start = time.time()
t_end = t_start + args.seconds
last_shown = -1
while time.time() < t_end:
    elapsed = int(time.time() - t_start)
    if elapsed != last_shown:
        print(f"  recording... {elapsed}/{args.seconds} s ({len(rows)} samples)", end="\r", flush=True)
        last_shown = elapsed
    line = ser.readline().decode(errors="ignore").strip()
    parts = line.split(",")
    if len(parts) != 7:
        continue
    try:
        rows.append([float(p) for p in parts])
    except ValueError:
        continue              # header or partial line

ser.close()
print()

if len(rows) < 2:
    sys.exit("No valid data received. Check the board and that the serial monitor is closed.")

# Safety net: keep only the most recent `seconds` of board time, so any stale
# data that was buffered before GO is dropped.
cutoff = rows[-1][0] - args.seconds * 1000
rows = [r for r in rows if r[0] >= cutoff]

with open(path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["ms", "ax", "ay", "az", "gx", "gy", "gz", "label", "session"])
    for r in rows:
        w.writerow(r + [args.label, args.session])

span = (rows[-1][0] - rows[0][0]) / 1000.0
print(f"Saved {len(rows)} samples to {path}")
print(f"Board time covered: {span:.1f} s (asked for {args.seconds} s), "
      f"actual sample rate ~{(len(rows) - 1) / span:.0f} Hz")