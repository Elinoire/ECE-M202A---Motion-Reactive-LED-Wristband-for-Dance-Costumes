// Motion-reactive LED wristband: full firmware for the Arduino Nano 33 BLE Sense Rev2.
//
//   IMU (~89 Hz) -> 50-sample window -> 56 features -> random forest -> movement
//   movement -> pattern table -> APA102 strip
//   BLE: notifies the current movement, and lets a client read/edit the pattern table.
//
// Build: PlatformIO env "wristband". First run `python3 tools/export_model.py` to create model.h.

#include <Arduino.h>
#include <ArduinoBLE.h>
#include <Arduino_BMI270_BMM150.h>

#if !__has_include("model.h")
#error "model.h is missing. Run: python3 tools/export_model.py"
#endif
#include "wb_features.h"
#include "model.h"
#include "classifier.h"
#include "patterns.h"
#include "apa102.h"

// ---------------- Settings ----------------
constexpr int     NUM_LEDS       = 60;     // how many LEDs you cut the strip to
constexpr uint8_t MAX_BRIGHTNESS = 160;    // cap (0-255) to protect the battery and regulator
constexpr int     LIVE_STRIDE    = 10;     // classify every 10 new samples (about 0.11 s)
constexpr int     CONFIRM        = 2;      // identical predictions in a row before the LEDs change
constexpr int     DEBUG_PIN      = 2;      // D2: toggles at every movement change (for latency measurement)

// ---------------- BLE ----------------
// Service and characteristics (custom UUIDs):
//   movement (read, notify): 1 byte, index of the current movement (see CLASS_NAMES in model.h)
//   config   (read, write):  write 6 bytes [movement, pattern, r, g, b, brightness] to change one movement;
//                            read returns the whole table, 5 bytes per movement [pattern, r, g, b, brightness]
// Patterns: 0 solid, 1 pulse, 2 chase, 3 strobe, 4 rainbow.
constexpr int TABLE_BYTES = 5 * wb::NUM_CLASSES;
BLEService wristbandService("8f2a0001-4c1d-4b7e-9a21-5e3d7c6b0001");
BLEByteCharacteristic movementChar("8f2a0002-4c1d-4b7e-9a21-5e3d7c6b0001", BLERead | BLENotify);
BLECharacteristic configChar("8f2a0003-4c1d-4b7e-9a21-5e3d7c6b0001", BLERead | BLEWrite,
                             TABLE_BYTES > 6 ? TABLE_BYTES : 6, false);

// ---------------- State ----------------
wb::Apa102 strip(NUM_LEDS);
wb::Rgb pixels[NUM_LEDS];
wb::PatternCfg table[wb::NUM_CLASSES];

float ring[wb::WIN][6];       // most recent samples (ring buffer)
int ring_head = 0;            // next write position (= oldest sample once full)
int ring_count = 0;
int since_classify = 0;

int current = -1;             // movement the LEDs are showing (-1 = none yet)
int last_pred = -1;
int streak = 0;
bool debug_state = false;

void publishTable() {
  uint8_t bytes[TABLE_BYTES];
  for (int i = 0; i < wb::NUM_CLASSES; i++) {
    bytes[5 * i + 0] = table[i].type;
    bytes[5 * i + 1] = table[i].r;
    bytes[5 * i + 2] = table[i].g;
    bytes[5 * i + 3] = table[i].b;
    bytes[5 * i + 4] = table[i].brightness;
  }
  configChar.writeValue(bytes, TABLE_BYTES);
}

void handleConfigWrite() {
  int len = configChar.valueLength();
  const uint8_t* v = configChar.value();
  if (len >= 6 && v[0] < wb::NUM_CLASSES && v[1] < wb::P_COUNT) {
    table[v[0]] = {v[1], v[2], v[3], v[4], v[5]};
  }
  publishTable();             // so the next read shows the current table
}

void fatal(uint8_t r) {       // show a solid color and stop (red = IMU failed, blue = BLE failed)
  for (int i = 0; i < NUM_LEDS; i++) pixels[i] = {r, 0, (uint8_t)(r ? 0 : 60)};
  strip.show(pixels);
  while (true) { delay(1000); }
}

void setup() {
  Serial.begin(115200);       // optional: only used for debug output when a computer is connected
  pinMode(DEBUG_PIN, OUTPUT);
  strip.begin();

  for (int i = 0; i < wb::NUM_CLASSES; i++) {
    const wb::DefaultPattern& d = wb::DEFAULT_PATTERNS[i];
    table[i] = {d.pattern, d.r, d.g, d.b, d.brightness};
  }

  if (!IMU.begin()) fatal(80);
  if (!BLE.begin()) fatal(0);

  BLE.setLocalName("Wristband");
  BLE.setDeviceName("Wristband");
  BLE.setAdvertisedService(wristbandService);
  wristbandService.addCharacteristic(movementChar);
  wristbandService.addCharacteristic(configChar);
  BLE.addService(wristbandService);
  movementChar.writeValue((uint8_t)255);   // 255 = none yet
  publishTable();
  BLE.advertise();

  if (Serial) {
    Serial.print("Classes:");
    for (int i = 0; i < wb::NUM_CLASSES; i++) { Serial.print(' '); Serial.print(wb::CLASS_NAMES[i]); }
    Serial.println();
  }
}

void classifyWindow() {
  uint32_t t0 = micros();

  float window[wb::WIN][6];   // linearize the ring buffer, oldest sample first
  for (int i = 0; i < wb::WIN; i++) {
    memcpy(window[i], ring[(ring_head + i) % wb::WIN], sizeof(window[i]));
  }
  float feats[wb::NFEAT];
  wb::compute_features(window, wb::MODEL_SAMPLE_HZ, feats);
  int pred = wb::predict(feats);

  uint32_t infer_us = micros() - t0;

  // Require CONFIRM identical predictions in a row before switching (stops flicker).
  if (pred == last_pred) streak++; else { last_pred = pred; streak = 1; }
  if (streak >= CONFIRM && pred != current) {
    current = pred;
    debug_state = !debug_state;
    digitalWrite(DEBUG_PIN, debug_state ? HIGH : LOW);
    movementChar.writeValue((uint8_t)current);   // notifies subscribed BLE clients
    if (Serial) {
      Serial.print(millis());
      Serial.print(',');
      Serial.print(wb::CLASS_NAMES[current]);
      Serial.print(",infer_us=");
      Serial.println(infer_us);
    }
  }
}

void loop() {
  BLE.poll();
  if (configChar.written()) handleConfigWrite();

  // New IMU sample
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    float* row = ring[ring_head];
    IMU.readAcceleration(row[0], row[1], row[2]);   // g
    IMU.readGyroscope(row[3], row[4], row[5]);      // deg/s
    ring_head = (ring_head + 1) % wb::WIN;
    if (ring_count < wb::WIN) ring_count++;
    since_classify++;

    if (ring_count == wb::WIN && since_classify >= LIVE_STRIDE) {
      since_classify = 0;
      classifyWindow();
    }
  }

  // LED frame, about 60 per second
  static uint32_t last_frame = 0;
  uint32_t now = millis();
  if (now - last_frame >= 16) {
    last_frame = now;
    if (current < 0) {
      for (int i = 0; i < NUM_LEDS; i++) pixels[i] = {0, 0, 0};
    } else {
      wb::PatternCfg p = table[current];
      if (p.brightness > MAX_BRIGHTNESS) p.brightness = MAX_BRIGHTNESS;
      wb::render_pattern(p, now, pixels, NUM_LEDS);
    }
    strip.show(pixels);
  }
}
