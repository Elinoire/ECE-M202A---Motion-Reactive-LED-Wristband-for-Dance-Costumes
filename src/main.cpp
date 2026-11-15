// Motion-reactive LED wristband: full firmware for the Arduino Nano 33 BLE Sense Rev2.
//
//   IMU (~89 Hz) -> 100-sample window -> 90 features -> random forest -> majority vote -> movement
//   movement -> pattern table -> APA102 strip
//   BLE (optional, USE_BLE): notifies the current movement, and lets a client read/edit the pattern table.
//
// Build: PlatformIO env "wristband". First run `python3 tools/export_model.py` to create model.h.

#include <Arduino.h>
#include <Wire.h>

// 1 = BLE on (app can read the movement and edit patterns), 0 = BLE completely off.
// With 0 the LEDs use the default patterns from model.h.
#define USE_BLE 0

#if USE_BLE
#include <ArduinoBLE.h>
#endif
#include <Arduino_BMI270_BMM150.h>

#if !__has_include("model.h")
#error "model.h is missing. Run: python3 tools/export_model.py"
#endif
#include "model.h"
#include "wb_features.h"
#include "classifier.h"
#include "patterns.h"
#include "apa102.h"

// The firmware features must match what the model was trained on.
static_assert(wb::WIN == wb::MODEL_WIN, "wb_features.h WIN differs from model.h MODEL_WIN");
static_assert(wb::NFEAT == wb::NUM_FEATURES, "wb_features.h NFEAT differs from model.h NUM_FEATURES");
static_assert(wb::FEAT_DOM_BINS == wb::DOM_BINS, "wb_features.h FEAT_DOM_BINS differs from model.h DOM_BINS");

// Optional boot-time check that C++ features == Python features (file made by export_model.py).
#define RUN_SELFTEST 1
#if RUN_SELFTEST && __has_include("model_selftest.h")
#include "model_selftest.h"
#define HAVE_SELFTEST 1
#endif

// ---------------- Settings ----------------
constexpr int     NUM_LEDS       = 60;     // how many LEDs you cut the strip to
constexpr uint8_t MAX_BRIGHTNESS = 160;    // cap (0-255) to protect the battery and regulator
constexpr int     LIVE_STRIDE    = wb::MODEL_STRIDE;  // 25 samples (0.28 s): what the evaluation assumed
constexpr int     SMOOTH         = wb::MODEL_SMOOTH;  // majority vote over the last N predictions
constexpr uint32_t I2C_HZ        = 400000; // IMU bus speed; the default 100 kHz made each read ~8 ms
constexpr int     DEBUG_PIN      = 2;      // D2: toggles at every movement change (for latency measurement)

#if USE_BLE
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
#endif

// ---------------- State ----------------
wb::Apa102 strip(NUM_LEDS);
wb::Rgb pixels[NUM_LEDS];
wb::PatternCfg table[wb::NUM_CLASSES];

float ring[wb::WIN][6];       // most recent samples (ring buffer)
int ring_head = 0;            // next write position (= oldest sample once full)
int ring_count = 0;
int since_classify = 0;
#ifdef WB_PROFILE
uint32_t rate_count = 0;      // IMU samples read, for the sample-rate printout
uint32_t read_us = 0, avail_us = 0, avail_calls = 0, led_us = 0, led_frames = 0;
#endif

int current = -1;             // movement the LEDs are showing (-1 = none yet)
int history[SMOOTH];          // last SMOOTH raw predictions (ring)
int hist_head = 0;
int hist_count = 0;
bool debug_state = false;

#if USE_BLE
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
#endif

void fatal(uint8_t r) {       // show a solid color and stop (red = IMU failed, blue = BLE failed)
  for (int i = 0; i < NUM_LEDS; i++) pixels[i] = {r, 0, (uint8_t)(r ? 0 : 60)};
  strip.show(pixels);
  while (true) { delay(1000); }
}

#ifdef HAVE_SELFTEST
// Feeds one real recorded window per class through the C++ features and the forest.
// Any feature index printed as MISMATCH is a bug in wb_features.h (index = FEATURE_NAMES in train.py).
void selfTest() {
  static float feats[wb::NFEAT];
  int bad_total = 0, bad_class = 0;
  for (int i = 0; i < wb::SELFTEST_N; i++) {
    const float (*win)[6] = (const float (*)[6]) & wb::SELFTEST_WINDOWS[i * wb::WIN * 6];
    const float* want = &wb::SELFTEST_FEATURES[i * wb::NFEAT];
    wb::compute_features(win, wb::MODEL_SAMPLE_HZ, feats);
    for (int j = 0; j < wb::NFEAT; j++) {
      if (!wb::MODEL_USES_PHASE && j >= 82) continue;   // phase features are skipped on purpose
      if (!wb::MODEL_USES_DOMFREQ && j >= 48 && j < 56) continue;  // skipped on purpose
      float tol = 2e-3f * fabsf(want[j]) + 1e-4f;
      if (fabsf(feats[j] - want[j]) > tol) {
        bad_total++;
        Serial.print("MISMATCH window "); Serial.print(i);
        Serial.print(" feature "); Serial.print(j);
        Serial.print(" got "); Serial.print(feats[j], 6);
        Serial.print(" want "); Serial.println(want[j], 6);
      }
    }
    if (wb::predict(feats) != wb::SELFTEST_EXPECTED[i]) bad_class++;
  }
  Serial.print("Self-test: "); Serial.print(bad_total); Serial.print(" feature mismatches, ");
  Serial.print(bad_class); Serial.print(" wrong classes out of "); Serial.print(wb::SELFTEST_N);
  Serial.println(" windows.");
}
#endif

void imuHook();               // defined below; used by setup()

void setup() {
  Serial.begin(115200);       // optional: only used for debug output when a computer is connected
#ifdef WB_PROFILE
  { uint32_t t = millis(); while (!Serial && millis() - t < 4000) {} }   // profile builds: wait so the boot lines are not missed
#endif
  pinMode(DEBUG_PIN, OUTPUT);
  strip.begin();

  for (int i = 0; i < wb::NUM_CLASSES; i++) {
    const wb::DefaultPattern& d = wb::DEFAULT_PATTERNS[i];
    table[i] = {d.pattern, d.r, d.g, d.b, d.brightness};
  }

  if (!IMU.begin()) fatal(80);
  Wire1.setClock(I2C_HZ);       // internal sensor bus on the Nano 33 BLE Sense Rev2; must come after IMU.begin()
  // wb::poll_hook = imuHook;   // DISABLED: each IMU read took ~10 ms here, which made the features take ~180 ms
#if USE_BLE
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
#endif

  if (Serial) {
    Serial.print("Classes:");
    for (int i = 0; i < wb::NUM_CLASSES; i++) { Serial.print(' '); Serial.print(wb::CLASS_NAMES[i]); }
    Serial.println();
#ifdef __ARM_FP
    Serial.print("hardware FPU flag: __ARM_FP="); Serial.println(__ARM_FP);
#else
    Serial.println("hardware FPU flag: NOT defined (software float!)");
#endif
#if defined(__OPTIMIZE_SIZE__)
    Serial.println("optimization: size (-Os)");
#elif defined(__OPTIMIZE__)
    Serial.println("optimization: speed (-O1/-O2/-O3)");
#else
    Serial.println("optimization: NONE (-O0)");
#endif
#ifdef HAVE_SELFTEST
    selfTest();
#endif
  }
}

// Majority vote over the last SMOOTH predictions (same idea as smooth() in train.py).
// A tie keeps the current movement if it is among the leaders, otherwise the lowest index.
int vote(int pred) {
  history[hist_head] = pred;
  hist_head = (hist_head + 1) % SMOOTH;
  if (hist_count < SMOOTH) hist_count++;
  int counts[wb::NUM_CLASSES] = {0};
  for (int i = 0; i < hist_count; i++) counts[history[i]]++;
  int best = 0;
  for (int c = 1; c < wb::NUM_CLASSES; c++) if (counts[c] > counts[best]) best = c;
  if (current >= 0 && counts[current] == counts[best]) return current;
  return best;
}

void classifyWindow() {
  uint32_t t0 = micros();

  static float window[wb::WIN][6];   // linearize the ring buffer, oldest sample first
  for (int i = 0; i < wb::WIN; i++) {
    memcpy(window[i], ring[(ring_head + i) % wb::WIN], sizeof(window[i]));
  }
  static float feats[wb::NFEAT];
  uint32_t t1 = micros();
  wb::compute_features(window, wb::MODEL_SAMPLE_HZ, feats);
  uint32_t t2 = micros();
  int pred = wb::predict(feats);
  uint32_t t3 = micros();
  int winner = vote(pred);

  uint32_t infer_us = t3 - t0;
  uint32_t feat_us = t2 - t1, tree_us = t3 - t2;

  if (winner != current) {
    current = winner;
    debug_state = !debug_state;
    digitalWrite(DEBUG_PIN, debug_state ? HIGH : LOW);
#if USE_BLE
    movementChar.writeValue((uint8_t)current);   // notifies subscribed BLE clients
#endif
    if (Serial) {
      Serial.print(millis());
      Serial.print(',');
      Serial.print(wb::CLASS_NAMES[current]);
      Serial.print(",infer_us=");
      Serial.print(infer_us);
      Serial.print(",feat_us=");
      Serial.print(feat_us);
      Serial.print(",tree_us=");
      Serial.print(tree_us);
#ifdef WB_PROFILE
      Serial.print(",setup=");  Serial.print(wb::prof_us[0]);
      Serial.print(",base=");   Serial.print(wb::prof_us[1]);
      Serial.print(",shape=");  Serial.print(wb::prof_us[2]);
      Serial.print(",grav=");   Serial.print(wb::prof_us[3]);
      Serial.print(",phase=");  Serial.print(wb::prof_us[4]);
#endif
      Serial.println();
    }
  }
}

// Read one IMU sample into the ring buffer, paced at the training rate (MODEL_SAMPLE_HZ).
// Training data was recorded at ~89 Hz, so the board samples at the same rate no matter how fast
// the I2C bus and the loop are. Returns true if it read a sample.
constexpr uint32_t SAMPLE_PERIOD_US = (uint32_t)(1000000.0f / wb::MODEL_SAMPLE_HZ + 0.5f);

bool readImuSample() {
  static uint32_t next_us = 0;
  uint32_t now = micros();
  if ((int32_t)(now - next_us) < 0) return false;                  // not time yet
#ifdef WB_PROFILE
  uint32_t pa = micros();
#endif
  bool ready = IMU.accelerationAvailable();                        // the sensor makes ~100 samples/s
#ifdef WB_PROFILE
  avail_us += micros() - pa;
  avail_calls++;
#endif
  if (!ready) return false;
#ifdef WB_PROFILE
  uint32_t pr = micros();
#endif
  float* row = ring[ring_head];
  IMU.readAcceleration(row[0], row[1], row[2]);   // g
  IMU.readGyroscope(row[3], row[4], row[5]);      // deg/s
#ifdef WB_PROFILE
  read_us += micros() - pr;
#endif
  // Schedule the next sample one period after the last SCHEDULED time (no drift); after a long
  // stall (e.g. classification) just restart the schedule instead of firing a burst of reads.
  next_us += SAMPLE_PERIOD_US;
  if ((int32_t)(now - next_us) > (int32_t)SAMPLE_PERIOD_US) next_us = now + SAMPLE_PERIOD_US;
  ring_head = (ring_head + 1) % wb::WIN;
  if (ring_count < wb::WIN) ring_count++;
  since_classify++;
#ifdef WB_PROFILE
  rate_count++;
#endif
  return true;
}

// Called from inside the feature code so no IMU samples are lost while it runs.
// Rate-limited: the IMU produces a sample about every 11 ms, so checking every 3 ms is plenty.
void imuHook() {
  static uint32_t last = 0;
  uint32_t now = micros();
  if (now - last < 3000) return;
  last = now;
  readImuSample();
}

void loop() {
#if USE_BLE
  BLE.poll();
  if (configChar.written()) handleConfigWrite();
#endif

  // New IMU sample
  if (readImuSample()) {
    if (ring_count == wb::WIN && since_classify >= LIVE_STRIDE) {
      since_classify = 0;
      classifyWindow();
    }
  }

#ifdef WB_PROFILE
  static uint32_t rate_t0 = 0;                    // effective IMU sample rate (should be about 89 Hz)
  uint32_t rt = millis();
  if (rt - rate_t0 >= 5000) {
    Serial.print("imu_rate_hz=");
    Serial.print(rate_count * 1000.0f / (float)(rt - rate_t0));
    Serial.print(" imu_read_us=");   Serial.print(rate_count ? read_us / rate_count : 0);
    Serial.print(" avail_check_us="); Serial.print(avail_calls ? avail_us / avail_calls : 0);
    Serial.print(" avail_calls_per_s="); Serial.print(avail_calls * 1000.0f / (float)(rt - rate_t0));
    Serial.print(" led_show_us=");   Serial.print(led_frames ? led_us / led_frames : 0);
    Serial.print(" led_fps=");       Serial.println(led_frames * 1000.0f / (float)(rt - rate_t0));
    rate_count = read_us = avail_us = avail_calls = led_us = led_frames = 0;
    rate_t0 = rt;
  }
#endif

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
#ifdef WB_PROFILE
    uint32_t pl = micros();
    strip.show(pixels);
    led_us += micros() - pl;
    led_frames++;
#else
    strip.show(pixels);
#endif
  }
}
