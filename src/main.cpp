// Motion-reactive LED wristband: firmware for the Arduino Nano 33 BLE Sense Rev2.
//
//   IMU (~89 Hz) -> 100-sample window -> 90 features -> random forest -> soft vote -> movement
//   movement -> pattern table -> APA102 strip
//
// Build: PlatformIO env "wristband". First run `python3 tools/export_model.py` to create model.h.

#include <Arduino.h>
#include <Wire.h>
#include <Arduino_BMI270_BMM150.h>

#if !__has_include("model.h")
#error "model.h is missing. Run: python3 tools/export_model.py"
#endif
#include "model.h"
#include "wb_features.h"
#include "patterns.h"
#include "apa102.h"

// The firmware features must match what the model was trained on.
static_assert(wb::WIN == wb::MODEL_WIN, "wb_features.h WIN differs from model.h MODEL_WIN");
static_assert(wb::NFEAT == wb::NUM_FEATURES, "wb_features.h NFEAT differs from model.h NUM_FEATURES");
static_assert(wb::FEAT_DOM_BINS == wb::DOM_BINS, "wb_features.h FEAT_DOM_BINS differs from model.h DOM_BINS");

// ---------------- Settings ----------------
constexpr int     NUM_LEDS       = 60;     // how many LEDs you cut the strip to
constexpr uint8_t MAX_BRIGHTNESS = 160;    // cap (0-255) to protect the battery and regulator
constexpr int     LIVE_STRIDE    = wb::MODEL_STRIDE;  // 25 samples (0.28 s): what the evaluation assumed
constexpr uint32_t I2C_HZ        = 400000; // IMU bus speed; the default 100 kHz made each read ~8 ms

// Soft vote (anti-flicker).
constexpr float EMA_ALPHA     = 0.6f;   // weight of the old score (higher = smoother, slower)
constexpr float SWITCH_MARGIN = 0.3f;   // a new movement must lead the current one by this much
constexpr float MIN_SCORE     = 0.35f;  // ...and have at least this much support
constexpr int   HOLD_PREDS    = 2;      // predictions that must pass between switches (2 = ~0.56 s)
// Fast switch: a clearly confident new movement switches at once, skipping the hold and the
// full margin. Set FAST_SCORE above 1.0f to disable.
constexpr float FAST_SCORE    = 0.55f;
constexpr float FAST_MARGIN   = 0.15f;

// LED color fade. The color follows its OWN, less smoothed score (cscore), so it starts moving
// toward a new movement sooner than the official switch above. It does not affect which
// movement is "current".
constexpr float COLOR_ALPHA   = 0.25f;  // color score smoothing (lower = reacts faster, 0 = raw votes)
constexpr float COLOR_EASE    = 0.4f;   // per LED frame (~60/s); higher = faster fade, 1.0 = no easing

// ---------------- State ----------------
wb::Apa102 strip(NUM_LEDS);
wb::Rgb pixels[NUM_LEDS];
wb::PatternCfg table[wb::NUM_CLASSES];

float ring[wb::WIN][6];       // most recent samples (ring buffer)
int ring_head = 0;            // next write position (= oldest sample once full)
int ring_count = 0;
int since_classify = 0;

int current = -1;             // movement the LEDs are showing (-1 = none yet)
float score[wb::NUM_CLASSES]; // smoothed (EMA) tree-vote fraction per movement; drives the switch
float cscore[wb::NUM_CLASSES];// faster-smoothed tree-vote fraction; drives the LED color only
int preds_since_switch = 0;   // predictions since the shown movement last changed

void fatal() {                // IMU failed: show solid red and stop
  for (int i = 0; i < NUM_LEDS; i++) pixels[i] = {80, 0, 0};
  strip.show(pixels);
  while (true) { delay(1000); }
}

void setup() {
  strip.begin();

  for (int i = 0; i < wb::NUM_CLASSES; i++) {
    const wb::DefaultPattern& d = wb::DEFAULT_PATTERNS[i];
    table[i] = {d.pattern, d.r, d.g, d.b, d.brightness};
  }
  for (int c = 0; c < wb::NUM_CLASSES; c++) { score[c] = 0.0f; cscore[c] = 0.0f; }

  if (!IMU.begin()) fatal();
  Wire1.setClock(I2C_HZ);       // internal sensor bus on the Nano 33 BLE Sense Rev2; must come after IMU.begin()
}

// Fraction of trees voting for each movement (a leaf stores its class index in TREE_LEFT).
void treeVotes(const float* f, float* frac) {
  int votes[wb::NUM_CLASSES] = {0};
  for (int t = 0; t < wb::NUM_TREES; t++) {
    int n = wb::TREE_ROOT[t];
    while (wb::TREE_FEATURE[n] >= 0) {
      n = (f[wb::TREE_FEATURE[n]] <= wb::TREE_THRESH[n]) ? wb::TREE_LEFT[n] : wb::TREE_RIGHT[n];
    }
    votes[wb::TREE_LEFT[n]]++;
  }
  for (int c = 0; c < wb::NUM_CLASSES; c++) frac[c] = votes[c] / (float)wb::NUM_TREES;
}

// Soft vote: exponential moving average of the per-movement tree-vote fractions.
// The shown movement changes when the new leader either
//   fast path: has at least FAST_SCORE and leads the current one by FAST_MARGIN, or
//   slow path: leads by SWITCH_MARGIN, has at least MIN_SCORE, and HOLD_PREDS predictions
//              passed since the last change.
// Returns the movement to show.
// Also updates cscore[], the faster score that only drives the LED color.
int vote(const float* frac) {
  preds_since_switch++;
  bool first = (current < 0);
  int best = 0;
  for (int c = 0; c < wb::NUM_CLASSES; c++) {
    score[c] = first ? frac[c] : EMA_ALPHA * score[c] + (1.0f - EMA_ALPHA) * frac[c];
    cscore[c] = first ? frac[c] : COLOR_ALPHA * cscore[c] + (1.0f - COLOR_ALPHA) * frac[c];
    if (score[c] > score[best]) best = c;           // ties -> lowest index
  }
  if (first) { preds_since_switch = 0; return best; }
  if (best == current) return current;

  bool fast = score[best] >= FAST_SCORE && score[best] >= score[current] + FAST_MARGIN;
  bool slow = preds_since_switch >= HOLD_PREDS && score[best] >= MIN_SCORE &&
              score[best] >= score[current] + SWITCH_MARGIN;
  if (!fast && !slow) return current;
  preds_since_switch = 0;
  return best;
}

void classifyWindow() {
  static float window[wb::WIN][6];   // linearize the ring buffer, oldest sample first
  for (int i = 0; i < wb::WIN; i++) {
    memcpy(window[i], ring[(ring_head + i) % wb::WIN], sizeof(window[i]));
  }
  static float feats[wb::NFEAT];
  wb::compute_features(window, wb::MODEL_SAMPLE_HZ, feats);
  float frac[wb::NUM_CLASSES];
  treeVotes(feats, frac);
  current = vote(frac);
}

// Read one IMU sample into the ring buffer, paced at the training rate (MODEL_SAMPLE_HZ).
// Training data was recorded at ~89 Hz, so the board samples at the same rate no matter how fast
// the I2C bus and the loop are. Returns true if it read a sample.
constexpr uint32_t SAMPLE_PERIOD_US = (uint32_t)(1000000.0f / wb::MODEL_SAMPLE_HZ + 0.5f);

bool readImuSample() {
  static uint32_t next_us = 0;
  uint32_t now = micros();
  if ((int32_t)(now - next_us) < 0) return false;                  // not time yet
  if (!IMU.accelerationAvailable()) return false;                  // the sensor makes ~100 samples/s

  float* row = ring[ring_head];
  IMU.readAcceleration(row[0], row[1], row[2]);   // g
  IMU.readGyroscope(row[3], row[4], row[5]);      // deg/s

  // Schedule the next sample one period after the last SCHEDULED time (no drift); after a long
  // stall (e.g. classification) just restart the schedule instead of firing a burst of reads.
  next_us += SAMPLE_PERIOD_US;
  if ((int32_t)(now - next_us) > (int32_t)SAMPLE_PERIOD_US) next_us = now + SAMPLE_PERIOD_US;
  ring_head = (ring_head + 1) % wb::WIN;
  if (ring_count < wb::WIN) ring_count++;
  since_classify++;
  return true;
}

void loop() {
  // New IMU sample
  if (readImuSample()) {
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
      wb::PatternCfg p = table[current];       // pattern type and brightness of the current movement
      // Target color = mix of the class colors weighted by cscore cubed (cubing lets the leading
      // class dominate sooner), then eased toward it every frame.
      float w[wb::NUM_CLASSES], sum = 0, r = 0, g = 0, b = 0;
      for (int c = 0; c < wb::NUM_CLASSES; c++) {
        w[c] = cscore[c] * cscore[c] * cscore[c];
        sum += w[c];
      }
      if (sum > 0) {
        for (int c = 0; c < wb::NUM_CLASSES; c++) {
          r += w[c] * table[c].r; g += w[c] * table[c].g; b += w[c] * table[c].b;
        }
        static float dr = 0, dg = 0, db = 0;            // displayed color, eased toward the target
        dr += COLOR_EASE * (r / sum - dr);
        dg += COLOR_EASE * (g / sum - dg);
        db += COLOR_EASE * (b / sum - db);
        p.r = (uint8_t)dr; p.g = (uint8_t)dg; p.b = (uint8_t)db;
      }
      if (p.brightness > MAX_BRIGHTNESS) p.brightness = MAX_BRIGHTNESS;
      wb::render_pattern(p, now, pixels, NUM_LEDS);
    }
    strip.show(pixels);
  }
}
