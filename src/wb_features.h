#pragma once
// Feature extraction for one window of IMU data.
// Must produce the same 56 numbers, in the same order, as features() in tools/wbcommon.py.

#include <math.h>
#include <stdint.h>

namespace wb {

constexpr int WIN = 50;          // samples per window
constexpr int NCH = 8;           // ax ay az gx gy gz |accel| |gyro|
constexpr int NFEAT = 7 * NCH;   // mean, std, min, max, mean|diff|, mean square, dominant frequency

// window: WIN rows, oldest first, each row = ax ay az gx gy gz
// out:    NFEAT features (7 blocks of NCH, one value per channel in each block)
inline void compute_features(const float window[][6], float sample_hz, float out[NFEAT]) {
  static float cos_t[WIN], sin_t[WIN];
  static bool tables_ready = false;
  if (!tables_ready) {
    for (int i = 0; i < WIN; i++) {
      float a = 2.0f * 3.14159265358979f * (float)i / (float)WIN;
      cos_t[i] = cosf(a);
      sin_t[i] = sinf(a);
    }
    tables_ready = true;
  }

  float x[WIN][NCH];
  for (int n = 0; n < WIN; n++) {
    for (int c = 0; c < 6; c++) x[n][c] = window[n][c];
    x[n][6] = sqrtf(x[n][0] * x[n][0] + x[n][1] * x[n][1] + x[n][2] * x[n][2]);
    x[n][7] = sqrtf(x[n][3] * x[n][3] + x[n][4] * x[n][4] + x[n][5] * x[n][5]);
  }

  for (int c = 0; c < NCH; c++) {
    float mean = 0.0f;
    for (int n = 0; n < WIN; n++) mean += x[n][c];
    mean /= (float)WIN;

    float var = 0.0f, sq = 0.0f, dsum = 0.0f;
    float mn = x[0][c], mx = x[0][c];
    for (int n = 0; n < WIN; n++) {
      float v = x[n][c];
      float d = v - mean;
      var += d * d;
      sq += v * v;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
      if (n > 0) dsum += fabsf(v - x[n - 1][c]);
    }

    out[0 * NCH + c] = mean;
    out[1 * NCH + c] = sqrtf(var / (float)WIN);
    out[2 * NCH + c] = mn;
    out[3 * NCH + c] = mx;
    out[4 * NCH + c] = dsum / (float)(WIN - 1);
    out[5 * NCH + c] = sq / (float)WIN;

    // Dominant frequency: strongest DFT bin among k = 1 .. WIN/2 (DC excluded).
    float best_power = -1.0f;
    int best_k = 1;
    for (int k = 1; k <= WIN / 2; k++) {
      float re = 0.0f, im = 0.0f;
      for (int n = 0; n < WIN; n++) {
        int idx = (k * n) % WIN;
        float d = x[n][c] - mean;
        re += d * cos_t[idx];
        im += d * sin_t[idx];
      }
      float p = re * re + im * im;
      if (p > best_power) { best_power = p; best_k = k; }
    }
    out[6 * NCH + c] = (float)best_k * sample_hz / (float)WIN;
  }
}

}  // namespace wb
