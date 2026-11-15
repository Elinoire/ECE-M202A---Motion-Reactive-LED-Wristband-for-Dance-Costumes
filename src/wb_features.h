#pragma once
// Feature extraction for one window of IMU data.
// Must produce the same 90 numbers, in the same order, as window_features() in
// src/train/train.py:  base (56) + shape (13) + gravity (13) + phase (8).
// Which of them the forest actually uses is decided by RECIPE in tools/export_model.py;
// the unused ones are still computed (cheap) so the vector layout never changes.

#include <math.h>
#include <stdint.h>

#include "model.h"   // for MODEL_USES_PHASE / MODEL_USES_DOMFREQ (unused feature groups are skipped)

namespace wb {

constexpr int WIN = 100;          // samples per window (must equal MODEL_WIN in model.h)
constexpr int NCH = 8;            // ax ay az gx gy gz |accel| |gyro|
constexpr int FEAT_DOM_BINS = 12; // must equal DOM_BINS in model.h
constexpr int NFEAT = 90;         // must equal NUM_FEATURES in model.h

// Optional callback: the firmware sets this so the IMU keeps being sampled while features are
// computed (otherwise samples arriving during the ~20 ms computation would be lost).
static void (*poll_hook)() = nullptr;
#define WB_POLL() do { if (wb::poll_hook) wb::poll_hook(); } while (0)

namespace detail {

constexpr float EPS = 1e-9f;

struct Trig { float c[WIN]; float s[WIN]; };
inline const Trig& trig() {
  static Trig t;
  static bool ready = false;
  if (!ready) {
    for (int i = 0; i < WIN; i++) {
      float a = (float)(6.283185307179586 * i / WIN);
      t.c[i] = cosf(a);
      t.s[i] = sinf(a);
    }
    ready = true;
  }
  return t;
}

// DFT bin k of x (same convention as numpy rfft: sum x[n] * exp(-2*pi*i*k*n/WIN)).
inline void dft_bin(const float* x, int k, float& re, float& im) {
  const Trig& t = trig();
  float r = 0.0f, i = 0.0f;
  int idx = 0;                       // (k * n) % WIN without a division; k <= WIN/2 so one subtract is enough
  for (int n = 0; n < WIN; n++) {
    r += x[n] * t.c[idx];
    i -= x[n] * t.s[idx];
    idx += k;
    if (idx >= WIN) idx -= WIN;
  }
  re = r;
  im = i;
}

inline float mean_of(const float* x) {
  float s = 0.0f;
  for (int n = 0; n < WIN; n++) s += x[n];
  return s / (float)WIN;
}

inline float var_of(const float* x, float m) {     // population variance (numpy default)
  float s = 0.0f;
  for (int n = 0; n < WIN; n++) { float d = x[n] - m; s += d * d; }
  return s / (float)WIN;
}

inline float mean_abs_diff(const float* x) {
  float s = 0.0f;
  for (int n = 1; n < WIN; n++) s += fabsf(x[n] - x[n - 1]);
  return s / (float)(WIN - 1);
}

inline float ptp(const float* x) {
  float mn = x[0], mx = x[0];
  for (int n = 1; n < WIN; n++) { if (x[n] < mn) mn = x[n]; if (x[n] > mx) mx = x[n]; }
  return mx - mn;
}

inline float zero_cross_rate(const float* x) {     // fraction of steps where the sign flips
  int c = 0;
  for (int n = 1; n < WIN; n++) c += ((x[n] < 0.0f) != (x[n - 1] < 0.0f));
  return (float)c / (float)(WIN - 1);
}

// Eigen-decomposition of a symmetric 3x3 matrix (cyclic Jacobi).
// evals ascending; evecs[:, j] is the eigenvector for evals[j] (like numpy eigh).
inline void eig3(const float A[3][3], float evals[3], float evecs[3][3]) {
  float a[3][3], v[3][3];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) { a[i][j] = A[i][j]; v[i][j] = (i == j) ? 1.0f : 0.0f; }
  for (int sweep = 0; sweep < 12; sweep++) {
    float off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
    float dia = a[0][0] * a[0][0] + a[1][1] * a[1][1] + a[2][2] * a[2][2];
    if (off <= 1e-12f * dia || off < 1e-36f) break;
    for (int p = 0; p < 2; p++) {
      for (int q = p + 1; q < 3; q++) {
        if (fabsf(a[p][q]) < 1e-30f) continue;
        float theta = (a[q][q] - a[p][p]) / (2.0f * a[p][q]);
        float t = (theta >= 0.0f ? 1.0f : -1.0f) / (fabsf(theta) + sqrtf(theta * theta + 1.0f));
        float c = 1.0f / sqrtf(t * t + 1.0f), s = t * c;
        for (int k = 0; k < 3; k++) {              // columns p, q of A
          float akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (int k = 0; k < 3; k++) {              // rows p, q of A
          float apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (int k = 0; k < 3; k++) {              // accumulate eigenvectors
          float vkp = v[k][p], vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  int order[3] = {0, 1, 2};
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2 - i; j++)
      if (a[order[j]][order[j]] > a[order[j + 1]][order[j + 1]]) {
        int tmp = order[j]; order[j] = order[j + 1]; order[j + 1] = tmp;
      }
  for (int j = 0; j < 3; j++) {
    evals[j] = a[order[j]][order[j]];
    for (int k = 0; k < 3; k++) evecs[k][j] = v[k][order[j]];
  }
}

// cov[i][j] = sum_n s[i][n] * s[j][n] / denom
inline void cov3(const float s[3][WIN], float denom, float cov[3][3]) {
  for (int i = 0; i < 3; i++)
    for (int j = i; j < 3; j++) {
      float acc = 0.0f;
      for (int n = 0; n < WIN; n++) acc += s[i][n] * s[j][n];
      cov[i][j] = cov[j][i] = acc / denom;
    }
}

// ---- base: 56 features -------------------------------------------------
inline void base_features(const float ch[NCH][WIN], float sample_hz, bool do_dft, float* out) {
  float d[WIN];
  for (int c = 0; c < NCH; c++) {
    WB_POLL();
    const float* x = ch[c];
    float m = mean_of(x);
    float var = 0.0f, sq = 0.0f, mn = x[0], mx = x[0];
    for (int n = 0; n < WIN; n++) {
      float v = x[n], dv = v - m;
      var += dv * dv;
      sq += v * v;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
      d[n] = dv;
    }
    out[0 * NCH + c] = m;
    out[1 * NCH + c] = sqrtf(var / (float)WIN);
    out[2 * NCH + c] = mn;
    out[3 * NCH + c] = mx;
    out[4 * NCH + c] = mean_abs_diff(x);
    out[5 * NCH + c] = sq / (float)WIN;
    if (!do_dft) { out[6 * NCH + c] = 0.0f; continue; }   // dominant frequency unused: skip the DFT
    float best = -1.0f;
    int bk = 1;
    for (int k = 1; k <= FEAT_DOM_BINS; k++) {
      float re, im;
      dft_bin(d, k, re, im);
      float p = re * re + im * im;
      if (p > best) { best = p; bk = k; }
    }
    out[6 * NCH + c] = (float)bk * sample_hz / (float)WIN;
  }
}

// ---- shape: 13 features ------------------------------------------------
inline void shape_features(const float ch[NCH][WIN], float* out) {
  static float s[3][WIN];
  float cov[3][3], ev[3], V[3][3], acc_top[3] = {0, 0, 0};
  int k = 0;
  for (int part = 0; part < 2; part++) {           // 0 = accel, 1 = gyro
    WB_POLL();
    for (int i = 0; i < 3; i++) {
      const float* x = ch[part * 3 + i];
      float m = mean_of(x);
      for (int n = 0; n < WIN; n++) s[i][n] = x[n] - m;
    }
    cov3(s, (float)(WIN - 1), cov);                // np.cov uses N-1
    for (int i = 0; i < 3; i++) cov[i][i] += EPS;
    eig3(cov, ev, V);
    float sum = ev[0] + ev[1] + ev[2];
    out[k++] = ev[2] / sum;                        // largest first
    out[k++] = ev[1] / sum;
    out[k++] = ev[0] / sum;
    float sd0 = sqrtf(cov[0][0]), sd1 = sqrtf(cov[1][1]), sd2 = sqrtf(cov[2][2]);
    out[k++] = cov[0][1] / (sd0 * sd1);
    out[k++] = cov[0][2] / (sd0 * sd2);
    out[k++] = cov[1][2] / (sd1 * sd2);
    if (part == 0) for (int i = 0; i < 3; i++) acc_top[i] = V[i][2];
  }
  float g[3], gn = 0.0f;
  for (int i = 0; i < 3; i++) { g[i] = mean_of(ch[i]); gn += g[i] * g[i]; }
  gn = sqrtf(gn) + EPS;
  out[k++] = fabsf((acc_top[0] * g[0] + acc_top[1] * g[1] + acc_top[2] * g[2]) / gn);
}

// ---- gravity frame -----------------------------------------------------
struct GravFrame {
  float av_d[WIN];    // vertical accel, demeaned
  float ap[WIN];      // accel along the main horizontal direction
  float spin[WIN];    // gyro about gravity
  float gperp[WIN];   // gyro magnitude perpendicular to gravity
  float evals[3];     // horizontal covariance eigenvalues, ascending
};

inline void gravity_frame(const float ch[NCH][WIN], GravFrame& gf) {
  static float ahd[WIN][3];
  float gh[3], gn = 0.0f;
  for (int i = 0; i < 3; i++) { gh[i] = mean_of(ch[i]); gn += gh[i] * gh[i]; }
  gn = sqrtf(gn) + EPS;
  for (int i = 0; i < 3; i++) gh[i] /= gn;

  float avm = 0.0f, ahm[3] = {0, 0, 0};
  for (int n = 0; n < WIN; n++) {
    float av = ch[0][n] * gh[0] + ch[1][n] * gh[1] + ch[2][n] * gh[2];
    gf.av_d[n] = av;
    avm += av;
    for (int i = 0; i < 3; i++) { ahd[n][i] = ch[i][n] - av * gh[i]; ahm[i] += ahd[n][i]; }
  }
  avm /= (float)WIN;
  for (int i = 0; i < 3; i++) ahm[i] /= (float)WIN;
  float cov[3][3] = {{0}};
  for (int n = 0; n < WIN; n++) {
    gf.av_d[n] -= avm;
    for (int i = 0; i < 3; i++) ahd[n][i] -= ahm[i];
    for (int i = 0; i < 3; i++)
      for (int j = i; j < 3; j++) cov[i][j] += ahd[n][i] * ahd[n][j];
  }
  for (int i = 0; i < 3; i++)
    for (int j = i; j < 3; j++) { cov[i][j] /= (float)WIN; cov[j][i] = cov[i][j]; }
  float V[3][3];
  eig3(cov, gf.evals, V);
  for (int n = 0; n < WIN; n++) {
    gf.ap[n] = ahd[n][0] * V[0][2] + ahd[n][1] * V[1][2] + ahd[n][2] * V[2][2];
    float gx = ch[3][n], gy = ch[4][n], gz = ch[5][n];
    float sp = gx * gh[0] + gy * gh[1] + gz * gh[2];
    gf.spin[n] = sp;
    float px = gx - sp * gh[0], py = gy - sp * gh[1], pz = gz - sp * gh[2];
    gf.gperp[n] = sqrtf(px * px + py * py + pz * pz);
  }
}

// ---- gravity: 13 features ----------------------------------------------
inline void gravity_features(const GravFrame& gf, float* out) {
  float e1 = fmaxf(gf.evals[2], 0.0f), e2 = fmaxf(gf.evals[1], 0.0f);
  float var_v = var_of(gf.av_d, mean_of(gf.av_d));
  float dot = 0.0f, na = 0.0f, nb = 0.0f;
  for (int n = 0; n < WIN; n++) {
    dot += gf.av_d[n] * gf.ap[n];
    na += gf.av_d[n] * gf.av_d[n];
    nb += gf.ap[n] * gf.ap[n];
  }
  float corr_vh = fabsf(dot) / (sqrtf(na) * sqrtf(nb) + EPS);
  float sm = mean_of(gf.spin), ss = sqrtf(var_of(gf.spin, sm));
  float sabs = 0.0f;
  for (int n = 0; n < WIN; n++) sabs += fabsf(gf.spin[n]);
  sabs /= (float)WIN;
  float gm = mean_of(gf.gperp), gs = sqrtf(var_of(gf.gperp, gm));
  int k = 0;
  out[k++] = sqrtf(var_v);
  out[k++] = sqrtf(e1 + e2);
  out[k++] = var_v / (var_v + e1 + e2 + EPS);
  out[k++] = e1 / (e1 + e2 + EPS);
  out[k++] = sm;
  out[k++] = ss;
  out[k++] = sabs;
  out[k++] = gm;
  out[k++] = gs;
  out[k++] = corr_vh;
  out[k++] = ptp(gf.av_d);
  out[k++] = ptp(gf.ap);
  out[k++] = mean_abs_diff(gf.av_d);
}

// |sin| and |cos| of the phase difference at the shared dominant frequency.
inline void phase_pair(const float* x, const float* y, float& sn, float& cs) {
  float best = -1.0f, bxr = 0, bxi = 0, byr = 0, byi = 0;
  for (int k = 1; k <= FEAT_DOM_BINS; k++) {
    WB_POLL();
    float xr, xi, yr, yi;
    dft_bin(x, k, xr, xi);
    dft_bin(y, k, yr, yi);
    float p = xr * xr + xi * xi + yr * yr + yi * yi;
    if (p > best) { best = p; bxr = xr; bxi = xi; byr = yr; byi = yi; }
  }
  float re = bxr * byr + bxi * byi;                // X * conj(Y)
  float im = bxi * byr - bxr * byi;
  float d = atan2f(im, re);
  sn = fabsf(sinf(d));
  cs = fabsf(cosf(d));
}

// ---- phase: 8 features -------------------------------------------------
inline void phase_features(const float ch[NCH][WIN], const GravFrame& gf, float* out) {
  static float a[3][WIN], s1[WIN], s2[WIN];
  float cov[3][3], ev[3], V[3][3];
  for (int i = 0; i < 3; i++) {
    float m = mean_of(ch[i]);
    for (int n = 0; n < WIN; n++) a[i][n] = ch[i][n] - m;
  }
  cov3(a, (float)WIN, cov);
  eig3(cov, ev, V);
  for (int n = 0; n < WIN; n++) {
    s1[n] = a[0][n] * V[0][2] + a[1][n] * V[1][2] + a[2][n] * V[2][2];
    s2[n] = a[0][n] * V[0][1] + a[1][n] * V[1][1] + a[2][n] * V[2][1];
  }
  int k = 0;
  phase_pair(s1, s2, out[k], out[k + 1]); k += 2;
  phase_pair(gf.av_d, gf.ap, out[k], out[k + 1]); k += 2;

  float best = -1.0f, pk = 0.0f, total = 0.0f;     // power in the main peak of s1
  for (int b = 1; b <= WIN / 2; b++) {
    WB_POLL();
    float re, im;
    dft_bin(s1, b, re, im);
    float p = re * re + im * im;
    total += p;
    if (b <= FEAT_DOM_BINS && p > best) { best = p; pk = p; }
  }
  out[k++] = pk / (total + EPS);
  out[k++] = zero_cross_rate(gf.av_d);
  out[k++] = zero_cross_rate(gf.ap);
  out[k++] = zero_cross_rate(s1);
}

}  // namespace detail

// Optional per-stage timing: add  -DWB_PROFILE  to build_flags, then main.cpp prints prof_us.
#ifdef WB_PROFILE
#include <Arduino.h>
static uint32_t prof_us[5];   // 0 channel setup, 1 base, 2 shape, 3 gravity, 4 phase
#define WB_PROF_START() uint32_t wb_t0 = micros()
#define WB_PROF_MARK(i) do { uint32_t wb_t = micros(); prof_us[i] = wb_t - wb_t0; wb_t0 = wb_t; } while (0)
#else
#define WB_PROF_START() do {} while (0)
#define WB_PROF_MARK(i) do {} while (0)
#endif

// window: WIN rows, oldest first, each row = ax ay az gx gy gz (g and deg/s, like the CSVs)
// out:    NFEAT features in the order of FEATURE_NAMES in train.py (see model.h)
inline void compute_features(const float window[][6], float sample_hz, float out[NFEAT]) {
  static float ch[NCH][WIN];
  static detail::GravFrame gf;
  WB_PROF_START();
  for (int n = 0; n < WIN; n++) {
    for (int c = 0; c < 6; c++) ch[c][n] = window[n][c];
    ch[6][n] = sqrtf(ch[0][n] * ch[0][n] + ch[1][n] * ch[1][n] + ch[2][n] * ch[2][n]);
    ch[7][n] = sqrtf(ch[3][n] * ch[3][n] + ch[4][n] * ch[4][n] + ch[5][n] * ch[5][n]);
  }
  WB_PROF_MARK(0);
  WB_POLL();
  detail::base_features(ch, sample_hz, MODEL_USES_DOMFREQ, out);              // 0..55
  WB_PROF_MARK(1);
  WB_POLL();
  detail::shape_features(ch, out + 56);                   // 56..68
  WB_PROF_MARK(2);
  WB_POLL();
  detail::gravity_frame(ch, gf);
  WB_POLL();
  detail::gravity_features(gf, out + 69);                 // 69..81
  WB_PROF_MARK(3);
  WB_POLL();
  if (MODEL_USES_PHASE) {
    detail::phase_features(ch, gf, out + 82);             // 82..89
  } else {
    for (int i = 82; i < NFEAT; i++) out[i] = 0.0f;       // unused by the forest, skip the DFT work
  }
  WB_PROF_MARK(4);
}

}  // namespace wb
