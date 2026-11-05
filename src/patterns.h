#pragma once
// LED patterns. Pure computation (no hardware calls), so it can be tested on a laptop.

#include <stdint.h>

namespace wb {

enum PatternType : uint8_t { P_SOLID = 0, P_PULSE = 1, P_CHASE = 2, P_STROBE = 3, P_RAINBOW = 4, P_COUNT = 5 };

struct Rgb { uint8_t r, g, b; };

struct PatternCfg {
  uint8_t type;        // PatternType
  uint8_t r, g, b;     // color (ignored by the rainbow pattern)
  uint8_t brightness;  // 0-255
};

inline uint8_t scale8(uint8_t v, uint8_t s) { return (uint8_t)(((uint16_t)v * s + 127) / 255); }

inline Rgb hue_to_rgb(uint8_t h) {
  uint8_t region = h / 43;
  uint8_t rem = (uint8_t)((h - region * 43) * 6);
  switch (region) {
    case 0: return {255, rem, 0};
    case 1: return {(uint8_t)(255 - rem), 255, 0};
    case 2: return {0, 255, rem};
    case 3: return {0, (uint8_t)(255 - rem), 255};
    case 4: return {rem, 0, 255};
    default: return {255, 0, (uint8_t)(255 - rem)};
  }
}

// Fill out[0..n-1] with one frame of the pattern at time now_ms.
inline void render_pattern(const PatternCfg& p, uint32_t now_ms, Rgb* out, int n) {
  for (int i = 0; i < n; i++) {
    Rgb c = {p.r, p.g, p.b};
    switch (p.type) {
      case P_SOLID:
        break;
      case P_PULSE: {
        uint32_t ph = now_ms % 1000;
        uint8_t v = (uint8_t)(ph < 500 ? ph * 255 / 500 : (1000 - ph) * 255 / 500);
        if (v < 20) v = 20;
        c = {scale8(p.r, v), scale8(p.g, v), scale8(p.b, v)};
        break;
      }
      case P_CHASE: {
        int head = (int)((now_ms / 40) % (uint32_t)n);
        int dist = (head - i + n) % n;
        uint8_t v = dist == 0 ? 255 : dist == 1 ? 128 : dist == 2 ? 64 : dist == 3 ? 32 : 0;
        c = {scale8(p.r, v), scale8(p.g, v), scale8(p.b, v)};
        break;
      }
      case P_STROBE:
        if (((now_ms / 60) % 2) == 0) c = {0, 0, 0};
        break;
      case P_RAINBOW:
        c = hue_to_rgb((uint8_t)((now_ms / 8 + (uint32_t)i * 256 / (uint32_t)n) & 0xFF));
        break;
      default:
        c = {0, 0, 0};
    }
    out[i] = {scale8(c.r, p.brightness), scale8(c.g, p.brightness), scale8(c.b, p.brightness)};
  }
}

}  // namespace wb
