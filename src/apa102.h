#pragma once
// Minimal APA102 / DotStar driver over hardware SPI.
// Nano 33 BLE wiring: strip DI -> D11 (MOSI), strip CI -> D13 (SCK).
// The strip wants 5 V power and 5 V-level signals. Use a level shifter on D11 and D13,
// power the strip from its own 5 V supply, and connect its GND to the Nano's GND.
//
// show() builds the whole frame in a buffer and sends it with ONE SPI call. Sending it a byte at a
// time (SPI.transfer(byte) in a loop) took ~5.6 ms for 60 LEDs on this core; one buffered transfer
// takes about 0.5 ms at 4 MHz.

#include <Arduino.h>
#include <SPI.h>
#include "patterns.h"

namespace wb {

class Apa102 {
 public:
  explicit Apa102(int n) : n_(n), len_(4 + 4 * n + n / 16 + 4), buf_(new uint8_t[len_]) {}

  void begin() { SPI.begin(); }

  void show(const Rgb* px) {
    uint8_t* p = buf_;
    for (int i = 0; i < 4; i++) *p++ = 0x00;                // start frame
    for (int i = 0; i < n_; i++) {
      *p++ = 0xFF;                                          // full global brightness (we scale RGB ourselves)
      *p++ = px[i].b;                                       // if colors look wrong, swap the order here
      *p++ = px[i].g;
      *p++ = px[i].r;
    }
    for (int i = 0; i < n_ / 16 + 4; i++) *p++ = 0xFF;      // end frame
    SPI.beginTransaction(SPISettings(4000000, MSBFIRST, SPI_MODE0));
    SPI.transfer(buf_, len_);                               // one call; buf_ is refilled on every show()
    SPI.endTransaction();
  }

 private:
  int n_;
  size_t len_;
  uint8_t* buf_;
};

}  // namespace wb
