#pragma once
// Minimal APA102 / DotStar driver over hardware SPI.
// Nano 33 BLE wiring: strip DI -> D11 (MOSI), strip CI -> D13 (SCK).
// The strip wants 5 V power and 5 V-level signals. Use a level shifter on D11 and D13,
// power the strip from its own 5 V supply, and connect its GND to the Nano's GND.

#include <Arduino.h>
#include <SPI.h>
#include "patterns.h"

namespace wb {

class Apa102 {
 public:
  explicit Apa102(int n) : n_(n) {}

  void begin() { SPI.begin(); }

  void show(const Rgb* px) {
    SPI.beginTransaction(SPISettings(4000000, MSBFIRST, SPI_MODE0));
    for (int i = 0; i < 4; i++) SPI.transfer(0x00);        // start frame
    for (int i = 0; i < n_; i++) {
      SPI.transfer(0xFF);                                   // full global brightness (we scale RGB ourselves)
      SPI.transfer(px[i].b);                                // if colors look wrong, swap the order here
      SPI.transfer(px[i].g);
      SPI.transfer(px[i].r);
    }
    for (int i = 0; i < n_ / 16 + 4; i++) SPI.transfer(0xFF);  // end frame
    SPI.endTransaction();
  }

 private:
  int n_;
};

}  // namespace wb
