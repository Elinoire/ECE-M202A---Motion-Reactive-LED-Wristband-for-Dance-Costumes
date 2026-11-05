#include <Arduino.h>

// IMU logger + LED command receiver for the Arduino Nano 33 BLE Sense Rev2.
//
// Streams:  ms,ax,ay,az,gx,gy,gz   (accel in g, gyro in deg/s) at the IMU's rate.
// Accepts one-letter commands over USB serial to set the built-in RGB LED:
//   S = still (off)   L = leftright (blue)   U = updown (green)   R = rotate (magenta)
//
// Library needed: Arduino_BMI270_BMM150 (installed automatically by PlatformIO).

#include <Arduino_BMI270_BMM150.h>

// The built-in RGB LED is active-low on this board (LOW = on).
void setLed(bool r, bool g, bool b) {
  digitalWrite(LEDR, r ? LOW : HIGH);
  digitalWrite(LEDG, g ? LOW : HIGH);
  digitalWrite(LEDB, b ? LOW : HIGH);
}

void handleCommand(char c) {
  switch (c) {
    case 'S': setLed(0, 0, 0); break;   // still
    case 'L': setLed(0, 0, 1); break;   // leftright
    case 'U': setLed(0, 1, 0); break;   // updown
    case 'R': setLed(1, 0, 1); break;   // rotate
    default: break;                     // ignore anything else
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { }           // wait for the computer to open the serial port

  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
  setLed(0, 0, 0);

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU");
    while (true) { }
  }

  Serial.print("# accel Hz: ");
  Serial.println(IMU.accelerationSampleRate());
  Serial.print("# gyro Hz: ");
  Serial.println(IMU.gyroscopeSampleRate());
  Serial.println("ms,ax,ay,az,gx,gy,gz");
}

void loop() {
  while (Serial.available()) {
    handleCommand((char)Serial.read());
  }

  float ax, ay, az, gx, gy, gz;

  // The sensor produces a new sample about every 11 ms; only print when both are ready.
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);

    char line[96];
    snprintf(line, sizeof(line), "%lu,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f",
             millis(), ax, ay, az, gx, gy, gz);
    Serial.println(line);
  }
}
