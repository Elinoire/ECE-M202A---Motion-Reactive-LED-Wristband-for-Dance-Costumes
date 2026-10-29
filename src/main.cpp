#include <Arduino.h>

// IMU data logger for the Arduino Nano 33 BLE Sense Rev2 (BMI270 + BMM150).
// Streams: ms,ax,ay,az,gx,gy,gz   (accel in g, gyro in deg/s) at the IMU's ~100 Hz.
// Library needed: Arduino_BMI270_BMM150 (install from Library Manager).

#include <Arduino_BMI270_BMM150.h>

void setup() {
  Serial.begin(115200);
  while (!Serial) { }           // wait for the computer to open the serial port

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
  float ax, ay, az, gx, gy, gz;

  // The sensor produces a new sample about every 10 ms; only print when both are ready.
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);

    char line[96];
    snprintf(line, sizeof(line), "%lu,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f",
             millis(), ax, ay, az, gx, gy, gz);
    Serial.println(line);
  }
}
