#include <Servo.h>

Servo servoMotor;
const int servoPin = 9;
char receivedChar;

void setup() {
  Serial.begin(9600);
  servoMotor.attach(servoPin);
  servoMotor.write(0); // Initial position
}

void loop() {
  if (Serial.available()) {
    receivedChar = Serial.read();
    if (receivedChar == 'A') {
      servoMotor.write(90); // Rotate servo
      delay(2000); // Hold for 2 sec
      servoMotor.write(0); // Return to initial position
    }
  }
}

