// RAMPS 1.4 + Mega 2560, X stepper
const int X_STEP = 54;
const int X_DIR = 55;
const int EN = 38;

long us_per_step = 1000;  // default speed
bool enabled = false;

void setup() {
  pinMode(X_STEP, OUTPUT);
  pinMode(X_DIR, OUTPUT);
  pinMode(EN, OUTPUT);
  digitalWrite(EN, HIGH);

  Serial.begin(115200);
}

void loop() {
  if (Serial.available()) handleSerial();

  if (!enabled) return;

  digitalWrite(X_STEP, HIGH);
  delayMicroseconds(us_per_step);
  digitalWrite(X_STEP, LOW);
  delayMicroseconds(us_per_step);
}

void handleSerial() {
  String cmd = Serial.readStringUntil('\n');

  if (cmd.startsWith("DIR")) {
    // DIR +1 or DIR -1
    if (cmd.indexOf("-1") > 0) digitalWrite(X_DIR, HIGH);
    else digitalWrite(X_DIR, LOW);
  }

  if (cmd.startsWith("SPD")) {
    // SPD <micros>
    // lower micros = faster
    long v = cmd.substring(4).toInt();
    if (v > 20) us_per_step = v;
  }

  if (cmd.startsWith("EN")) {
    enabled = true;
    digitalWrite(EN, LOW);
  }

  if (cmd.startsWith("DIS")) {
    enabled = false;
    digitalWrite(EN, HIGH);
  }
}