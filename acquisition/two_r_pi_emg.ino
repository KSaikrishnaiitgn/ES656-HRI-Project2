
// =====================================================
// Vicharak Shrike Lite (RP2040)
// Dual Channel EMG Acquisition
//
// GPIO26 = Tibialis Anterior (ADC0)
// GPIO27 = Gastrocnemius Medialis (ADC1)
// Sampling frequency = 2000 Hz per channel
// Baud rate = 1000000
// =====================================================

const int emgPin1 = 26;
const int emgPin2 = 27;

const unsigned long samplingInterval = 500; // 500 us

unsigned long nextSampleTime = 0;

int emgValue1 = 0;
int emgValue2 = 0;

void setup() {

  Serial.begin(1000000);

  analogReadResolution(12);

  pinMode(emgPin1, INPUT);
  pinMode(emgPin2, INPUT);

  nextSampleTime = micros() + samplingInterval;
}

void loop() {

  unsigned long currentMicros = micros();

  if ((int32_t)(currentMicros - nextSampleTime) >= 0) {

    // Read both EMG channels
    emgValue1 = analogRead(emgPin1);
    emgValue2 = analogRead(emgPin2);

    // Send both readings in one CSV line
    Serial.print(emgValue1);
    Serial.print(",");
    Serial.println(emgValue2);

    // Schedule next sampling cycle
    nextSampleTime += samplingInterval;

    // Avoid back-to-back samples if serial transmission
    // causes the sampling cycle to fall behind.
    currentMicros = micros();

    if ((int32_t)(currentMicros - nextSampleTime) >= 0) {
      nextSampleTime = currentMicros + samplingInterval;
    }
  }
}
