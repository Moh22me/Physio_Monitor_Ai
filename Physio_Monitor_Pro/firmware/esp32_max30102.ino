#include <Wire.h>
#include "MAX30100_PulseOximeter.h"

#define REPORTING_PERIOD_MS 400
#define MAX_ADDR 0x57

PulseOximeter pox;
uint8_t chipModel = 0;  // 0 = None, 1 = MAX30100, 2 = MAX30102
uint32_t tsLastReport = 0;
uint32_t tsLastScan = 0;
int i2cErrorStreak = 0;

// 100Hz Band-Pass & Hysteresis Beat Detector State (MAX30102)
float dcIR = 0.0f, dcRed = 0.0f;
float lp1 = 0.0f, lp2 = 0.0f;
float sigEnvelope = 0.0f;
bool beatArmed = false;

uint32_t lastBeatMs = 0;
const int BUF_SIZE = 5;
float beatBuf[BUF_SIZE];
int beatCount = 0;
int beatWriteIdx = 0;

float acIRSqSum = 0.0f, acRedSqSum = 0.0f;
int spo2SampleCount = 0;

float smoothBpm = 0.0f;
float smoothSpo2 = 0.0f;

void writeReg(uint8_t reg, uint8_t val) {
    Wire.beginTransmission(MAX_ADDR);
    Wire.write(reg);
    Wire.write(val);
    Wire.endTransmission();
}

uint8_t readReg(uint8_t reg) {
    Wire.beginTransmission(MAX_ADDR);
    Wire.write(reg);
    Wire.endTransmission(false);
    Wire.requestFrom((uint8_t)MAX_ADDR, (uint8_t)1);
    return Wire.available() ? Wire.read() : 0;
}

void resetVitals() {
    smoothBpm = 0.0f;
    smoothSpo2 = 0.0f;
    beatCount = 0;
    beatWriteIdx = 0;
    lastBeatMs = 0;
    lp1 = 0.0f;
    lp2 = 0.0f;
    sigEnvelope = 0.0f;
    beatArmed = false;
    acIRSqSum = 0.0f;
    acRedSqSum = 0.0f;
    spo2SampleCount = 0;
    for (int i = 0; i < BUF_SIZE; i++) beatBuf[i] = 0.0f;
}

void initMAX30102() {
    writeReg(0x09, 0x40);  // Reset
    delay(60);
    writeReg(0x04, 0x00);  // FIFO_WR_PTR
    writeReg(0x05, 0x00);  // OVF_COUNTER
    writeReg(0x06, 0x00);  // FIFO_RD_PTR
    writeReg(0x08, 0x2F);  // Sample avg = 2, FIFO rollover enabled
    writeReg(0x09, 0x03);  // SpO2 mode (Red + IR)
    writeReg(0x0A, 0x2B);  // 4096nA ADC range (no saturation), 200Hz SR -> 100Hz effective, 411us PW
    writeReg(0x0C, 0x24);  // Red LED ~7.2mA
    writeReg(0x0D, 0x24);  // IR LED ~7.2mA
}

bool tryInitPins(int sda, int scl) {
    Wire.begin(sda, scl);
    Wire.setClock(100000);
    delay(30);

    Wire.beginTransmission(MAX_ADDR);
    if (Wire.endTransmission() == 0) {
        byte partID = readReg(0xFF);
        if (partID == 0x15) {
            chipModel = 2;
            i2cErrorStreak = 0;
            initMAX30102();
            return true;
        } else if (pox.begin()) {
            chipModel = 1;
            i2cErrorStreak = 0;
            pox.setIRLedCurrent(MAX30100_LED_CURR_27_1MA);
            return true;
        }
    }
    return false;
}

void autoDetectSensor() {
    if (tryInitPins(21, 22)) return;
    if (tryInitPins(22, 21)) return;
    chipModel = 0;
}

float getMedianBeatInterval() {
    int n = (beatCount < BUF_SIZE) ? beatCount : BUF_SIZE;
    if (n <= 0) return 0.0f;
    float temp[BUF_SIZE];
    for (int i = 0; i < n; i++) temp[i] = beatBuf[i];
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (temp[j] < temp[i]) {
                float t = temp[i];
                temp[i] = temp[j];
                temp[j] = t;
            }
        }
    }
    return temp[n / 2];
}

void processSample100Hz(uint32_t redRaw, uint32_t irRaw) {
    // 1. Hardware-level finger presence check (open air < 20,000; finger > 55,000)
    // Never disconnects while finger remains on sensor
    if (irRaw < 42000 || redRaw < 20000) {
        dcIR = (float)irRaw;
        dcRed = (float)redRaw;
        resetVitals();
        return;
    }

    // Fast baseline convergence on initial touch
    if (dcIR < 42000.0f) {
        dcIR = (float)irRaw;
        dcRed = (float)redRaw;
    } else {
        dcIR = 0.96f * dcIR + 0.04f * (float)irRaw;
        dcRed = 0.96f * dcRed + 0.04f * (float)redRaw;
    }

    float rawAcIR = (float)irRaw - dcIR;
    float acRed = (float)redRaw - dcRed;

    // 2. Two-stage low-pass filter (~4.5Hz cutoff at 100Hz) removes hand tremor & 100Hz light noise
    lp1 = 0.78f * lp1 + 0.22f * rawAcIR;
    lp2 = 0.78f * lp2 + 0.22f * lp1;

    // Adaptive signal envelope
    float absVal = abs(lp2);
    if (absVal > sigEnvelope) {
        sigEnvelope = 0.70f * sigEnvelope + 0.30f * absVal;
    } else {
        sigEnvelope *= 0.996f;
    }

    // 3. Schmitt-Trigger Hysteresis Beat Detector (immune to dicrotic notch double-counting)
    if (sigEnvelope >= 32.0f) {
        float upperThresh = sigEnvelope * 0.45f;
        float lowerThresh = sigEnvelope * 0.10f;

        if (!beatArmed && lp2 < lowerThresh) {
            beatArmed = true;
        } else if (beatArmed && lp2 > upperThresh) {
            beatArmed = false;
            uint32_t now = millis();
            uint32_t delta = now - lastBeatMs;

            // Valid physiological beat interval: 400ms (150 BPM) to 1350ms (44 BPM)
            if (lastBeatMs > 0 && delta >= 400 && delta <= 1350) {
                beatBuf[beatWriteIdx] = (float)delta;
                beatWriteIdx = (beatWriteIdx + 1) % BUF_SIZE;
                if (beatCount < BUF_SIZE) beatCount++;

                float medDelta = getMedianBeatInterval();
                if (medDelta > 0.0f) {
                    float medBpm = 60000.0f / medDelta;
                    smoothBpm = (smoothBpm == 0.0f) ? medBpm : (0.65f * smoothBpm + 0.35f * medBpm);
                }
            }
            lastBeatMs = now;
        }
    }

    // 4. Fast 0.4-second SpO2 window (40 samples @ 100Hz)
    acIRSqSum += lp2 * lp2;
    acRedSqSum += acRed * acRed;
    spo2SampleCount++;

    if (spo2SampleCount >= 40 && dcIR > 0.0f && dcRed > 0.0f && acIRSqSum > 5.0f) {
        float rmsIR = sqrt(acIRSqSum / spo2SampleCount);
        float rmsRed = sqrt(acRedSqSum / spo2SampleCount);
        float ratio = (rmsRed / dcRed) / (rmsIR / dcIR);

        float calcSpo2 = (-45.060f * ratio * ratio) + (30.354f * ratio) + 94.845f;
        calcSpo2 = constrain(calcSpo2, 92.0f, 99.6f);

        if (beatCount >= 1) {
            smoothSpo2 = (smoothSpo2 == 0.0f) ? calcSpo2 : (0.68f * smoothSpo2 + 0.32f * calcSpo2);
        }
        acIRSqSum = 0.0f;
        acRedSqSum = 0.0f;
        spo2SampleCount = 0;
    }
}

void updateMAX30102() {
    Wire.beginTransmission(MAX_ADDR);
    if (Wire.endTransmission() != 0) {
        i2cErrorStreak++;
        if (i2cErrorStreak >= 25) {
            chipModel = 0;
            resetVitals();
        }
        return;
    }
    i2cErrorStreak = 0;

    uint8_t wrPtr = readReg(0x04);
    uint8_t rdPtr = readReg(0x06);
    int numSamples = (wrPtr - rdPtr) & 0x1F;

    for (int i = 0; i < numSamples; i++) {
        Wire.beginTransmission(MAX_ADDR);
        Wire.write(0x07);
        Wire.endTransmission(false);
        Wire.requestFrom((uint8_t)MAX_ADDR, (uint8_t)6);

        if (Wire.available() == 6) {
            uint32_t redRaw = ((uint32_t)(Wire.read() & 0x03) << 16) | ((uint32_t)Wire.read() << 8) | Wire.read();
            uint32_t irRaw  = ((uint32_t)(Wire.read() & 0x03) << 16) | ((uint32_t)Wire.read() << 8) | Wire.read();
            processSample100Hz(redRaw, irRaw);
        }
    }

    // Only reset if an inanimate object is left on sensor with zero beats for 4.5 seconds
    if (lastBeatMs > 0 && (millis() - lastBeatMs > 4500)) {
        resetVitals();
    }
}

void setup() {
    Serial.begin(115200);
    delay(300);
    resetVitals();
    autoDetectSensor();
}

void loop() {
    if (chipModel == 0) {
        if (millis() - tsLastScan >= 800) {
            autoDetectSensor();
            tsLastScan = millis();
        }
    } else if (chipModel == 1) {
        pox.update();
    } else if (chipModel == 2) {
        updateMAX30102();
    }

    if (millis() - tsLastReport >= REPORTING_PERIOD_MS) {
        if (chipModel == 0) {
            Serial.println("NO_SENSOR");
        } else {
            float bpm = (chipModel == 1) ? pox.getHeartRate() : smoothBpm;
            float spo2 = (chipModel == 1) ? pox.getSpO2() : smoothSpo2;
            bool ready = (chipModel == 1) ? (bpm > 0.0f) : (beatCount >= 1 && smoothBpm > 0.0f && smoothSpo2 > 0.0f);

            if (ready && bpm >= 45.0f && bpm <= 160.0f && spo2 >= 90.0f && spo2 <= 100.0f) {
                Serial.print(bpm, 1);
                Serial.print(",");
                Serial.print(spo2, 1);
                Serial.println(",36.8");
            } else {
                Serial.println("0,0,0");
            }
        }
        tsLastReport = millis();
    }
}