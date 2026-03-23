/*
 * ═══════════════════════════════════════════════════════════════════════
 *  smart_crib_arduino.ino — Smart Crib "Spinal Cord"
 * ═══════════════════════════════════════════════════════════════════════
 *
 *  Hardware:
 *    • DS18B20   — Waterproof temperature sensor (OneWire)
 *    • Pulse Sensor — Analog heart-rate sensor
 *    • NEMA 23   — Stepper motor via TB6600 driver
 *
 *  Communication:
 *    • Serial @ 115200 baud
 *    • TX (to Pi)  :  "BPM,Temp\n"  every 1 000 ms
 *    • RX (from Pi):  'C' = Cry (start rocking)
 *                      'S' = Stop (begin 3-min cooldown, then idle)
 *
 *  State Machine:
 *    IDLE ──['C']──▶ ROCKING ──['S']──▶ COOLDOWN (180 s) ──▶ IDLE
 *    ROCKING ──['S']──▶ COOLDOWN
 *    Any state ──['C']──▶ ROCKING
 *
 *  CRITICAL: Zero delay() calls. All timing via millis().
 *
 *  Libraries required:
 *    - AccelStepper       (for smooth non-blocking motor control)
 *    - OneWire            (for DS18B20)
 *    - DallasTemperature  (for DS18B20)
 *
 *  Author : Senior Embedded Systems Engineering Team
 *  Target : Arduino Mega 2560 / Uno
 * ═══════════════════════════════════════════════════════════════════════
 */

#include <AccelStepper.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// ═══════════════════════════════════════════════════════════════════════
// PIN DEFINITIONS
// ═══════════════════════════════════════════════════════════════════════

// -- TB6600 Stepper Driver --
#define MOTOR_PUL_PIN   3   // Pulse (Step) — connect to TB6600 PUL+
#define MOTOR_DIR_PIN   4   // Direction    — connect to TB6600 DIR+
#define MOTOR_ENA_PIN   5   // Enable       — connect to TB6600 ENA+
                            // GND pins → Arduino GND

// -- DS18B20 Temperature Sensor --
#define ONEWIRE_PIN     7   // Data pin (with 4.7kΩ pull-up to 5V)

// -- Pulse Sensor --
#define PULSE_PIN       A0  // Analog input from Pulse Sensor signal wire

// ═══════════════════════════════════════════════════════════════════════
// CONFIGURATION CONSTANTS
// ═══════════════════════════════════════════════════════════════════════

#define SERIAL_BAUD       115200

// Timing intervals (milliseconds)
#define SENSOR_INTERVAL   1000UL   // Report sensors every 1 second
#define TEMP_INTERVAL     2000UL   // DS18B20 read every 2s (750ms conversion)
#define COOLDOWN_DURATION 180000UL // 3 minutes in milliseconds

// Motor rocking parameters
#define ROCK_AMPLITUDE    400      // Steps per half-swing (adjust for crib)
#define ROCK_MAX_SPEED    600.0    // Steps per second (gentle)
#define ROCK_ACCELERATION 300.0    // Steps per second² (smooth accel/decel)

// Pulse Sensor peak detection
#define PULSE_THRESHOLD   550      // Analog reading threshold for beat detection
#define PULSE_MIN_IBI     300UL    // Minimum 300 ms between beats (200 BPM cap)
#define PULSE_MAX_IBI     2000UL   // Maximum 2000 ms between beats (30 BPM floor)

// ═══════════════════════════════════════════════════════════════════════
// STATE MACHINE
// ═══════════════════════════════════════════════════════════════════════

enum CribState {
    STATE_IDLE,       // Motor stopped, waiting for command
    STATE_ROCKING,    // Motor oscillating back and forth
    STATE_COOLDOWN    // Motor still rocking, but timer running → IDLE
};

CribState currentState = STATE_IDLE;

// ═══════════════════════════════════════════════════════════════════════
// OBJECT INSTANCES
// ═══════════════════════════════════════════════════════════════════════

// AccelStepper in DRIVER mode (Step + Dir pins)
AccelStepper stepper(AccelStepper::DRIVER, MOTOR_PUL_PIN, MOTOR_DIR_PIN);

// DS18B20
OneWire           oneWire(ONEWIRE_PIN);
DallasTemperature tempSensor(&oneWire);

// ═══════════════════════════════════════════════════════════════════════
// TIMING VARIABLES (millis-based, no delay)
// ═══════════════════════════════════════════════════════════════════════

unsigned long lastSensorReport  = 0;   // Last time we sent "BPM,Temp\n"
unsigned long lastTempRequest   = 0;   // Last DS18B20 conversion request
unsigned long cooldownStartTime = 0;   // When cooldown began

// ═══════════════════════════════════════════════════════════════════════
// SENSOR VARIABLES
// ═══════════════════════════════════════════════════════════════════════

// -- Temperature --
float currentTemp = 0.0;
bool  tempReady   = false;

// -- Pulse Sensor (non-blocking peak detection) --
int           currentBPM       = 0;
unsigned long lastBeatTime     = 0;
bool          pulseHigh        = false;   // Debounce flag
unsigned long beatIntervals[5] = {0};     // Rolling buffer for averaging
int           beatIndex        = 0;
int           beatCount        = 0;

// -- Motor direction --
bool rockingForward = true;   // true = moving to +AMPLITUDE, false = -AMPLITUDE

// ═══════════════════════════════════════════════════════════════════════
// SETUP
// ═══════════════════════════════════════════════════════════════════════

void setup() {
    // -- Serial --
    Serial.begin(SERIAL_BAUD);
    while (!Serial) { ; }  // Wait for serial on USB boards

    // -- Motor --
    pinMode(MOTOR_ENA_PIN, OUTPUT);
    disableMotor();  // Start with motor disabled (saves power)

    stepper.setMaxSpeed(ROCK_MAX_SPEED);
    stepper.setAcceleration(ROCK_ACCELERATION);
    stepper.setCurrentPosition(0);

    // -- Temperature Sensor --
    tempSensor.begin();
    tempSensor.setResolution(12);          // 12-bit resolution (~750 ms)
    tempSensor.setWaitForConversion(false); // NON-BLOCKING mode
    tempSensor.requestTemperatures();       // Start first conversion
    lastTempRequest = millis();

    // -- Pulse Sensor --
    pinMode(PULSE_PIN, INPUT);

    // -- Ready --
    Serial.println("CRIB_READY");
}

// ═══════════════════════════════════════════════════════════════════════
// MAIN LOOP — Zero delay() calls
// ═══════════════════════════════════════════════════════════════════════

void loop() {
    unsigned long now = millis();

    // ── 1. Read Serial Commands from Raspberry Pi ─────────────────────
    handleSerialCommands();

    // ── 2. Read Sensors (non-blocking) ────────────────────────────────
    readPulseSensor(now);
    readTemperature(now);

    // ── 3. Report Sensors to Pi every 1000 ms ─────────────────────────
    if (now - lastSensorReport >= SENSOR_INTERVAL) {
        lastSensorReport = now;
        reportSensors();
    }

    // ── 4. State Machine Logic ────────────────────────────────────────
    switch (currentState) {

        case STATE_IDLE:
            // Motor is disabled, nothing to do.
            break;

        case STATE_ROCKING:
            // Smooth oscillation — AccelStepper handles acceleration
            runRockingMotion();
            break;

        case STATE_COOLDOWN:
            // Motor still rocking, but check if cooldown expired
            runRockingMotion();
            if (now - cooldownStartTime >= COOLDOWN_DURATION) {
                transitionToIdle();
            }
            break;
    }

    // ── 5. Run stepper (MUST be called every loop iteration) ──────────
    // AccelStepper::run() computes the next step pulse if needed.
    // It is inherently non-blocking.
    stepper.run();
}

// ═══════════════════════════════════════════════════════════════════════
// SERIAL COMMAND HANDLER
// ═══════════════════════════════════════════════════════════════════════

void handleSerialCommands() {
    while (Serial.available() > 0) {
        char cmd = Serial.read();

        switch (cmd) {
            case 'C':
                // Cry detected → start rocking immediately
                transitionToRocking();
                break;

            case 'S':
                // Stop/Silence → begin 3-minute cooldown
                transitionToCooldown();
                break;

            default:
                // Ignore unknown characters (\n, \r, etc.)
                break;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// STATE TRANSITIONS
// ═══════════════════════════════════════════════════════════════════════

void transitionToRocking() {
    if (currentState != STATE_ROCKING) {
        currentState = STATE_ROCKING;
        enableMotor();
        rockingForward = true;
        stepper.moveTo(ROCK_AMPLITUDE);
    }
}

void transitionToCooldown() {
    if (currentState == STATE_ROCKING || currentState == STATE_COOLDOWN) {
        currentState    = STATE_COOLDOWN;
        cooldownStartTime = millis();
        // Motor continues to rock during cooldown (gentle wind-down)
    }
}

void transitionToIdle() {
    currentState = STATE_IDLE;
    // Gently stop: set target to current position (decelerates to stop)
    stepper.moveTo(stepper.currentPosition());
    // After deceleration completes, disable motor to save power
    // We check in the main loop via stepper.isRunning()
    // For safety, disable immediately (motor will hold briefly via driver)
    disableMotor();
    stepper.setCurrentPosition(0);
}

// ═══════════════════════════════════════════════════════════════════════
// MOTOR ROCKING LOGIC (Non-blocking oscillation)
// ═══════════════════════════════════════════════════════════════════════

void runRockingMotion() {
    // When the stepper reaches its target, reverse direction
    if (stepper.distanceToGo() == 0) {
        if (rockingForward) {
            stepper.moveTo(-ROCK_AMPLITUDE);
            rockingForward = false;
        } else {
            stepper.moveTo(ROCK_AMPLITUDE);
            rockingForward = true;
        }
    }
}

void enableMotor() {
    digitalWrite(MOTOR_ENA_PIN, LOW);  // TB6600: LOW = enabled
}

void disableMotor() {
    digitalWrite(MOTOR_ENA_PIN, HIGH); // TB6600: HIGH = disabled
}

// ═══════════════════════════════════════════════════════════════════════
// TEMPERATURE SENSOR (DS18B20 — Non-blocking)
// ═══════════════════════════════════════════════════════════════════════

void readTemperature(unsigned long now) {
    // DS18B20 needs ~750 ms for 12-bit conversion.
    // We request a new reading every TEMP_INTERVAL ms.
    if (now - lastTempRequest >= TEMP_INTERVAL) {
        // Read the result from the previous request
        float reading = tempSensor.getTempCByIndex(0);
        if (reading != DEVICE_DISCONNECTED_C && reading > -50.0 && reading < 85.0) {
            currentTemp = reading;
        }
        // Request next conversion (non-blocking)
        tempSensor.requestTemperatures();
        lastTempRequest = now;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PULSE SENSOR (Analog — Non-blocking peak detection)
// ═══════════════════════════════════════════════════════════════════════
//
// Simple threshold-based beat detection:
//   - When the analog signal rises above PULSE_THRESHOLD → beat start
//   - When it falls below → beat end (ready for next beat)
//   - IBI (Inter-Beat Interval) is averaged over 5 beats for stability
//

void readPulseSensor(unsigned long now) {
    int sensorValue = analogRead(PULSE_PIN);

    if (sensorValue > PULSE_THRESHOLD && !pulseHigh) {
        // Rising edge — beat detected
        pulseHigh = true;

        unsigned long ibi = now - lastBeatTime;
        lastBeatTime = now;

        // Validate the inter-beat interval
        if (ibi >= PULSE_MIN_IBI && ibi <= PULSE_MAX_IBI) {
            // Store in rolling buffer
            beatIntervals[beatIndex] = ibi;
            beatIndex = (beatIndex + 1) % 5;
            if (beatCount < 5) beatCount++;

            // Compute average IBI → BPM
            unsigned long sum = 0;
            for (int i = 0; i < beatCount; i++) {
                sum += beatIntervals[i];
            }
            unsigned long avgIBI = sum / beatCount;
            currentBPM = (int)(60000UL / avgIBI);

            // Sanity clamp
            if (currentBPM < 30)  currentBPM = 0;  // Likely noise
            if (currentBPM > 200) currentBPM = 0;  // Likely noise
        }

    } else if (sensorValue < PULSE_THRESHOLD - 50) {
        // Falling edge — reset for next beat (with hysteresis of 50)
        pulseHigh = false;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SENSOR REPORTING (CSV to Raspberry Pi)
// ═══════════════════════════════════════════════════════════════════════

void reportSensors() {
    // Format: "BPM,Temp\n"
    // Example: "72,36.50\n"
    Serial.print(currentBPM);
    Serial.print(",");
    Serial.println(currentTemp, 2);  // 2 decimal places
}
