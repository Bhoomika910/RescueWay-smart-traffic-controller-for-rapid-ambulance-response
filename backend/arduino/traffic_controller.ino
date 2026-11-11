// RescueWay Traffic Controller Firmware
// Arduino Uno
// Zone 1: D2(Red) D3(Yellow) D4(Green)
// Zone 2: D5(Red) D6(Yellow) D7(Green)
// Zone 3: D8(Red) D9(Yellow) D10(Green)
// Zone 4: D11(Red) D12(Yellow) D13(Green)
//
// Safety-relevant behavior added after a design-review audit found the
// yellow -> all-red -> green clearance sequencing that this system's
// documentation calls "a hardware safety constraint" was in fact enforced
// ONLY by asyncio.sleep() calls on the Python side, with nothing here
// backing that guarantee up if the Python scheduler, a race condition, or
// a future bug ever sent commands faster than intended. Two independent,
// firmware-level backstops were added:
//
//   1. MIN_CROSS_ZONE_INTERVAL_MS — refuses to green a DIFFERENT zone
//      less than that long after the last zone change, regardless of
//      what the host asks for. This does not replace the Python-side
//      yellow/all-red timing (which still governs the actual driver
//      experience) — it is a hard floor beneath it that holds even if
//      the host misbehaves.
//   2. WATCHDOG_TIMEOUT_MS — if no valid command arrives for this long
//      (e.g. the controller process crashed or the serial link died),
//      the firmware forces all-red on its own rather than freezing the
//      intersection in its last state indefinitely, which is what
//      happened previously: nothing here ever re-asserted a safe state
//      without a fresh command telling it to.
//
// What this is NOT: a substitute for a real, independent conflict monitor
// (the NEMA TS2 "Malfunction Management Unit" equivalent) wired directly
// to the physical outputs on separate hardware. That remains out of scope
// for this breadboard/prototype controller — see backend/ARCHITECTURE.md.

const int PINS[4][3] = {
  {2,  3,  4 },
  {5,  6,  7 },
  {8,  9,  10},
  {11, 12, 13}
};

// Minimum time between greening two DIFFERENT zones. Intentionally a
// little under the Python side's real intended yellow(2s)+all-red(1s) =
// 3000ms clearance interval, so ordinary serial/scheduling jitter on a
// legitimate command sequence isn't rejected — this is a hard floor
// against a clearly-too-fast/buggy sequence, not a replacement for the
// full timed transition.
const unsigned long MIN_CROSS_ZONE_INTERVAL_MS = 2500;

// If no valid command is received for this long, force all-red rather
// than holding the last commanded state indefinitely. Generous relative
// to the ~5s normal decision cycle so it never fires during ordinary
// operation gaps — this only catches a genuinely dead controller/link.
const unsigned long WATCHDOG_TIMEOUT_MS = 10000;

int lastGreenZone = -1;               // -1 = none yet (all-red at boot)
unsigned long lastZoneChangeMs = 0;
unsigned long lastCommandMs = 0;
bool watchdogTripped = false;

void setup() {
  Serial.begin(9600);
  for (int z = 0; z < 4; z++)
    for (int c = 0; c < 3; c++)
      pinMode(PINS[z][c], OUTPUT);
  allRed();
  lastCommandMs = millis();
  Serial.println("READY");
}

void loop() {
  checkWatchdog();

  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    bool zoneFound = false;
    String zoneStr = parseField(cmd, "ZONE", zoneFound);
    bool signalFound = false;
    String signal = parseField(cmd, "CMD", signalFound);

    if (!zoneFound || !signalFound) {
      Serial.println("ERR:malformed_command");
      return;
    }

    int zone = zoneStr.toInt() - 1;

    if (zone < 0 || zone > 3) {
      Serial.println("ERR:invalid_zone");
      return;
    }

    if (signal != "green" && signal != "yellow" && signal != "red") {
      // Previously any unrecognized signal string silently fell through
      // to the red branch while still echoing the bogus string back in
      // the "OK:" ack — the host would believe a command it never sent
      // was applied. Now rejected explicitly instead.
      Serial.println("ERR:invalid_signal");
      return;
    }

    // Firmware-level minimum-dwell interlock (see file header). A green
    // request for a zone other than the currently-green one, arriving
    // too soon after the last change, is refused outright rather than
    // executed — the host will see ERR:too_soon in the ack and (per the
    // Python-side _ack_confirms check) will NOT treat this as a
    // confirmed state change.
    unsigned long now = millis();
    if (signal == "green" && lastGreenZone != -1 && zone != lastGreenZone
        && (now - lastZoneChangeMs) < MIN_CROSS_ZONE_INTERVAL_MS) {
      Serial.println("ERR:too_soon");
      return;
    }

    allRed();

    if (signal == "green")       setColor(zone, LOW,  LOW,  HIGH);
    else if (signal == "yellow") setColor(zone, LOW,  HIGH, LOW);
    else                         setColor(zone, HIGH, LOW,  LOW);

    if (signal == "green") {
      if (zone != lastGreenZone) lastZoneChangeMs = now;
      lastGreenZone = zone;
    }
    lastCommandMs = now;
    watchdogTripped = false;

    Serial.println("OK:zone=" + String(zone + 1) + ",signal=" + signal);
  }
}

void checkWatchdog() {
  if (watchdogTripped) return;
  if (millis() - lastCommandMs > WATCHDOG_TIMEOUT_MS) {
    allRed();
    lastGreenZone = -1;
    watchdogTripped = true;   // force all-red once, don't spam every loop() iteration
    Serial.println("WATCHDOG:forced_all_red");
  }
}

// Returns the field's value and sets `found` to whether the key was
// actually present. Previously a missing key made indexOf return -1,
// which fed a bogus-but-in-bounds start offset into substring() instead
// of being detected as "field absent" — this made a malformed command's
// behavior depend on incidental string contents rather than failing
// cleanly.
String parseField(String data, String key, bool &found) {
  int keyStart = data.indexOf(key + ":");
  if (keyStart == -1) {
    found = false;
    return "";
  }
  found = true;
  int start = keyStart + key.length() + 1;
  int end   = data.indexOf(",", start);
  if (end == -1) end = data.length();
  return data.substring(start, end);
}

void allRed() {
  for (int z = 0; z < 4; z++) setColor(z, HIGH, LOW, LOW);
}

void setColor(int z, int r, int y, int g) {
  digitalWrite(PINS[z][0], r);
  digitalWrite(PINS[z][1], y);
  digitalWrite(PINS[z][2], g);
}
