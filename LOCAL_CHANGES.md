# Local changes applied 2026-08-12

Four small fixes on top of `rescueway_full_codebase_revised.zip`. No behaviour
of the controller, RL agents, shield, or evaluation harness was touched.

1. **frontend/src/components/unified/TopBar.jsx** — Arduino badge logic was
   inverted: any truthy `arduino_ack` (including simulated `SIM:` acks) showed
   the hardware label. Now shows `HW` (green) only for real `OK:` acks from the
   board, `SIM` (gray) otherwise.

2. **frontend/src/components/unified/TopBar.test.jsx** — assertions updated to
   the corrected behaviour, plus a new regression test covering the `SIM:` ack
   case that triggered the original bug. Full suite: 22/22 passing.

3. **frontend/src/App.jsx** (bottom status bar) — same inverted-ack bug fixed;
   label reads `Arduino (HW)` only on `OK:` acks, amber dot in simulation.

4. **backend/config.py** and **backend/.env.example** — default
   `RESCUEWAY_ARDUINO_PORT` changed `COM20` → `COM3` (the Uno's actual port on
   the demo machine), so the backend connects to hardware without setting an
   env var each session. Override still works via `RESCUEWAY_ARDUINO_PORT`.
