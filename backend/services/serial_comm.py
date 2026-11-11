# services/serial_comm.py
import threading
import time

import serial
import serial.tools.list_ports

from config import ARDUINO_BAUD, ARDUINO_PORT, HARDWARE_ENABLED


class SerialBridge:
    def __init__(self):
        self.port = ARDUINO_PORT
        self.baud = ARDUINO_BAUD
        self.conn = None
        self.connected = False
        self.simulation_mode = False
        self._lock = threading.Lock()

        # Reliability stats
        self.commands_sent = 0
        self.commands_success = 0
        self.commands_failed = 0
        self.reconnect_attempts = 0
        self.last_error = None
        self.last_ack = None
        self.last_command_time = None

    def connect(self):
        if not HARDWARE_ENABLED:
            print("Hardware disabled by configuration. Running in simulation mode.")
            self.connected = False
            self.simulation_mode = True
            return "SIMULATION_MODE"

        print(f"Connecting to Arduino on {self.port}...")
        try:
            self.conn = serial.Serial(self.port, self.baud, timeout=2)
            time.sleep(2)
            response = self.conn.readline().decode("utf-8", errors="ignore").strip()
            print(f"Arduino says: {response}")
            self.connected = True
            self.simulation_mode = False
            return response
        except Exception as e:
            print(f"Arduino connection failed: {e}")
            self.connected = False
            self.simulation_mode = True
            self.last_error = str(e)
            return "SIMULATION_MODE"

    def send_command(self, zone, signal, duration=30):
        with self._lock:
            self.commands_sent += 1
            self.last_command_time = time.time()

            if self.simulation_mode or not self.conn:
                self.commands_success += 1
                ack = f"SIM:zone={zone},signal={signal}"
                self.last_ack = ack
                return ack

            cmd = f"ZONE:{zone},CMD:{signal},DUR:{duration}\n"

            # Try sending with retry
            for attempt in range(3):
                try:
                    if not self.conn.is_open:
                        self._reconnect()

                    # Flush any stale data sitting in the buffer
                    self.conn.reset_input_buffer()

                    self.conn.write(cmd.encode("utf-8"))
                    self.conn.flush()
                    ack = self.conn.readline().decode("utf-8", errors="ignore").strip()

                    if ack:
                        self.commands_success += 1
                        self.last_ack = ack
                        return ack

                    # Empty ack — retry
                    if attempt < 2:
                        time.sleep(0.1)
                        continue

                except serial.SerialException as e:
                    self.last_error = str(e)
                    if attempt < 2:
                        self._reconnect()
                        continue

                except Exception as e:
                    self.last_error = str(e)
                    break

            # All retries failed
            self.commands_failed += 1
            self.last_error = f"Failed after 3 attempts: {cmd.strip()}"
            return f"ERR:failed,zone={zone}"

    def _reconnect(self):
        self.reconnect_attempts += 1
        try:
            if self.conn and self.conn.is_open:
                self.conn.close()
            time.sleep(0.5)
            self.conn = serial.Serial(self.port, self.baud, timeout=2)
            time.sleep(2)
            response = self.conn.readline().decode("utf-8", errors="ignore").strip()
            self.connected = True
            self.simulation_mode = False
            print(f"Arduino reconnected: {response}")
        except Exception as e:
            self.connected = False
            self.simulation_mode = True
            self.last_error = f"Reconnect failed: {e}"
            print(f"Arduino reconnect failed: {e}")

    def get_status(self):
        return {
            "connected": self.connected,
            "simulation_mode": self.simulation_mode,
            "port": self.port,
            "baud": self.baud,
            "commands_sent": self.commands_sent,
            "commands_success": self.commands_success,
            "commands_failed": self.commands_failed,
            "success_rate": round(
                (
                    (self.commands_success / self.commands_sent * 100)
                    if self.commands_sent > 0
                    else 100
                ),
                1,
            ),
            "reconnect_attempts": self.reconnect_attempts,
            "last_error": self.last_error,
            "last_ack": self.last_ack,
        }

    def close(self):
        if self.conn and self.conn.is_open:
            self.conn.close()
            print("Serial port closed.")
        self.connected = False
