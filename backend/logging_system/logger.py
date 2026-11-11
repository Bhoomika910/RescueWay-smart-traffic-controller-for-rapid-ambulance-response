# logging_system/logger.py
import json
import logging
import os
import time
from collections import deque
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Create logs directory
from config import LOG_BACKUP_COUNT, LOG_DIR, LOG_MAX_BYTES, LOG_MEMORY_SIZE

os.makedirs(LOG_DIR, exist_ok=True)


class RescueWayLogger:
    """
    Centralized logging system for RescueWay.

    Logs to:
    1. Console (colored, human-readable)
    2. File (structured, rotated)
    3. In-memory buffer (for dashboard display)
    """

    def __init__(self, max_memory_logs: int = LOG_MEMORY_SIZE):
        self.memory_logs = deque(maxlen=max_memory_logs)
        self.start_time = time.time()

        # Counters
        self.counts = {
            "decisions": 0,
            "signal_changes": 0,
            "emergencies": 0,
            "errors": 0,
            "warnings": 0,
            "arduino_commands": 0,
        }

        # Set up Python logger
        self.logger = logging.getLogger("rescueway")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # clear existing

        # Console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console_fmt = logging.Formatter(
            "%(asctime)s │ %(levelname)-7s │ %(message)s", datefmt="%H:%M:%S"
        )
        console.setFormatter(console_fmt)
        self.logger.addHandler(console)

        # File handler (rotates at 5MB, keeps 3 backups)
        file_path = os.path.join(LOG_DIR, "rescueway.log")
        file_handler = RotatingFileHandler(
            file_path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT
        )
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
        file_handler.setFormatter(file_fmt)
        self.logger.addHandler(file_handler)

        self.logger.info("RescueWay logging system initialized")

    def _store(self, level, category, message, data=None):
        """Store log entry in memory buffer."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed": round(time.time() - self.start_time, 2),
            "level": level,
            "category": category,
            "message": message,
            "data": data or {},
        }
        self.memory_logs.append(entry)
        return entry

    # ─── Decision Logging ─────────────────────────────────

    def log_decision(
        self,
        action,
        action_name,
        reward,
        state,
        queue_lengths=None,
        mode="normal",
        ambulance_id=None,
    ):
        self.counts["decisions"] += 1
        tag = f" | {ambulance_id}" if ambulance_id else ""
        msg = f"[DECISION] {action_name} | reward={reward:+.1f} | mode={mode}{tag}"
        self.logger.info(msg)
        self._store(
            "INFO",
            "decision",
            msg,
            {
                "action": action,
                "action_name": action_name,
                "reward": reward,
                "state": state,
                "queue_lengths": queue_lengths,
                "mode": mode,
                "ambulance_id": ambulance_id,
            },
        )

    # ─── Signal Change Logging ────────────────────────────

    def log_signal_change(self, zone, color, previous_zone=None):
        self.counts["signal_changes"] += 1
        msg = f"[SIGNAL] Zone {zone} -> {color.upper()}"
        if previous_zone and previous_zone != zone:
            msg += f" (was Zone {previous_zone})"
        self.logger.info(msg)
        self._store(
            "INFO",
            "signal",
            msg,
            {
                "zone": zone,
                "color": color,
                "previous_zone": previous_zone,
            },
        )

    # ─── Emergency Logging ────────────────────────────────

    def log_emergency_start(self, zone, traffic, ambulance_id=None):
        self.counts["emergencies"] += 1
        tag = f" | {ambulance_id}" if ambulance_id else ""
        msg = f"[EMERGENCY] Ambulance detected in Zone {zone} | traffic={traffic}{tag}"
        self.logger.warning(msg)
        self._store(
            "WARNING",
            "emergency",
            msg,
            {
                "zone": zone,
                "traffic": traffic,
                "event": "start",
                "ambulance_id": ambulance_id,
            },
        )

    def log_emergency_end(self, zone, duration, ambulance_id=None):
        tag = f" | {ambulance_id}" if ambulance_id else ""
        msg = f"[EMERGENCY] Cleared Zone {zone} after {duration:.1f}s{tag}"
        self.logger.info(msg)
        self._store(
            "INFO",
            "emergency",
            msg,
            {
                "zone": zone,
                "duration": duration,
                "event": "end",
                "ambulance_id": ambulance_id,
            },
        )

    # ─── Arduino Logging ─────────────────────────────────

    def log_arduino_command(self, zone, color, ack):
        self.counts["arduino_commands"] += 1
        msg = f"[ARDUINO] ZONE:{zone},CMD:{color} -> {ack}"
        self.logger.debug(msg)
        self._store(
            "DEBUG",
            "arduino",
            msg,
            {
                "zone": zone,
                "color": color,
                "ack": ack,
            },
        )

    def log_arduino_error(self, error):
        self.counts["errors"] += 1
        msg = f"[ARDUINO ERROR] {error}"
        self.logger.error(msg)
        self._store(
            "ERROR",
            "arduino",
            msg,
            {
                "error": str(error),
            },
        )

    # ─── Traffic Logging ─────────────────────────────────

    def log_traffic_pattern_change(self, pattern):
        msg = f"[TRAFFIC] Pattern changed to {pattern}"
        self.logger.info(msg)
        self._store(
            "INFO",
            "traffic",
            msg,
            {
                "pattern": pattern,
            },
        )

    def log_traffic_stats(self, queue_lengths, system_metrics):
        total = sum(queue_lengths.values())
        msg = (
            f"[TRAFFIC] Queues: {queue_lengths} | "
            f"Total: {total} | "
            f"Cleared: {system_metrics.get('total_vehicles_cleared', 0)}"
        )
        self.logger.debug(msg)
        self._store(
            "DEBUG",
            "traffic",
            msg,
            {
                "queue_lengths": queue_lengths,
                "system_metrics": system_metrics,
            },
        )

    # ─── System Logging ──────────────────────────────────

    def log_system_event(self, event, details=None):
        msg = f"[SYSTEM] {event}"
        self.logger.info(msg)
        self._store("INFO", "system", msg, details or {})

    def log_warning(self, message, data=None):
        self.counts["warnings"] += 1
        msg = f"[WARNING] {message}"
        self.logger.warning(msg)
        self._store("WARNING", "system", msg, data or {})

    def log_error(self, message, data=None):
        self.counts["errors"] += 1
        msg = f"[ERROR] {message}"
        self.logger.error(msg)
        self._store("ERROR", "system", msg, data or {})

    # ─── Query Methods ───────────────────────────────────

    def get_recent_logs(self, count=50, level=None, category=None):
        """Get recent log entries with optional filters."""
        logs = list(self.memory_logs)

        if level:
            logs = [entry for entry in logs if entry["level"] == level.upper()]
        if category:
            logs = [entry for entry in logs if entry["category"] == category]

        return logs[-count:]

    def get_counts(self):
        return dict(self.counts)

    def get_summary(self):
        return {
            "uptime_seconds": round(time.time() - self.start_time, 1),
            "total_logs": len(self.memory_logs),
            "counts": self.get_counts(),
            "log_file": os.path.join(LOG_DIR, "rescueway.log"),
        }

    def export_logs(self, filepath=None):
        """Export all in-memory logs to JSON file."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(LOG_DIR, f"export_{timestamp}.json")

        with open(filepath, "w") as f:
            json.dump(
                {
                    "exported_at": datetime.now().isoformat(),
                    "summary": self.get_summary(),
                    "logs": list(self.memory_logs),
                },
                f,
                indent=2,
            )

        self.logger.info(f"Logs exported to {filepath}")
        return filepath


# Global logger instance
app_logger = RescueWayLogger()
