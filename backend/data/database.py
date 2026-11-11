# data/database.py
"""
SQLite persistence for RescueWay.
Stores emergency events, simulation history, and benchmark results.
"""
import json
import os
import sqlite3
from datetime import datetime

from config import DB_PATH

# Bump this whenever a table's columns change. All four tables are still
# created via `CREATE TABLE IF NOT EXISTS`, which is a silent no-op against
# an existing database file — a column added to the schema in code would
# leave every pre-existing row without it, and the first thing to notice
# would be a confusing KeyError/sqlite3.OperationalError deep inside a
# background asyncio task rather than a clear failure at startup. This
# version number, stored in its own tiny table, lets startup fail loudly
# and immediately instead, with a clear message pointing at the mismatch.
SCHEMA_VERSION = 1


class SchemaVersionMismatch(RuntimeError):
    """Raised at startup when the on-disk database's schema_version doesn't
    match SCHEMA_VERSION — the on-disk file predates a schema change and
    needs an explicit migration (or, for this project's actual dev-scale
    usage, deleting/recreating the sqlite file) before it's safe to use."""


class RescueWayDB:
    def __init__(self):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self.db_path = DB_PATH
        self._init_tables()
        self._check_schema_version()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _check_schema_version(self):
        conn = self._get_conn()
        row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        if row is None:
            # Fresh database (or one created before this check existed) —
            # stamp it with the current version rather than failing, since
            # there's no prior version to conflict with.
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
            conn.commit()
        elif row["version"] != SCHEMA_VERSION:
            conn.close()
            raise SchemaVersionMismatch(
                f"Database at {self.db_path} has schema_version={row['version']}, "
                f"but this code expects {SCHEMA_VERSION}. A table's columns changed "
                f"since this file was created; CREATE TABLE IF NOT EXISTS silently "
                f"leaves old rows without new columns rather than erroring, so this "
                f"check exists to fail loudly instead. For this project's dev-scale "
                f"usage, deleting the .db file (data is regenerable, not authoritative) "
                f"is normally the right fix; a real migration script would be needed "
                f"if the existing data must be preserved."
            )
        conn.close()

    def _init_tables(self):
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS emergency_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                zone INTEGER NOT NULL,
                traffic_pattern TEXT,
                distance_m REAL,
                eta_seconds REAL,
                green_duration INTEGER,
                rl_action INTEGER,
                rl_action_name TEXT,
                reward REAL,
                response_time_s REAL,
                queue_lengths TEXT,
                resolved INTEGER DEFAULT 0,
                ambulance_id TEXT,
                status TEXT DEFAULT 'active',
                preempted_by TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS simulation_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                step INTEGER,
                mode TEXT,
                traffic_pattern TEXT,
                active_signal INTEGER,
                queue_lengths TEXT,
                avg_wait_time REAL,
                avg_congestion REAL,
                vehicles_cleared INTEGER,
                fairness_index REAL
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS benchmark_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                episodes INTEGER,
                traffic_pattern TEXT,
                emergency_probability REAL,
                results TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS rl_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                step INTEGER,
                mode TEXT,
                state INTEGER,
                action INTEGER,
                action_name TEXT,
                reward REAL,
                queue_lengths TEXT,
                emergency_active INTEGER,
                ambulance_zone INTEGER
            )
        """
        )

        conn.commit()
        conn.close()
        print(f"Database initialized: {self.db_path}")

    # ─── Emergency Events ────────────────────────────

    def save_emergency(
        self,
        zone,
        traffic,
        distance,
        eta,
        green_duration,
        action,
        action_name,
        reward,
        queue_lengths,
        ambulance_id=None,
        status="active",
    ):
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO emergency_events
            (timestamp, zone, traffic_pattern, distance_m, eta_seconds,
             green_duration, rl_action, rl_action_name, reward, queue_lengths,
             ambulance_id, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                datetime.now().isoformat(),
                zone,
                traffic,
                distance,
                eta,
                green_duration,
                action,
                action_name,
                reward,
                json.dumps(queue_lengths),
                ambulance_id,
                status,
            ),
        )
        conn.commit()
        conn.close()

    def mark_preempted(self, ambulance_id, preempted_by):
        """Mark an ambulance as preempted by another."""
        conn = self._get_conn()
        conn.execute(
            """
            UPDATE emergency_events
            SET status = 'preempted', preempted_by = ?
            WHERE ambulance_id = ? AND resolved = 0
        """,
            (preempted_by, ambulance_id),
        )
        conn.commit()
        conn.close()

    def resolve_emergency(self, zone, response_time):
        conn = self._get_conn()
        conn.execute(
            """
            UPDATE emergency_events
            SET resolved = 1, response_time_s = ?
            WHERE id = (
                SELECT id FROM emergency_events
                WHERE zone = ? AND resolved = 0
                ORDER BY id ASC LIMIT 1
            )
        """,
            (response_time, zone),
        )
        conn.commit()
        conn.close()

    def get_emergency_history(self, limit=50):
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT * FROM emergency_events
            ORDER BY id DESC LIMIT ?
        """,
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ─── Simulation Snapshots ────────────────────────

    def save_snapshot(
        self, step, mode, pattern, signal, queue_lengths, wait_time, congestion, cleared, fairness
    ):
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO simulation_snapshots
            (timestamp, step, mode, traffic_pattern, active_signal,
             queue_lengths, avg_wait_time, avg_congestion,
             vehicles_cleared, fairness_index)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                datetime.now().isoformat(),
                step,
                mode,
                pattern,
                signal,
                json.dumps(queue_lengths),
                wait_time,
                congestion,
                cleared,
                fairness,
            ),
        )
        conn.commit()
        conn.close()

    # ─── Benchmark Results ───────────────────────────

    def save_benchmark(self, episodes, pattern, emergency_prob, results):
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO benchmark_results
            (timestamp, episodes, traffic_pattern,
             emergency_probability, results)
            VALUES (?, ?, ?, ?, ?)
        """,
            (datetime.now().isoformat(), episodes, pattern, emergency_prob, json.dumps(results)),
        )
        conn.commit()
        conn.close()

    def get_benchmark_history(self, limit=10):
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT * FROM benchmark_results
            ORDER BY id DESC LIMIT ?
        """,
            (limit,),
        ).fetchall()
        conn.close()
        results = []
        for r in rows:
            d = dict(r)
            d["results"] = json.loads(d["results"])
            results.append(d)
        return results

    # ─── RL Decisions ────────────────────────────────

    def save_decision(
        self,
        step,
        mode,
        state,
        action,
        action_name,
        reward,
        queue_lengths,
        emergency_active,
        ambulance_zone,
    ):
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO rl_decisions
            (timestamp, step, mode, state, action, action_name,
             reward, queue_lengths, emergency_active, ambulance_zone)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                datetime.now().isoformat(),
                step,
                mode,
                state,
                action,
                action_name,
                reward,
                json.dumps(queue_lengths),
                1 if emergency_active else 0,
                ambulance_zone,
            ),
        )
        conn.commit()
        conn.close()

    def get_decision_history(self, limit=100, mode=None):
        conn = self._get_conn()
        if mode:
            rows = conn.execute(
                """
                SELECT * FROM rl_decisions
                WHERE mode = ?
                ORDER BY id DESC LIMIT ?
            """,
                (mode, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM rl_decisions
                ORDER BY id DESC LIMIT ?
            """,
                (limit,),
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ─── Statistics ──────────────────────────────────

    def get_stats(self):
        conn = self._get_conn()
        stats = {
            "total_emergencies": conn.execute("SELECT COUNT(*) FROM emergency_events").fetchone()[
                0
            ],
            "resolved_emergencies": conn.execute(
                "SELECT COUNT(*) FROM emergency_events WHERE resolved = 1"
            ).fetchone()[0],
            "avg_response_time": conn.execute(
                "SELECT AVG(response_time_s) FROM emergency_events WHERE resolved = 1"
            ).fetchone()[0]
            or 0,
            "total_snapshots": conn.execute("SELECT COUNT(*) FROM simulation_snapshots").fetchone()[
                0
            ],
            "total_benchmarks": conn.execute("SELECT COUNT(*) FROM benchmark_results").fetchone()[
                0
            ],
            "total_decisions": conn.execute("SELECT COUNT(*) FROM rl_decisions").fetchone()[0],
            "db_path": self.db_path,
            "db_size_kb": (
                round(os.path.getsize(self.db_path) / 1024, 1)
                if os.path.exists(self.db_path)
                else 0
            ),
        }
        conn.close()
        return stats


# Global instance
db = RescueWayDB()
