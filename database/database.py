"""
SQLite database for helmet-violation records.

Schema
------
violations
  id                INTEGER PK AUTOINCREMENT
  timestamp         TEXT    ISO 8601
  track_id          INTEGER nullable
  plate_number      TEXT    e.g. KA01AB1234 or UNKNOWN
  plate_confidence  REAL    0-1
  helmet_confidence REAL    0-1
  helmet_status     TEXT    NO_HELMET
  evidence_image_path  TEXT (legacy column - kept for backwards compat)
  plate_image_path  TEXT    relative path under project root
  violation_image_path TEXT relative path under project root
  video_source      TEXT    filename of the source video
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS violations (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp            TEXT    NOT NULL,
    track_id             INTEGER,
    plate_number         TEXT,
    plate_confidence     REAL,
    helmet_confidence    REAL,
    helmet_status        TEXT    NOT NULL,
    evidence_image_path  TEXT,
    plate_image_path     TEXT,
    violation_image_path TEXT,
    video_source         TEXT,
    event_key            TEXT
)
"""

SAMPLES_SCHEMA = """
CREATE TABLE IF NOT EXISTS violation_image_samples (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    violation_id   INTEGER NOT NULL,
    sample_index   INTEGER NOT NULL,
    video_seconds  REAL NOT NULL,
    image_path     TEXT NOT NULL,
    UNIQUE(violation_id, sample_index)
)
"""

# Columns added in later versions - applied automatically as ALTER TABLE
_MIGRATIONS = [
    ("track_id",             "ALTER TABLE violations ADD COLUMN track_id INTEGER"),
    ("plate_confidence",     "ALTER TABLE violations ADD COLUMN plate_confidence REAL"),
    ("helmet_confidence",    "ALTER TABLE violations ADD COLUMN helmet_confidence REAL"),
    ("violation_image_path", "ALTER TABLE violations ADD COLUMN violation_image_path TEXT"),
    ("event_key",            "ALTER TABLE violations ADD COLUMN event_key TEXT"),
]


class ViolationDatabase:
    def __init__(self, path: str = "violations.db"):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(SCHEMA)
            conn.execute(SAMPLES_SCHEMA)
            self._migrate(conn)
            self._deduplicate(conn)
            conn.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS uq_violations_event
                   ON violations(video_source, event_key)
                   WHERE event_key IS NOT NULL"""
            )
            conn.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS uq_violations_plate
                   ON violations(upper(trim(plate_number)))
                   WHERE plate_number IS NOT NULL
                     AND upper(trim(plate_number)) <> 'UNKNOWN'
                     AND length(trim(plate_number)) >= 6
                     AND trim(plate_number) <> ''"""
            )

    # ------------------------------------------------------------------
    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _migrate(self, conn: sqlite3.Connection):
        """Add any missing columns so old databases remain compatible."""
        existing = {row[1] for row in conn.execute("PRAGMA table_info(violations)")}
        for col, ddl in _MIGRATIONS:
            if col not in existing:
                conn.execute(ddl)

    @staticmethod
    def _deduplicate(conn: sqlite3.Connection):
        """Keep the strongest existing row before enforcing one event/plate per DB."""
        conn.execute(
            """UPDATE violations SET plate_number='UNKNOWN'
               WHERE plate_number IS NOT NULL
                 AND upper(trim(plate_number)) <> 'UNKNOWN'
                 AND length(trim(plate_number)) < 6"""
        )
        conn.execute(
            """UPDATE violations
               SET event_key = CASE
                   WHEN track_id IS NOT NULL THEN 'id:' || track_id
                   ELSE 'legacy:' || id
               END
               WHERE event_key IS NULL OR trim(event_key) = ''"""
        )
        # Prefer a readable plate, then the strongest plate detection, then newest row.
        conn.execute(
            """DELETE FROM violations
               WHERE id NOT IN (
                   SELECT id FROM (
                       SELECT id,
                              ROW_NUMBER() OVER (
                                  PARTITION BY video_source, event_key
                                  ORDER BY CASE WHEN upper(trim(coalesce(plate_number,'')))
                                                     NOT IN ('', 'UNKNOWN')
                                           AND length(trim(coalesce(plate_number,''))) >= 6
                                           THEN 1 ELSE 0 END DESC,
                                           coalesce(plate_confidence, 0) DESC,
                                           id DESC
                              ) AS rn
                       FROM violations
                   ) WHERE rn = 1
               )"""
        )
        conn.execute(
            """DELETE FROM violations
               WHERE id NOT IN (
                   SELECT id FROM (
                       SELECT id,
                              ROW_NUMBER() OVER (
                                  PARTITION BY upper(trim(plate_number))
                                  ORDER BY coalesce(plate_confidence, 0) DESC, id DESC
                              ) AS rn
                       FROM violations
                       WHERE plate_number IS NOT NULL
                         AND upper(trim(plate_number)) NOT IN ('', 'UNKNOWN')
                         AND length(trim(plate_number)) >= 6
                   ) WHERE rn = 1
               )
               AND plate_number IS NOT NULL
               AND upper(trim(plate_number)) NOT IN ('', 'UNKNOWN')
               AND length(trim(plate_number)) >= 6"""
        )

    # ------------------------------------------------------------------
    def add_violation(
        self,
        timestamp: str,
        track_id: int | None,
        plate_number: str,
        plate_confidence: float,
        helmet_confidence: float,
        helmet_status: str,
        plate_image_path: str | None,
        violation_image_path: str,
        video_source: str,
        event_key: str | None = None,
    ) -> int | None:
        with self._connect() as conn:
            if event_key is None:
                event_key = f"id:{track_id}" if track_id is not None else None
            if plate_number and plate_number.strip().upper() != "UNKNOWN" and len(plate_number.strip()) < 6:
                plate_number = "UNKNOWN"

            # For unreadable plates, deduplicate reruns by source and tracker event.
            if event_key is not None:
                existing_event = conn.execute(
                    "SELECT id, plate_number FROM violations WHERE video_source = ? AND event_key = ? LIMIT 1",
                    (video_source, event_key),
                ).fetchone()
                if existing_event:
                    old_plate = (existing_event["plate_number"] or "UNKNOWN").strip().upper()
                    new_plate = (plate_number or "UNKNOWN").strip().upper()
                    if old_plate == "UNKNOWN" and new_plate != "UNKNOWN":
                        existing_plate = conn.execute(
                            "SELECT id FROM violations WHERE upper(trim(plate_number)) = upper(trim(?)) LIMIT 1",
                            (plate_number,),
                        ).fetchone()
                        if existing_plate:
                            conn.execute(
                                "DELETE FROM violation_image_samples WHERE violation_id = ?",
                                (existing_event["id"],),
                            )
                            conn.execute("DELETE FROM violations WHERE id=?", (existing_event["id"],))
                            return None
                        conn.execute(
                            """UPDATE violations SET timestamp=?, track_id=?, plate_number=?,
                               plate_confidence=?, helmet_confidence=?, helmet_status=?,
                               evidence_image_path=?, plate_image_path=?, violation_image_path=?
                               WHERE id=?""",
                            (timestamp, track_id, plate_number, plate_confidence,
                             helmet_confidence, helmet_status, violation_image_path,
                             plate_image_path, violation_image_path, existing_event["id"]),
                        )
                        return int(existing_event["id"])
                    return None

            # A recognized plate is the durable vehicle identity across runs.
            if plate_number and plate_number.strip().upper() != "UNKNOWN":
                existing_plate = conn.execute(
                    "SELECT id FROM violations WHERE upper(trim(plate_number)) = upper(trim(?)) LIMIT 1",
                    (plate_number,),
                ).fetchone()
                if existing_plate:
                    return None

            cur = conn.execute(
                """INSERT INTO violations
                   (timestamp, track_id, plate_number, plate_confidence,
                    helmet_confidence, helmet_status,
                    evidence_image_path,
                    plate_image_path,
                    violation_image_path, video_source, event_key)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    timestamp,
                    track_id,
                    plate_number,
                    plate_confidence,
                    helmet_confidence,
                    helmet_status,
                    violation_image_path,   # evidence_image_path (legacy NOT NULL compat)
                    plate_image_path,
                    violation_image_path,
                    video_source,
                    event_key,
                ),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def all(self) -> list[dict]:
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(
                "SELECT * FROM violations ORDER BY id DESC"
            )]

    def replace_image_samples(self, violation_id: int, samples: list[dict]) -> None:
        """Replace the timestamped rider crops belonging to one violation."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM violation_image_samples WHERE violation_id = ?",
                (violation_id,),
            )
            conn.executemany(
                """INSERT INTO violation_image_samples
                   (violation_id, sample_index, video_seconds, image_path)
                   VALUES (?, ?, ?, ?)""",
                [
                    (violation_id, sample["sample_index"], sample["video_seconds"], sample["image_path"])
                    for sample in samples
                ],
            )

    def image_samples(self, violation_id: int) -> list[dict]:
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(
                """SELECT sample_index, video_seconds, image_path
                   FROM violation_image_samples
                   WHERE violation_id = ? ORDER BY sample_index""",
                (violation_id,),
            )]

    def event_id(self, video_source: str, event_key: str) -> int | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM violations WHERE video_source = ? AND event_key = ? LIMIT 1",
                (video_source, event_key),
            ).fetchone()
            return int(row["id"]) if row else None

    def clear(self):
        with self._connect() as conn:
            conn.execute("DELETE FROM violation_image_samples")
            conn.execute("DELETE FROM violations")
            conn.execute("DELETE FROM sqlite_sequence WHERE name='violations'")
