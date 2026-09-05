from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class OutputRecord:
    name: str
    path: str
    output_type: str
    producer: str
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SQLiteStateStore:
    """Persist run traces and output metadata."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.path))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS task_events "
                "(id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, event_json TEXT NOT NULL)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS state_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    step INTEGER NOT NULL,
                    node TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS step_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    step INTEGER NOT NULL,
                    after_node TEXT NOT NULL,
                    action TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    risk_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS human_review_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS outputs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    output_type TEXT NOT NULL,
                    producer TEXT NOT NULL,
                    description TEXT NOT NULL
                )
                """
            )

    def add_snapshot(self, step: int, node: str, snapshot: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO state_snapshots(step, node, snapshot_json) VALUES (?, ?, ?)",
                (step, node, json.dumps(snapshot, ensure_ascii=False)),
            )

    def add_task_event(self, task: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute("INSERT INTO task_events(task_id, event_json) VALUES (?, ?)",
                         (task["task_id"], json.dumps(task, ensure_ascii=False)))

    def add_step_decision(
        self,
        *,
        step: int,
        after_node: str,
        action: str,
        reason: str,
        risk: dict[str, Any],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO step_decisions(step, after_node, action, reason, risk_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (step, after_node, action, reason, json.dumps(risk, ensure_ascii=False)),
            )

    def add_human_review_item(self, item: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO human_review_queue(item_json) VALUES (?)",
                (json.dumps(item, ensure_ascii=False),),
            )

    def add_output(self, record: OutputRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO outputs(name, path, output_type, producer, description)
                VALUES (?, ?, ?, ?, ?)
                """,
                (record.name, record.path, record.output_type, record.producer, record.description),
            )
