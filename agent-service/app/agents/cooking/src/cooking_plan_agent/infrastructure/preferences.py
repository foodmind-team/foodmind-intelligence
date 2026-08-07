"""P5-4: 长期偏好存储（SQLite，key=user_id）。

存储内容：饮食限制、过敏原、常做菜品、口味偏好。
仅记录用户显式提供/确认过的信息（隐私：不记录原始菜谱文本）。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path


class PreferenceStore:
    """User long-term preference storage backed by a single SQLite table.

    Design (P5-4):
      - key = user_id (stable caller-supplied identity);
      - payload is a JSON object of confirmed preferences only;
      - ``get`` returns {} for unknown users (zero-regression for
        requests without a user_id);
      - ``put`` is an upsert (INSERT OR REPLACE) so a later confirmed
        preference overwrites an earlier one.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id    TEXT PRIMARY KEY,
                payload    TEXT NOT NULL,          -- JSON: {"dietary_restrictions": [...], "allergens": [...]}
                updated_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def get(self, user_id: str) -> dict[str, object]:
        row = self._conn.execute("SELECT payload FROM user_preferences WHERE user_id = ?", (user_id,)).fetchone()
        if row is None:
            return {}
        try:
            value = json.loads(row[0])
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    def put(self, user_id: str, payload: dict[str, object]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO user_preferences (user_id, payload, updated_at) VALUES (?, ?, ?)",
            (
                user_id,
                json.dumps(payload, ensure_ascii=False),
                datetime.now().isoformat(),
            ),
        )
        self._conn.commit()
