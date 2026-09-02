"""
Minimal SQLite persistence layer.
Stores every incoming ticket + the bot's decision (answered vs escalated) for
later analysis, dashboarding, and eval.
"""
import sqlite3
import json
from datetime import datetime, timezone
from contextlib import contextmanager

from app.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS ticket_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    customer_message TEXT NOT NULL,
    decision TEXT NOT NULL,              -- 'answered' | 'escalated'
    escalation_reasons TEXT,             -- JSON list of reasons, if escalated
    answer TEXT,                         -- generated answer, if answered
    answer_confidence REAL,
    retrieval_top_score REAL,
    sentiment_score REAL,
    matched_keyword TEXT,
    sources TEXT                         -- JSON list of source doc/ticket ids used
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(settings.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute(SCHEMA)


def log_ticket(
    customer_message: str,
    decision: str,
    escalation_reasons: list[str] | None,
    answer: str | None,
    answer_confidence: float | None,
    retrieval_top_score: float | None,
    sentiment_score: float | None,
    matched_keyword: str | None,
    sources: list[str] | None,
) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO ticket_log (
                created_at, customer_message, decision, escalation_reasons,
                answer, answer_confidence, retrieval_top_score, sentiment_score,
                matched_keyword, sources
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                customer_message,
                decision,
                json.dumps(escalation_reasons or []),
                answer,
                answer_confidence,
                retrieval_top_score,
                sentiment_score,
                matched_keyword,
                json.dumps(sources or []),
            ),
        )
        return cur.lastrowid


def get_recent_logs(limit: int = 50) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM ticket_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]


def get_stats() -> dict:
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) c FROM ticket_log").fetchone()["c"]
        answered = conn.execute(
            "SELECT COUNT(*) c FROM ticket_log WHERE decision = 'answered'"
        ).fetchone()["c"]
        escalated = total - answered
        deflection_rate = (answered / total) if total else 0.0
        return {
            "total": total,
            "answered": answered,
            "escalated": escalated,
            "deflection_rate": round(deflection_rate, 4),
        }
