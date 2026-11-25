"""Tiny SQLite store that records every uploaded training image."""
import datetime as dt
import sqlite3
from contextlib import closing

from src import config


def _conn():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with closing(_conn()) as c, c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS uploads(
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 filename TEXT, label TEXT, path TEXT,
                 uploaded_at TEXT, used_in_retrain INTEGER DEFAULT 0)"""
        )


def add_upload(filename: str, label: str, path: str):
    with closing(_conn()) as c, c:
        c.execute(
            "INSERT INTO uploads(filename,label,path,uploaded_at) VALUES(?,?,?,?)",
            (filename, label, path, dt.datetime.utcnow().isoformat(timespec="seconds")),
        )


def pending_count() -> int:
    with closing(_conn()) as c:
        return c.execute("SELECT COUNT(*) FROM uploads WHERE used_in_retrain=0").fetchone()[0]


def total_count() -> int:
    with closing(_conn()) as c:
        return c.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]


def counts_by_label() -> dict:
    with closing(_conn()) as c:
        rows = c.execute("SELECT label, COUNT(*) n FROM uploads GROUP BY label").fetchall()
    return {r["label"]: r["n"] for r in rows}


def mark_all_used():
    with closing(_conn()) as c, c:
        c.execute("UPDATE uploads SET used_in_retrain=1 WHERE used_in_retrain=0")
