import logging
import os
import sqlite3
from datetime import datetime

from app.config import settings

DB_PATH = settings.data_dir / "kvihtai.db"

# Whether the database was made new when the app started, and where a broken
# one was put. Every set is also on disk as sets/<id>/result.json, so a new
# database is filled again from there; see app.station.
CREATED = False
BROKEN = None


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    return connection


def _create_tables() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sets (
                set_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        # The whole analysis of a set, as app.analysis.report writes it; `sets`
        # holds the same set in the shape of the API contract.
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                set_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise sqlite3.DatabaseError("quick_check found damage")


def initialize_db() -> None:
    """Open the database, or start a new one if it cannot be read.

    The Pi is switched off at the wall, and an SD card can lose a write. A
    database that cannot be read would take the whole API down with it, so it
    is moved aside rather than repaired here.
    """
    global CREATED, BROKEN
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    CREATED = not DB_PATH.exists()
    try:
        _create_tables()
    except sqlite3.DatabaseError as e:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        aside = DB_PATH.with_name(f"{DB_PATH.name}.broken-{stamp}")
        os.replace(DB_PATH, aside)
        for suffix in ("-wal", "-shm"):
            try:
                os.remove(f"{DB_PATH}{suffix}")
            except OSError:
                pass
        logging.getLogger("kvihtai.db").warning(f"the database could not be read ({e}); moved to {aside.name}")
        CREATED = True
        BROKEN = aside.name
        _create_tables()


initialize_db()
