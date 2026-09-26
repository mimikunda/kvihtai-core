import sqlite3

from app.config import settings

DB_PATH = settings.data_dir / "kvihtai.db"


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    return connection


def initialize_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
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


initialize_db()
