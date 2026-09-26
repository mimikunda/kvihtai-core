import json

from app.api.schemas import SetSummary
from app.db.database import get_connection


def _serialize_set_summary(set_summary: SetSummary) -> str:
    if hasattr(set_summary, "model_dump_json"):
        return set_summary.model_dump_json()
    return set_summary.json()


def _deserialize_set_summary(payload: str) -> SetSummary:
    if hasattr(SetSummary, "model_validate_json"):
        return SetSummary.model_validate_json(payload)
    return SetSummary.parse_raw(payload)


def save_set(set_summary: SetSummary) -> None:
    payload = _serialize_set_summary(set_summary)
    timestamp = set_summary.timestamp.isoformat()

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO sets (set_id, timestamp, payload)
            VALUES (?, ?, ?)
            ON CONFLICT(set_id) DO UPDATE SET
                timestamp = excluded.timestamp,
                payload = excluded.payload
            """,
            (set_summary.set_id, timestamp, payload),
        )


def get_latest_set() -> SetSummary | None:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT payload FROM sets ORDER BY timestamp DESC, set_id DESC LIMIT 1"
        ).fetchone()

    if row is None:
        return None

    return _deserialize_set_summary(row["payload"])


def save_result(set_id: str, started_at: str, result: dict) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO results (set_id, started_at, payload)
            VALUES (?, ?, ?)
            ON CONFLICT(set_id) DO UPDATE SET
                started_at = excluded.started_at,
                payload = excluded.payload
            """,
            (set_id, started_at, json.dumps(result)),
        )


def get_result(set_id: str) -> dict | None:
    with get_connection() as connection:
        row = connection.execute("SELECT payload FROM results WHERE set_id = ?", (set_id,)).fetchone()
    return None if row is None else json.loads(row["payload"])


def list_results(limit: int = 100) -> list[dict]:
    """The newest sets first, whole."""
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT payload FROM results ORDER BY started_at DESC, set_id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [json.loads(row["payload"]) for row in rows]


def delete_result(set_id: str) -> bool:
    with get_connection() as connection:
        gone = connection.execute("DELETE FROM results WHERE set_id = ?", (set_id,)).rowcount
        connection.execute("DELETE FROM sets WHERE set_id = ?", (set_id,))
    return gone > 0


def save_set_summary(data: dict) -> None:
    """Store a set given as a plain dict in the shape of SetSummary."""
    if hasattr(SetSummary, "model_validate"):
        save_set(SetSummary.model_validate(data))
    else:
        save_set(SetSummary.parse_obj(data))
