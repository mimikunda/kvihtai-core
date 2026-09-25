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
