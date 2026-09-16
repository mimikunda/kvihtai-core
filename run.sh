#!/usr/bin/env bash
# Development launcher for the KvihtAI Core daemon.
#
# Environment overrides:
#   KVIHTAI_HOST    bind address   (default 0.0.0.0)
#   KVIHTAI_PORT    bind port      (default 8080)
#   KVIHTAI_RELOAD  auto-reload    (default 1, set 0 to disable)

set -euo pipefail

cd "$(dirname "$0")"

VENV=".venv"

if [ ! -x "$VENV/bin/uvicorn" ]; then
    echo "No virtualenv found in $VENV." >&2
    echo "Create it with:" >&2
    echo "  python3 -m venv $VENV && $VENV/bin/pip install -r requirements.txt" >&2
    exit 1
fi

HOST="${KVIHTAI_HOST:-0.0.0.0}"
PORT="${KVIHTAI_PORT:-8080}"
RELOAD="${KVIHTAI_RELOAD:-1}"

args=(app.main:app --host "$HOST" --port "$PORT")

case "$RELOAD" in
    1|true|yes|on) args+=(--reload) ;;
esac

exec "$VENV/bin/uvicorn" "${args[@]}"
