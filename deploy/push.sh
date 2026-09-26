#!/usr/bin/env bash
# Copy the code and the built web app from this machine to a Pi, and restart
# the station there if it runs as a service.
#
#   deploy/push.sh david@192.168.1.174
#
# The Pi keeps its own virtualenv and data; only app/, tools/ and deploy/ of
# this repository and kvihtai-web's dist/ are copied. Build the web app first
# with `npm run build` in kvihtai-web.

set -euo pipefail

TARGET="${1:?usage: deploy/push.sh user@host}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
WEB="${KVIHTAI_WEB:-$HERE/../kvihtai-web}"

rsync -a --delete --exclude __pycache__ \
    "$HERE/app" "$HERE/tools" "$HERE/deploy" "$HERE/requirements.txt" \
    "$TARGET:kvihtai-core/"

if [ -d "$WEB/dist" ]; then
    ssh "$TARGET" mkdir -p kvihtai-web
    rsync -a --delete "$WEB/dist" "$TARGET:kvihtai-web/"
else
    echo "no $WEB/dist: the web app was not copied (run npm run build there)" >&2
fi

ssh "$TARGET" 'if systemctl is-enabled --quiet kvihtai 2>/dev/null; then sudo systemctl restart kvihtai && echo "kvihtai restarted"; fi'
