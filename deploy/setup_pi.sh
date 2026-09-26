#!/usr/bin/env bash
# Set a Raspberry Pi up as a KvihtAI station. Run it on the Pi, as the user the
# station should run as, after deploy/push.sh has copied the code over:
#
#   bash ~/kvihtai-core/deploy/setup_pi.sh
#
# It needs sudo. It is safe to run again: what exists is replaced.
#
#   the station      a systemd service that starts at boot: camera, sets, API
#                    and the web app on port 8080
#   the hotspot      Wi-Fi "KvihtAI", opened at boot when no known Wi-Fi is in
#                    range; the station is then at http://10.42.0.1:8080
#
# Environment:
#   VENV                       virtualenv to run in (default ~/kvihtai-core/.venv)
#   KVIHTAI_HOTSPOT_SSID       default KvihtAI
#   KVIHTAI_HOTSPOT_PASSWORD   default: the one already set, or a new random one

set -euo pipefail

VENV="${VENV:-$HOME/kvihtai-core/.venv}"
SSID="${KVIHTAI_HOTSPOT_SSID:-KvihtAI}"
HOTSPOT=kvihtai-hotspot
HERE="$(cd "$(dirname "$0")" && pwd)"

if ! "$VENV/bin/python" -c "import fastapi, uvicorn, pydantic_settings, cv2, picamera2" 2>/dev/null; then
    echo "The virtualenv $VENV cannot import what the station needs." >&2
    echo "See docs/RASPBERRY_PI.md, Install; then run this again, with VENV=... if it lives elsewhere." >&2
    exit 1
fi

render() {
    sed -e "s|@USER@|$USER|g" -e "s|@HOME@|$HOME|g" -e "s|@VENV@|$VENV|g" "$1"
}

echo "== station service"
render "$HERE/kvihtai.service" | sudo tee /etc/systemd/system/kvihtai.service >/dev/null
mkdir -p "$HOME/kvihtai-data"

echo "== hotspot"
PASSWORD="${KVIHTAI_HOTSPOT_PASSWORD:-}"
if [ -z "$PASSWORD" ] && nmcli -t connection show "$HOTSPOT" >/dev/null 2>&1; then
    PASSWORD=$(sudo nmcli -s -g 802-11-wireless-security.psk connection show "$HOTSPOT")
fi
if [ -z "$PASSWORD" ]; then
    # no 0/o or 1/l, so that it can be read off a screen and typed on a phone
    PASSWORD=$(python3 -c "import secrets; print(''.join(secrets.choice('abcdefghjkmnpqrstuvwxyz23456789') for _ in range(12)))")
fi
sudo nmcli connection delete "$HOTSPOT" >/dev/null 2>&1 || true
sudo nmcli connection add type wifi ifname wlan0 con-name "$HOTSPOT" autoconnect no ssid "$SSID" \
    802-11-wireless.mode ap 802-11-wireless.band bg 802-11-wireless.channel 6 \
    ipv4.method shared ipv4.addresses 10.42.0.1/24 ipv6.method disabled \
    wifi-sec.key-mgmt wpa-psk wifi-sec.proto rsn wifi-sec.pairwise ccmp wifi-sec.group ccmp \
    wifi-sec.psk "$PASSWORD" >/dev/null
# a name for the station on its own network
sudo mkdir -p /etc/NetworkManager/dnsmasq-shared.d
echo "address=/kvihtai.lan/10.42.0.1" | sudo tee /etc/NetworkManager/dnsmasq-shared.d/kvihtai.conf >/dev/null
render "$HERE/kvihtai-network.service" | sudo tee /etc/systemd/system/kvihtai-network.service >/dev/null

sudo systemctl daemon-reload
sudo systemctl enable kvihtai-network.service >/dev/null
sudo systemctl enable kvihtai.service >/dev/null
sudo systemctl restart kvihtai.service

cat <<DONE

The station runs now and starts at every boot.

  At home, on a known Wi-Fi:  http://$(hostname -I | awk '{print $1}'):8080
  Anywhere else:              join Wi-Fi "$SSID", password $PASSWORD,
                              then open http://10.42.0.1:8080

  Logs:     journalctl -u kvihtai -f
  Hotspot:  bash $HERE/network.sh hotspot   (and "wifi" to go back)
DONE
