#!/usr/bin/env bash
# Join a known Wi-Fi if one is in range; otherwise open the hotspot, so that a
# phone can reach the station at the gym, where there is no network it knows.
#
# Run once at boot by kvihtai-network.service. Waiting for a known network
# first matters: NetworkManager would bring up an access point at once, since
# one is always "available", and a Pi at home would never join its Wi-Fi.
#
#   network.sh            decide as at boot
#   network.sh hotspot    open the hotspot now
#   network.sh wifi       leave the hotspot and join a known Wi-Fi

set -u
HOTSPOT=kvihtai-hotspot
WAIT_S=40

joined() {
    local conn
    conn=$(nmcli -t -f DEVICE,STATE,CONNECTION device | awk -F: '$1 == "wlan0" && $2 == "connected" {print $3}')
    [ -n "$conn" ] && [ "$conn" != "$HOTSPOT" ]
}

case "${1:-boot}" in
    hotspot)
        exec nmcli connection up "$HOTSPOT"
        ;;
    wifi)
        nmcli connection down "$HOTSPOT" 2>/dev/null
        exec nmcli device connect wlan0
        ;;
esac

for _ in $(seq "$WAIT_S"); do
    if joined; then
        echo "on a known Wi-Fi, no hotspot"
        exit 0
    fi
    sleep 1
done
echo "no known Wi-Fi in range: opening the hotspot"
nmcli connection up "$HOTSPOT"
