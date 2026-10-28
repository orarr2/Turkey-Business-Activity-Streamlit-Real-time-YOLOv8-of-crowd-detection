# Turkey - Business Activity VM shortcuts.
# Sourced from /etc/profile.d/turkey-shortcuts.sh on every interactive login.
# To (re)install from this file:
#   sudo cp /opt/turkey-footfall/src/deploy/gcp-vm/tr-shortcuts.sh \
#           /etc/profile.d/turkey-shortcuts.sh
#   source /etc/profile.d/turkey-shortcuts.sh

_tr_banner() {
    cat <<'BANNER'
===========================================================
  Turkey Footfall VM - operational
===========================================================
  tr-log         stream collector logs live (Ctrl+C exits)
  tr-health      VM resources + service health + heartbeat
  tr-slots       current grid (which cam in each slot)
  tr-rounds      last 20 detection rounds (person/vehicles)
  tr-digest      fire the daily digest NOW (email + PDF)
  tr-services    systemctl status of collector + digest
  tr-restart     restart the collector service
  tr-ibb         test if IBB Istanbul CDN is back up
  tr-firestore   today's projected Firestore write quota
  tr-help        show this menu again
===========================================================
BANNER
}

tr-log() { sudo journalctl -u collector.service -f; }

tr-health() {
    echo "=== VM resources ==="
    free -h | awk 'NR==1 || NR==2'
    echo
    df -h / | awk 'NR==1 || NR==2'
    echo
    uptime
    echo
    echo "=== services ==="
    systemctl is-active collector.service digest.timer 2>&1 | paste <(echo -e "collector.service\ndigest.timer") -
    echo
    echo "=== last heartbeat age ==="
    local last=$(sudo journalctl -u collector.service -n 200 --no-pager 2>/dev/null \
        | grep -oE '[0-9]+-[0-9]+-[0-9]+T[0-9]+:[0-9]+:[0-9]+' | tail -1)
    if [ -n "$last" ]; then
        local age=$(( $(date +%s) - $(date -d "$last" +%s 2>/dev/null || echo 0) ))
        echo "  last round: $last  ($age seconds ago)"
    else
        echo "  no rounds found in last 200 log lines"
    fi
}

tr-slots() {
    echo "=== current grid (COLLECTOR_COUNTRY_LADDER) ==="
    grep -A1 "^COLLECTOR_COUNTRY_LADDER" /opt/turkey-footfall/src/app/cameras.py \
        | head -1
    echo
    echo "=== last slot assignments (from logs) ==="
    sudo journalctl -u collector.service -n 400 --no-pager 2>/dev/null \
        | grep -oE 'slot_[0-9]+ \([a-z_0-9]+\)' | sort -u
    echo
    echo "=== recent rotations ==="
    sudo journalctl -u collector.service -n 400 --no-pager 2>/dev/null \
        | grep -E "slot_[0-9]+: ->" | tail -8
}

tr-rounds() {
    sudo journalctl -u collector.service -n 400 --no-pager 2>/dev/null \
        | grep -E "slot_[0-9]+ \([a-z_0-9]+\): (person=|MISS)" \
        | tail -20
}

tr-digest() {
    echo "=== firing digest.service NOW ==="
    sudo systemctl start digest.service
    sleep 2
    sudo journalctl -u digest.service -n 30 --no-pager --since "2 minutes ago"
}

tr-services() {
    sudo systemctl status collector.service digest.timer digest.service --no-pager -l 2>&1 \
        | head -60
}

tr-restart() {
    echo "=== restarting collector.service ==="
    sudo systemctl restart collector.service
    sleep 3
    systemctl is-active collector.service && echo "collector: ACTIVE"
}

tr-ibb() {
    echo "=== direct IBB CDN probe from VM ==="
    curl -sI --max-time 10 "https://kamerayayin.ibb.istanbul/turistikcam/taksim.stream/playlist.m3u8" 2>&1 \
        | head -3 || echo "  (no response - CDN still down)"
    echo
    echo "=== via Cloudflare proxy ==="
    local URL=$(sudo grep IBB_PROXY_URL /etc/turkey-footfall/proxy.env 2>/dev/null | cut -d= -f2)
    local SEC=$(sudo grep IBB_PROXY_SECRET /etc/turkey-footfall/proxy.env 2>/dev/null | cut -d= -f2)
    if [ -n "$URL" ]; then
        curl -sI --max-time 15 \
            "$URL/https://kamerayayin.ibb.istanbul/turistikcam/taksim.stream/playlist.m3u8" \
            -H "X-Proxy-Secret: $SEC" 2>&1 | grep -E "^HTTP|error code" | head -2
    fi
}

tr-firestore() {
    sudo journalctl -u collector.service -n 300 --no-pager 2>/dev/null \
        | grep -E "(Firestore|writes/day|free tier)" | tail -5
}

tr-help() { _tr_banner; }

# Show the banner on every interactive login
if [ -n "$PS1" ] && [ -z "$TURKEY_BANNER_SHOWN" ]; then
    export TURKEY_BANNER_SHOWN=1
    _tr_banner
fi
