#!/bin/bash
# الحوت الخاطف - cron wrapper (1 min)
cd /data/trading28

# Check if daemon is running
if ! pgrep -f "khatef_daemon.py" > /dev/null; then
    nohup python3 -u khatef_daemon.py > khatef_log.txt 2>&1 &
    sleep 2
fi

# Only deliver if report changed since last time
REPORT_FILE="khatef_report.txt"
HASH_FILE="/tmp/khatef_last_hash"

if [ -f "$REPORT_FILE" ]; then
    NEW_HASH=$(md5sum "$REPORT_FILE" | cut -d' ' -f1)
    OLD_HASH=$(cat "$HASH_FILE" 2>/dev/null)
    
    if [ "$NEW_HASH" != "$OLD_HASH" ]; then
        echo "$NEW_HASH" > "$HASH_FILE"
        cat "$REPORT_FILE"
    fi
fi
