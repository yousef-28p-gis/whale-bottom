#!/bin/bash
# 🐋 صياد القاع — كرون مغلف
# يشغل الدايمن إذا واقف، ويوصل التقرير إذا في تغيير

DAEMON="/data/trading28/sayad_alqae_daemon.py"
REPORT="/data/trading28/sayad_alqae_report.txt"

# تشغيل الدايمن إذا مش شغال
if ! pgrep -f "sayad_alqae_daemon.py" > /dev/null; then
    cd /data/trading28 && nohup python3 -u "$DAEMON" > /data/trading28/sayad_alqae_log.txt 2>&1 &
    echo "🐋 صياد القاع — تم تشغيل الدايمن"
else
    # توصيل التقرير
    if [ -f "$REPORT" ]; then
        cat "$REPORT"
    fi
fi
