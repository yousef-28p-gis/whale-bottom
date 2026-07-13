#!/usr/bin/env python3
"""
Simulate multiple rapid signals to test monitor + hunter pipeline.
"""

import json, os, sys
from datetime import datetime, timezone

# Step 1: Simulate monitor writing multiple batches
SIGNALS_FILE = '/data/trading28/live_signals.json'

# Batch 1: 3 signals at once
batch1 = [
    {"symbol": "TEST1", "dt": "2026-07-12T16:00:00+00:00", "volume_usdt": 300000, "price": 1.5, "msg_id": 999991},
    {"symbol": "TEST2", "dt": "2026-07-12T16:00:30+00:00", "volume_usdt": 400000, "price": 2.5, "msg_id": 999992},
    {"symbol": "TEST3", "dt": "2026-07-12T16:01:00+00:00", "volume_usdt": 500000, "price": 3.5, "msg_id": 999993},
]

# Simulate monitor.py overwrite behavior (the bug)
with open(SIGNALS_FILE, 'w') as f:
    json.dump(batch1, f, default=str)
print(f'Monitor run 1: wrote {len(batch1)} signals')

# Batch 2: 1 more signal (overwrites!)
batch2 = [
    {"symbol": "TEST4", "dt": "2026-07-12T16:02:00+00:00", "volume_usdt": 600000, "price": 4.5, "msg_id": 999994},
]
with open(SIGNALS_FILE, 'w') as f:
    json.dump(batch2, f, default=str)
print(f'Monitor run 2: wrote {len(batch2)} signals (OVERWROTE!)')

# Step 2: Run hunter_live.py
print('\n--- Running hunter_live.py ---')
os.system('cd /data/trading28 && python3 hunter_live.py 2>&1')

# Step 3: Check what's in pending
print('\n--- Checking pending ---')
if os.path.exists('/data/trading28/live_pending.json'):
    with open('/data/trading28/live_pending.json') as f:
        pending = json.load(f)
    print(f'Pending signals: {len(pending)}')
    for p in pending:
        print(f'  {p["symbol"]} (msg_id={p["msg_id"]})')
    
    # Check if TEST1-TEST3 are in pending
    pending_ids = {p['msg_id'] for p in pending}
    for i in [999991, 999992, 999993]:
        if i in pending_ids:
            print(f'  ✅ msg_id {i} in pending')
        else:
            print(f'  ❌ msg_id {i} MISSING!')
