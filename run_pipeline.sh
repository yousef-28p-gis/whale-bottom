#!/bin/bash
# WhaleSniper Live Pipeline
# 1. Fetch new signals from @WhaleSniper
# 2. Check whale confirmation on pending signals
# Only outputs when there are NEW confirmations (hunter_live.py is silent otherwise)

cd /data/trading28

# Step 1: Monitor for new signals (always run, silent)
python3 monitor.py > /dev/null 2>&1

# Step 2: Check whale confirmation (only prints on new confirmations)
python3 hunter_live.py
