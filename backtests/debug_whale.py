#!/usr/bin/env python3
"""Debug: trace every filter in whale_bottom check_entry"""
import json, numpy as np, pandas as pd
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/data/trading28/backtests/pattern_data'
TP = 3.5; SL = 1.5; WHALE_MIN = 0.50; STR = 50
BLOCK_HOURS = {1, 3, 6, 12, 0, 4}; BLOCK_WEEKDAY = 3

with open(f'{DATA_DIR}/15m_30d.json') as f:
    all_data = json.load(f)

# EXACT compute_indicators
def compute_indicators(df):
    df = df.copy()
    LB = 30
    df['lo'] = df['low'].rolling(LB).min()
    df['lc'] = abs(df['low'] - df['low'].shift(1)) / df['low'] * 100
    df['sm'] = df['lc'].ewm(span=3, adjust=False).mean()
    df['hi'] = df['sm'].rolling(LB).max()
    df['raw'] = np.where(df['low'] <= df['lo'], (df['sm'] + df['hi'] * 2) / 3, 0)
    df['whale'] = df['raw'].ewm(span=3, adjust=False).mean().fillna(0)
    df['spike'] = (df['whale'] > df['whale'].shift(1)) & (df['whale'].shift(1) <= 0.03)
    df['wf'] = df['whale'].rolling(2).mean()
    df['ws'] = df['whale'].rolling(5).mean()
    df['wp'] = df['whale'].rolling(50).max()
    df['str'] = (df['whale'] / df['wp'].replace(0, np.nan) * 100).fillna(0)
    df['vma'] = df['volume'].rolling(20).mean()
    df['entry'] = (df['spike'] & (df['wf'] > df['ws']) & (df['str'] > STR) & (df['volume'] > df['vma'] * 1.0))
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

counts = {
    'total_candles': 0,
    'entry_true': 0,
    'whale_min': 0,
    'next_whale_pass': 0,
    'rsi_pass': 0,
    'weekday_pass': 0,
    'hour_pass': 0,
    'pump24_pass': 0,
    'green_pass': 0,
}

for coin, data in list(all_data.items())[:5]:  # Check first 5 coins
    df = pd.DataFrame({
        'open': data['open'], 'high': data['high'],
        'low': data['low'], 'close': data['close'],
        'volume': data['volume'],
    })
    df['ts'] = pd.to_datetime([datetime.fromtimestamp(t/1000, tz=timezone.utc) for t in data['ts']])
    
    if len(df) < 200: continue
    df_w = compute_indicators(df)
    n = len(df_w)
    
    for i in range(100, n-2):
        row = df_w.iloc[i]
        counts['total_candles'] += 1
        
        if not row['entry']:
            continue
        counts['entry_true'] += 1
        
        whale_val = float(row['whale'])
        if whale_val < WHALE_MIN:
            continue
        counts['whale_min'] += 1
        
        if i + 1 < len(df_w):
            if float(df_w.iloc[i+1]['whale']) >= 0.35:
                continue
        counts['next_whale_pass'] += 1
        
        rsi = float(row['rsi'])
        if np.isnan(rsi) or rsi >= 25:
            continue
        counts['rsi_pass'] += 1
        
        ts = row['ts']
        if ts.weekday() == BLOCK_WEEKDAY:
            continue
        counts['weekday_pass'] += 1
        
        if ts.hour in BLOCK_HOURS:
            continue
        counts['hour_pass'] += 1
        
        ps = max(0, i-96)
        pb = float(df_w.iloc[ps]['close'])
        ep = float(row['close'])
        pump24 = (ep-pb)/pb*100 if pb != 0 else 0
        if pump24 >= 0:
            continue
        counts['pump24_pass'] += 1
        
        if i+1 >= len(df_w): continue
        if float(df_w.iloc[i+1]['close']) <= float(df_w.iloc[i+1]['open']):
            continue
        counts['green_pass'] += 1

print("🔍 FILTER FUNNEL (5 coins sample):")
for k, v in counts.items():
    print(f"  {k}: {v}")
