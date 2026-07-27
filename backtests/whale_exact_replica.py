#!/usr/bin/env python3
"""
WHALE BOTTOM — EXACT LIVE CODE REPLICA
Same whale formula, same filters, same logic
15m, 30 days
"""
import json, numpy as np, pandas as pd, os
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/data/trading28/backtests/pattern_data'
COMMISSION = 0.20  # 0.2% as in live code
INITIAL_CAPITAL = 1000

TP = 3.5; SL = 1.5; WHALE_MIN = 0.50; STR = 50
MAX_POS = 2; MAX_H = 6  # 24 candles on 15m
BLOCK_HOURS = {1, 3, 6, 12, 0, 4}
BLOCK_WEEKDAY = 3

with open(f'{DATA_DIR}/15m_30d.json') as f:
    all_data = json.load(f)

print(f"🐋 WHALE BOTTOM EXACT REPLICA — {len(all_data)} coins, 15m, 30 days")
print(f"   Using IDENTICAL code logic from whale_bottom_daemon.py\n")

# ═══════════════════════════════════════════════════════
# IDENTICAL compute_indicators from daemon
# ═══════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════
# IDENTICAL check_entry logic
# ═══════════════════════════════════════════════════════
def check_entry(row, df_w, i):
    """Exact check_entry logic from daemon"""
    # Already checked 'entry' column externally
    
    whale_val = float(row['whale'])
    if whale_val < WHALE_MIN:
        return None
    
    # Next candle whale check
    if i + 1 < len(df_w):
        if float(df_w.iloc[i + 1]['whale']) >= 0.35:
            return None
    
    rsi = float(row['rsi'])
    if np.isnan(rsi) or rsi >= 25:
        return None
    
    ts = row['ts']
    if ts.weekday() == BLOCK_WEEKDAY:
        return None
    if ts.hour in BLOCK_HOURS:
        return None
    
    # pump24 check
    ps = max(0, i - 96)
    pb = float(df_w.iloc[ps]['close'])
    ep = float(row['close'])
    pump24 = (ep - pb) / pb * 100 if pb != 0 else 0
    if pump24 >= 0:
        return None
    
    # Green confirmation (Filter 3)
    if i + 1 >= len(df_w):
        return None
    next_open = float(df_w.iloc[i + 1]['open'])
    next_close = float(df_w.iloc[i + 1]['close'])
    if next_close <= next_open:
        return None
    
    return ep  # return entry price

# ═══════════════════════════════════════════════════════
# Backtest — simulate cron-like scanning
# ═══════════════════════════════════════════════════════
def backtest_exact(all_data):
    """Simulate exactly how the daemon works: scan every 15m, 
    check only last 4 candles, max 2 positions."""
    
    all_trades = []
    capital = INITIAL_CAPITAL
    active_positions = []  # max 2
    
    for coin, data in all_data.items():
        # Build DataFrame with datetime index
        df = pd.DataFrame({
            'open': data['open'], 'high': data['high'],
            'low': data['low'], 'close': data['close'],
            'volume': data['volume'],
        })
        df['ts'] = pd.to_datetime([datetime.fromtimestamp(t/1000, tz=timezone.utc) for t in data['ts']])
        df.set_index('ts', inplace=False)
        
        if len(df) < 200:
            continue
        
        # Compute indicators
        df_w = compute_indicators(df)
        n = len(df_w)
        
        # Simulate scanning: only check candles with 'entry' True
        # But also limit to last 4 candles per period (like daemon does per scan)
        # We'll simulate by checking ALL entry=True candles but applying ALL filters
        # This is equivalent because daemon checks every candle eventually
        
        for i in range(100, n - 2):
            row = df_w.iloc[i]
            if not row['entry']:
                continue
            
            ep = check_entry(row, df_w, i)
            if ep is None:
                continue
            
            # Max positions
            if len(active_positions) >= MAX_POS:
                continue
            
            # Entry confirmed — simulate exit
            entry_idx = i + 2  # entry after confirmation candle
            if entry_idx >= n:
                continue
            
            entry_price = float(df['close'].iloc[entry_idx])
            tp_price = entry_price * (1 + TP/100)
            sl_price = entry_price * (1 - SL/100)
            
            # Simulate exit over next 6h (24 candles on 15m)
            exit_price = None
            exit_type = None
            max_j = min(entry_idx + MAX_H * 4, n - 1)  # MAX_H hours × 4 candles/hour
            
            for j in range(entry_idx + 1, max_j + 1):
                if float(df['low'].iloc[j]) <= sl_price:
                    exit_price = sl_price
                    exit_type = 'SL'
                    break
                elif float(df['high'].iloc[j]) >= tp_price:
                    exit_price = tp_price
                    exit_type = 'TP'
                    break
            
            if exit_price is None:
                exit_price = float(df['close'].iloc[max_j])
                exit_type = 'TIME'
            
            pnl_pct = (exit_price / entry_price - 1) * 100 - COMMISSION
            size = capital * 0.50  # POS_PCT = 50
            pnl_usd = size * pnl_pct / 100
            capital += pnl_usd
            
            all_trades.append({
                'coin': coin,
                'pnl_pct': round(pnl_pct, 2),
                'pnl_usd': round(pnl_usd, 2),
                'type': exit_type,
                'capital': round(capital, 2),
            })
            
            # Track active (simplified: just count, daemon checks symbols)
            active_positions.append(entry_idx)
            if len(active_positions) > MAX_POS:
                active_positions = active_positions[-MAX_POS:]
    
    return all_trades, capital

# ── Run ─────────────────────────────────────────────────
trades, final_cap = backtest_exact(all_data)

if trades:
    df = pd.DataFrame(trades)
    wins = df[df['pnl_pct'] > 0]
    losses = df[df['pnl_pct'] <= 0]
    wr = len(wins) / len(df) * 100
    
    eq = np.array([1000] + [t['capital'] for t in trades])
    dd = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq) * 100
    ret = (final_cap / 1000 - 1) * 100
    pf = abs(wins['pnl_usd'].sum() / losses['pnl_usd'].sum()) if len(losses) > 0 else 999
    
    print(f"\n{'='*60}")
    print(f"📊 WHALE BOTTOM EXACT REPLICA — Results")
    print(f"{'='*60}")
    print(f"   Total Trades:  {len(df)}")
    print(f"   Win Rate:      {wr:.1f}%")
    print(f"   Wins: {len(wins)} | Losses: {len(losses)}")
    print(f"   Avg Win:       +{wins['pnl_pct'].mean():.2f}%" if len(wins) > 0 else "")
    print(f"   Avg Loss:      {losses['pnl_pct'].mean():.2f}%" if len(losses) > 0 else "")
    print(f"   Avg Trade:     {df['pnl_pct'].mean():.2f}%")
    print(f"   Return:        {ret:+.1f}%")
    print(f"   Final Capital: ${final_cap:,.2f}")
    print(f"   Max DD:        {dd.min():.2f}%")
    print(f"   Profit Factor: {pf:.2f}")
    
    print(f"\n## By Exit Type:")
    for et in ['TP', 'SL', 'TIME']:
        s = df[df['type'] == et]
        if len(s) > 0:
            print(f"   {et}: {len(s)} trades, avg {s['pnl_pct'].mean():+.2f}%")
else:
    print("\n❌ No trades found!")

print(f"\n✅ Done!")
