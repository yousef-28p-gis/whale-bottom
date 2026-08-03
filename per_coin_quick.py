#!/usr/bin/env python3
"""Quick per-coin profit analysis — 3m strategy"""
import json, os, sys
import numpy as np, pandas as pd
from datetime import datetime, timezone

DATA_DIR = '/data/trading28/data/3m_4months'
SHARIAH_FILE = '/data/trading28/config/shariah_coins.json'
COMM = 0.20
TP = 1.3
SL = 0.5
PL = 12
TRAIL = 0.02
MAX_HOLD_H = 4
WHALE_MIN = 0.10
RSI_MAX = 35
SPIKE_MIN = 1.5
STABLES = {'USDT','USDC','BUSD','DAI','TUSD','USDE','XUSD','BFUSD','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','USD1','EUR'}

def compute_indicators(df):
    df = df.copy()
    df['low_lc'] = df['low'].rolling(2).min()
    df['low_sm'] = df['low_lc'].rolling(3).min()
    df['low_hi'] = df['low_sm'].rolling(5).min()
    df['low_raw'] = df['low_hi'].rolling(7).min()
    w = (df['low'].values - df['low_raw'].values) / np.where(df['low_raw'].values != 0, df['low_raw'].values, np.nan) * 100
    df['whale'] = np.clip(w, 0, None)
    vm = df['volume'].rolling(20).mean().values
    df['spike'] = df['volume'].values / np.where(vm != 0, vm, np.nan)
    delta = df['close'].diff().values
    gain = pd.Series(np.where(delta > 0, delta, 0)).rolling(14).mean().values
    loss = pd.Series(np.where(delta < 0, -delta, 0)).rolling(14).mean().values
    df['rsi'] = 100 - 100 / (1 + gain / np.where(loss != 0, loss, np.nan))
    return df

def find_signals(df):
    n = len(df)
    if n < 100:
        return []
    
    whale = df['whale'].values
    spike = df['spike'].values
    rsi = df['rsi'].values
    close = df['close'].values
    open_ = df['open'].values
    
    # Signal mask
    mask = (whale >= WHALE_MIN) & (spike >= SPIKE_MIN) & (rsi < RSI_MAX)
    mask &= ~np.isnan(whale) & ~np.isnan(spike) & ~np.isnan(rsi)
    
    # Confirmation: next candle green
    green = np.zeros(n, dtype=bool)
    green[1:] = close[1:] > open_[1:]
    
    signals = []
    for i in range(50, n - 1):
        if mask[i] and green[i + 1]:
            signals.append({
                'idx': i,
                'entry_price': close[i + 1],
                'entry_time': df.index[i + 1],
                'tp': close[i + 1] * (1 + TP/100),
                'sl': close[i + 1] * (1 - SL/100),
            })
    return signals

def simulate_exit(sig, df):
    """Simulate exit for a single signal"""
    entry_idx = sig['idx'] + 1  # confirmation candle index
    entry_price = sig['entry_price']
    tp = sig['tp']
    sl = sig['sl']
    pl_price = entry_price + (tp - entry_price) * (PL / 100)
    
    max_bars = int(MAX_HOLD_H * 60 / 3)
    peak = entry_price
    trail_price = entry_price
    pl_triggered = False
    
    end_idx = min(entry_idx + max_bars + 1, len(df))
    
    for j in range(entry_idx + 1, end_idx):
        c = float(df.iloc[j]['close'])
        h = float(df.iloc[j]['high'])
        
        # Timeout
        if j - entry_idx >= max_bars:
            return 'TIME', round((c / entry_price - 1) * 100 - COMM, 4)
        # TP (use high for realistic TP hit)
        if h >= tp:
            return 'TP', round(TP - COMM, 4)
        # SL - close only
        if c <= sl:
            return 'SL', round((c / entry_price - 1) * 100 - COMM, 4)
        # PL + Trail (use high for trail)
        if pl_triggered:
            if h > peak:
                peak = h
                trail_price = h * (1 - TRAIL/100)
            if c <= trail_price:
                return 'TRAIL', round((trail_price / entry_price - 1) * 100 - COMM, 4)
        else:
            if h >= pl_price:
                pl_triggered = True
                peak = h
                trail_price = h * (1 - TRAIL/100)
    
    # Should not reach here but handle
    last_c = float(df.iloc[end_idx - 1]['close'])
    return 'TIME', round((last_c / entry_price - 1) * 100 - COMM, 4)

# Load coins
with open(SHARIAH_FILE) as f:
    shariah = json.load(f)
EXCLUDE = {'ETH','BTC','TRX','XRP','QI','LSK','GLMR','XTZ','YFI'}
COINS = [c for c in shariah['halal'] + shariah['halal2'] if c not in STABLES and c not in EXCLUDE]

results = []
for ci, coin in enumerate(COINS):
    fpath = f'{DATA_DIR}/{coin}.json'
    if not os.path.exists(fpath):
        continue
    
    with open(fpath) as f:
        raw = json.load(f)
    
    df = pd.DataFrame(raw)
    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df.set_index('ts', inplace=True)
    
    df = compute_indicators(df)
    signals = find_signals(df)
    
    wins = 0
    losses = 0
    total_pnl = 0
    
    for sig in signals:
        reason, pnl = simulate_exit(sig, df)
        total_pnl += pnl
        if pnl > 0:
            wins += 1
        else:
            losses += 1
    
    total = wins + losses
    wr = wins / total * 100 if total > 0 else 0
    
    results.append({
        'coin': coin,
        'signals': total,
        'wins': wins,
        'losses': losses,
        'wr': wr,
        'total_pnl': round(total_pnl, 2),
        'avg_pnl': round(total_pnl / total, 2) if total > 0 else 0,
    })
    
    if (ci + 1) % 20 == 0:
        print(f'  {ci+1}/{len(COINS)}', flush=True)

# Sort by total_pnl ascending (worst first)
results.sort(key=lambda x: x['total_pnl'])

print(f'\n{"="*70}')
print(f'عملات بأقل من 10% ربح:')
print(f'{"="*70}')
for r in results:
    if r['total_pnl'] < 10:
        print(f"  {r['coin']:12s} | {r['signals']:4d} صفقة | WR {r['wr']:5.1f}% | ربح {r['total_pnl']:+.2f}% | متوسط {r['avg_pnl']:+.2f}%")

print(f'\n{"="*70}')
print(f'عملات بأقل من 5% ربح:')
print(f'{"="*70}')
for r in results:
    if r['total_pnl'] < 5:
        print(f"  {r['coin']:12s} | {r['signals']:4d} صفقة | WR {r['wr']:5.1f}% | ربح {r['total_pnl']:+.2f}% | متوسط {r['avg_pnl']:+.2f}%")

print(f'\n{"="*70}')
print(f'عملات خسرانة:')
print(f'{"="*70}')
for r in results:
    if r['total_pnl'] < 0:
        print(f"  {r['coin']:12s} | {r['signals']:4d} صفقة | WR {r['wr']:5.1f}% | ربح {r['total_pnl']:+.2f}% | متوسط {r['avg_pnl']:+.2f}%")
