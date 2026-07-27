#!/usr/bin/env python3
"""
BIG PUMP STRATEGIES — Target 10%+ gains
Testing filters that predict BIG moves, not just any pump.
"""
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/data/trading28/backtests/pattern_data'
COMMISSION = 0.002
INITIAL_CAPITAL = 1000

cache_file = os.path.join(DATA_DIR, 'daily_all.json')
with open(cache_file) as f:
    all_data = json.load(f)

with open('/data/trading28/config/shariah_coins.json') as f:
    config = json.load(f)
coins_raw = config['halal'] + config['halal2']
seen = set()
coins_raw = [c for c in coins_raw if not (c in seen or seen.add(c))]
blacklist = {'QTUM','ZRO','IOTX','DYM','DGB','SAPIEN','XLM','EDU','BTC','INIT','PARTI','ROBO','PYTH','ANKR'}
valid_coins = set(c for c in coins_raw if c not in blacklist)

print(f"🚀 BIG PUMP STRATEGIES — Target 10%+")

def to_df(data):
    return pd.DataFrame({
        'open': data['open'], 'high': data['high'],
        'low': data['low'], 'close': data['close'], 'volume': data['volume'],
    }, index=pd.to_datetime(data['dates']))

def compute_indicators(df):
    close = df['close']; open_ = df['open']
    high = df['high']; low = df['low']; volume = df['volume']
    ind = {}
    
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    ind['rsi'] = 100 - (100 / (1 + gain.rolling(14).mean() / loss.rolling(14).mean()))
    ind['pct'] = close.pct_change() * 100
    
    # Volume
    ind['vol_ratio'] = volume / volume.rolling(20).mean()
    ind['vol_5d_ratio'] = volume.rolling(5).mean() / volume.rolling(20).mean()
    
    # Price position
    high20 = high.rolling(20).max()
    low20 = low.rolling(20).min()
    ind['range_pos'] = (close - low20) / (high20 - low20)
    
    # Consecutive red
    red = (ind['pct'] < 0).astype(int)
    streak = [0]
    for i in range(1, len(red)):
        streak.append(streak[-1] + 1 if red.iloc[i] else 0)
    ind['red_streak'] = pd.Series(streak, index=red.index)
    
    # SMA
    ind['sma20'] = close.rolling(20).mean()
    ind['sma50'] = close.rolling(50).mean()
    
    # ATR for volatility
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': abs(high - close.shift()),
        'lc': abs(low - close.shift()),
    }).max(axis=1)
    ind['atr_pct'] = tr.rolling(14).mean() / close * 100
    
    # Bollinger
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    ind['bb_pos'] = (close - (sma20 - 2*std20)) / (4*std20)
    
    # Whale bar (simple version for this test)
    vol_mean = volume.rolling(24).mean()
    vol_std = volume.rolling(24).std()
    ind['whale'] = (volume > vol_mean + 2*vol_std).astype(int)
    ind['whale_3d'] = ind['whale'].rolling(3).sum()  # whales in last 3 days
    
    return ind

# ── Entry Functions (no whale requirement unless specified) ──

def s1_rs30_red(ind, i):
    """RSI<30 + PrevRed (best strategy, but with higher TP)"""
    return (ind['rsi'].iloc[i] < 30 and ind['pct'].iloc[i] < 0 and
            not np.isnan(ind['rsi'].iloc[i]))

def s2_rs25_red3(ind, i):
    """RSI<25 + 3+ Red Days (catching deeper oversold)"""
    return (ind['rsi'].iloc[i] < 25 and ind['red_streak'].iloc[i] >= 3 and
            not np.isnan(ind['rsi'].iloc[i]))

def s3_rs25_bottom_vol(ind, i):
    """RSI<25 + Bottom20% + Vol>1.5x"""
    return (ind['rsi'].iloc[i] < 25 and ind['range_pos'].iloc[i] < 0.2 and
            ind['vol_ratio'].iloc[i] > 1.5 and not np.isnan(ind['rsi'].iloc[i]))

def s4_rs20_red2_bottom(ind, i):
    """RSI<20 + 2+Red + Bottom15% (extreme conditions)"""
    return (ind['rsi'].iloc[i] < 20 and ind['red_streak'].iloc[i] >= 2 and
            ind['range_pos'].iloc[i] < 0.15 and not np.isnan(ind['rsi'].iloc[i]))

def s5_rs30_red_bb0(ind, i):
    """RSI<30 + PrevRed + Below BB (oversold + momentum setup)"""
    return (ind['rsi'].iloc[i] < 30 and ind['pct'].iloc[i] < 0 and
            ind['bb_pos'].iloc[i] < 0.1 and not np.isnan(ind['rsi'].iloc[i]))

def s6_rs30_red_vol_whale(ind, i):
    """RSI<30 + PrevRed + Vol + Whale3d (bringing whale back as filter)"""
    return (ind['rsi'].iloc[i] < 30 and ind['pct'].iloc[i] < 0 and
            ind['vol_ratio'].iloc[i] > 1.3 and ind['whale_3d'].iloc[i] >= 1 and
            not np.isnan(ind['rsi'].iloc[i]))

def s7_rs25_red_atr(ind, i):
    """RSI<25 + PrevRed + High ATR (volatility expansion before move)"""
    return (ind['rsi'].iloc[i] < 25 and ind['pct'].iloc[i] < 0 and
            ind['atr_pct'].iloc[i] > ind['atr_pct'].rolling(20).mean().iloc[i] * 1.3 and
            not np.isnan(ind['rsi'].iloc[i]))

def s8_20red_rs25(ind, i):
    """20% red day + RSI<30 (capitulation candle)"""
    return (ind['pct'].iloc[i] < -15 and ind['rsi'].iloc[i] < 35 and
            not np.isnan(ind['rsi'].iloc[i]))

# ── Test multiple TP/SL combos ──────────────────────────
ENTRY_STRATEGIES = [
    ("RSI<30+PrevRed", s1_rs30_red),
    ("RSI<25+3RedDays", s2_rs25_red3),
    ("RSI<25+Bottom+Vol", s3_rs25_bottom_vol),
    ("RSI<20+2Red+Bottom15%", s4_rs20_red2_bottom),
    ("RSI<30+PrevRed+BB<0", s5_rs30_red_bb0),
    ("RSI<30+PrevRed+Vol+Whale", s6_rs30_red_vol_whale),
    ("RSI<25+PrevRed+ATR", s7_rs25_red_atr),
    ("-20%Day+RSI<35", s8_20red_rs25),
]

TP_SL_COMBOS = [
    (0.10, 0.05, "TP10/SL5"),
    (0.12, 0.05, "TP12/SL5"),
    (0.15, 0.06, "TP15/SL6"),
    (0.10, 0.04, "TP10/SL4"),
]

# ── Backtest ────────────────────────────────────────────
def backtest(all_data, entry_fn, tp, sl, max_hold=7):
    all_signals = []
    for coin, data in all_data.items():
        if coin not in valid_coins: continue
        df = to_df(data)
        if len(df) < 60: continue
        ind = compute_indicators(df)
        n = len(df)
        
        for i in range(30, n - 1):
            try:
                ok = entry_fn(ind, i)
            except:
                ok = False
            if not ok: continue
            
            # Confirmation: next candle green
            if df['close'].iloc[i+1] <= df['open'].iloc[i+1]:
                continue
            
            entry_idx = i + 1
            if entry_idx >= n: continue
            entry_price = df['close'].iloc[entry_idx]
            
            all_signals.append({
                'coin': coin, 'idx': entry_idx,
                'entry_price': entry_price,
                'date': df.index[entry_idx],
            })
    
    all_signals.sort(key=lambda s: s['date'])
    trades = []
    capital = INITIAL_CAPITAL
    active = {}
    
    for sig in all_signals:
        coin = sig['coin']; entry_idx = sig['idx']
        if coin in active and active[coin] > entry_idx: continue
        
        data = all_data[coin]
        close_arr = np.array(data['close'])
        high_arr = np.array(data['high'])
        low_arr = np.array(data['low'])
        n = len(close_arr)
        
        tp_price = sig['entry_price'] * (1 + tp)
        sl_price = sig['entry_price'] * (1 - sl)
        
        exit_price = None; exit_type = None; exit_idx = None
        
        for j in range(entry_idx + 1, min(entry_idx + max_hold, n)):
            if low_arr[j] <= sl_price:
                exit_price = sl_price; exit_type = 'SL'; exit_idx = j; break
            elif high_arr[j] >= tp_price:
                exit_price = tp_price; exit_type = 'TP'; exit_idx = j; break
        
        if exit_price is None:
            end = min(entry_idx + max_hold, n - 1)
            exit_price = close_arr[end]
            exit_type = 'TIME'
            exit_idx = end
        
        pnl_pct = (exit_price / sig['entry_price'] - 1) * 100 - COMMISSION * 100
        size = capital * 0.10
        pnl_usd = size * pnl_pct / 100
        capital += pnl_usd
        
        trades.append({
            'pnl_pct': pnl_pct, 'pnl_usd': pnl_usd,
            'type': exit_type, 'capital': capital,
            'entry_date': sig['date'],
        })
        active[coin] = exit_idx
        active = {k: v for k, v in active.items() if v > entry_idx}
    
    return trades, capital

# ── Run all combos ──────────────────────────────────────
print(f"\n{'='*95}")
print(f"🚀 BIG PUMP STRATEGIES — Results (60-day backtest, all coins)")
print(f"{'='*95}")

all_results = []

for tp, sl, label in TP_SL_COMBOS:
    print(f"\n{'─'*95}")
    print(f"📐 {label}")
    print(f"{'─'*95}")
    print(f"{'Entry Strategy':<30s} {'Trades':>6s} {'WR':>6s} {'Return':>8s} {'MaxDD':>7s} {'PF':>6s} {'TP%':>6s} {'Avg':>7s}")
    print(f"{'-'*80}")
    
    for name, fn in ENTRY_STRATEGIES:
        trades, final_cap = backtest(all_data, fn, tp, sl)
        
        if not trades:
            all_results.append({'name': f"{name} | {label}", 'trades': 0, 'wr': 0, 'return': 0, 'max_dd': 0, 'pf': 0})
            print(f"{name:<30s} {'0':>6s} {'-':>6s} {'-':>8s} {'-':>7s} {'-':>6s} {'-':>6s} {'-':>7s}")
            continue
        
        df = pd.DataFrame(trades)
        wins = df[df['pnl_pct'] > 0]
        losses = df[df['pnl_pct'] <= 0]
        wr = len(wins) / len(df) * 100
        
        eq = np.array([INITIAL_CAPITAL] + [t['capital'] for t in trades])
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak * 100
        
        ret = (final_cap / INITIAL_CAPITAL - 1) * 100
        
        pf = abs(wins['pnl_usd'].sum() / losses['pnl_usd'].sum()) if len(losses) > 0 and losses['pnl_usd'].sum() != 0 else float('inf')
        
        tp_hit = len(df[df['type'] == 'TP'])
        
        ret_str = f"+{ret:.1f}%" if ret > 0 else f"{ret:.1f}%"
        print(f"{name:<30s} {len(df):>6d} {wr:>5.1f}% {ret_str:>8s} {dd.min():>6.2f}% {pf:>5.2f} {tp_hit:>5d} {df['pnl_pct'].mean():>+6.2f}%")
        
        all_results.append({
            'name': f"{name} | {label}",
            'trades': len(df), 'wr': round(wr, 1),
            'return': round(ret, 1), 'max_dd': round(dd.min(), 2),
            'pf': round(pf, 2), 'tp_hits': tp_hit,
            'avg_pnl': round(df['pnl_pct'].mean(), 2),
        })

# ── Best overall ────────────────────────────────────────
all_results.sort(key=lambda r: -r['return'])

print(f"\n{'='*95}")
print(f"🏆 TOP 10 BIG-PUMP STRATEGIES (Ranked by Return)")
print(f"{'='*95}")
for i, r in enumerate(all_results[:10]):
    if r['trades'] == 0: continue
    print(f"  #{i+1}: {r['name']:<45s} {r['trades']:>4d} trades | WR {r['wr']}% | Return {r['return']:+.1f}% | DD {r['max_dd']}% | PF {r['pf']} | TP:{r['tp_hits']}")
