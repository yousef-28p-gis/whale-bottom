#!/usr/bin/env python3
"""
CANDLESTICK PATTERN STRATEGIES — Engulfing, Hammer, Stars, etc.
No whale indicators. Pure candlestick patterns + RSI/Volume filters.
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

# ── Load daily data ────────────────────────────────────
cache_file = os.path.join(DATA_DIR, 'daily_all.json')
if not os.path.exists(cache_file):
    print("❌ No daily cache. Run pattern_discovery.py first.")
    exit(1)

with open(cache_file) as f:
    all_data = json.load(f)

with open('/data/trading28/config/shariah_coins.json') as f:
    config = json.load(f)
coins_raw = config['halal'] + config['halal2']
seen = set()
coins_raw = [c for c in coins_raw if not (c in seen or seen.add(c))]
blacklist = {'QTUM','ZRO','IOTX','DYM','DGB','SAPIEN','XLM','EDU','BTC','INIT','PARTI','ROBO','PYTH','ANKR'}
valid_coins = set(c for c in coins_raw if c not in blacklist)

print(f"🕯️ CANDLESTICK PATTERN STRATEGIES")
print(f"   Coins: {len(all_data)} | No whale indicators")

def to_df(data):
    df = pd.DataFrame({
        'open': data['open'], 'high': data['high'],
        'low': data['low'], 'close': data['close'], 'volume': data['volume'],
    }, index=pd.to_datetime(data['dates']))
    return df

def compute_indicators(df):
    close = df['close']; open_ = df['open']
    high = df['high']; low = df['low']; volume = df['volume']
    
    ind = {}
    
    # RSI(14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    ind['rsi'] = 100 - (100 / (1 + gain.rolling(14).mean() / loss.rolling(14).mean()))
    
    ind['pct'] = close.pct_change() * 100
    ind['vol_ratio'] = volume / volume.rolling(20).mean()
    ind['sma20'] = close.rolling(20).mean()
    ind['sma50'] = close.rolling(50).mean()
    
    # Candle properties
    ind['body'] = abs(close - open_)
    ind['upper_wick'] = high - np.maximum(close, open_)
    ind['lower_wick'] = np.minimum(close, open_) - low
    ind['total_range'] = high - low
    ind['is_green'] = close > open_
    ind['is_red'] = close < open_
    ind['body_pct'] = ind['body'] / ind['total_range'] * 100  # body % of total range
    
    # Price position
    high20 = high.rolling(20).max()
    low20 = low.rolling(20).min()
    ind['range_pos'] = (close - low20) / (high20 - low20)
    
    # Consecutive red days
    red = (ind['pct'] < 0).astype(int)
    streak_vals = [0]
    for i in range(1, len(red)):
        streak_vals.append(streak_vals[-1] + 1 if red.iloc[i] else 0)
    ind['red_streak'] = pd.Series(streak_vals, index=red.index)
    
    return ind

# ── Pattern Detection Functions ─────────────────────────

def is_bullish_engulfing(ind, i):
    """Candle[i-1] red, Candle[i] green engulfing it."""
    if i < 1:
        return False
    prev_red = ind['is_red'].iloc[i-1]
    curr_green = ind['is_green'].iloc[i]
    if not (prev_red and curr_green):
        return False
    # Current open < prev close AND current close > prev open
    engulfing = (ind['open'].iloc[i] < ind['close'].iloc[i-1] and 
                 ind['close'].iloc[i] > ind['open'].iloc[i-1])
    return engulfing

def is_piercing_line(ind, i):
    """Red candle then green that opens below prev low but closes > 50% of prev body."""
    if i < 1:
        return False
    prev_red = ind['is_red'].iloc[i-1]
    curr_green = ind['is_green'].iloc[i]
    if not (prev_red and curr_green):
        return False
    prev_body = ind['body'].iloc[i-1]
    if prev_body == 0:
        return False
    # Current close > 50% into previous body
    mid_point = (ind['open'].iloc[i-1] + ind['close'].iloc[i-1]) / 2
    return (ind['open'].iloc[i] < ind['low'].iloc[i-1] and 
            ind['close'].iloc[i] > mid_point)

def is_hammer(ind, i):
    """Small body at top, long lower wick (2x body), little upper wick."""
    body = ind['body'].iloc[i]
    lower = ind['lower_wick'].iloc[i]
    upper = ind['upper_wick'].iloc[i]
    total = ind['total_range'].iloc[i]
    if total == 0 or body == 0:
        return False
    return (lower >= 2 * body and 
            upper <= body * 0.5 and
            ind['body_pct'].iloc[i] < 30)

def is_inverted_hammer(ind, i):
    """Small body at bottom, long upper wick, little lower wick."""
    body = ind['body'].iloc[i]
    upper = ind['upper_wick'].iloc[i]
    lower = ind['lower_wick'].iloc[i]
    if body == 0:
        return False
    return (upper >= 2 * body and 
            lower <= body * 0.5 and
            ind['body_pct'].iloc[i] < 30)

def is_morning_star(ind, i):
    """Red → Doji/small → Green. 3-candle reversal."""
    if i < 2:
        return False
    c0_red = ind['is_red'].iloc[i-2]
    c1_small = ind['body_pct'].iloc[i-1] < 20  # doji or small body
    c2_green = ind['is_green'].iloc[i]
    if not (c0_red and c1_small and c2_green):
        return False
    # C2 should close above midpoint of C0
    mid_c0 = (ind['open'].iloc[i-2] + ind['close'].iloc[i-2]) / 2
    return ind['close'].iloc[i] > mid_c0

def is_three_white_soldiers(ind, i):
    """Three consecutive green candles with increasing closes."""
    if i < 2:
        return False
    c0 = ind['is_green'].iloc[i-2] and ind['body_pct'].iloc[i-2] > 30
    c1 = ind['is_green'].iloc[i-1] and ind['body_pct'].iloc[i-1] > 30
    c2 = ind['is_green'].iloc[i] and ind['body_pct'].iloc[i] > 30
    if not (c0 and c1 and c2):
        return False
    return (ind['close'].iloc[i-1] > ind['close'].iloc[i-2] and
            ind['close'].iloc[i] > ind['close'].iloc[i-1])

def is_tweezer_bottom(ind, i):
    """Two candles with same/similar lows (support)."""
    if i < 1:
        return False
    diff = abs(ind['low'].iloc[i] - ind['low'].iloc[i-1])
    avg_low = (ind['low'].iloc[i] + ind['low'].iloc[i-1]) / 2
    if avg_low == 0:
        return False
    return (diff / avg_low < 0.005 and  # <0.5% difference
            ind['is_green'].iloc[i] and  # second is green
            ind['is_red'].iloc[i-1])     # first is red

# ── Strategy Definitions ────────────────────────────────

STRATEGIES = []

# 1: Pure Bullish Engulfing
def s1(ind, i):
    return is_bullish_engulfing(ind, i)
STRATEGIES.append(("BullishEngulfing", s1, 0.05, 0.03, 5))

# 2: Bullish Engulfing + RSI<40 + Downtrend (below SMA20)
def s2(ind, i):
    return (is_bullish_engulfing(ind, i) and
            ind['rsi'].iloc[i] < 40 and
            ind['close'].iloc[i] < ind['sma20'].iloc[i] and
            not np.isnan(ind['rsi'].iloc[i]))
STRATEGIES.append(("Engulf+RSI<40+<SMA20", s2, 0.05, 0.03, 5))

# 3: Bullish Engulfing + RSI<30 + Volume > 1.5x
def s3(ind, i):
    return (is_bullish_engulfing(ind, i) and
            ind['rsi'].iloc[i] < 30 and
            ind['vol_ratio'].iloc[i] > 1.5 and
            not np.isnan(ind['rsi'].iloc[i]))
STRATEGIES.append(("Engulf+RSI<30+Vol", s3, 0.05, 0.03, 5))

# 4: Bullish Engulfing + 2+ Red Days Before
def s4(ind, i):
    return (is_bullish_engulfing(ind, i) and
            ind['red_streak'].iloc[i-1] >= 2)  # 2+ red before the engulfing
STRATEGIES.append(("Engulf+2RedDays", s4, 0.05, 0.03, 5))

# 5: Hammer + RSI<30
def s5(ind, i):
    return (is_hammer(ind, i) and
            ind['rsi'].iloc[i] < 30 and
            not np.isnan(ind['rsi'].iloc[i]))
STRATEGIES.append(("Hammer+RSI<30", s5, 0.05, 0.03, 5))

# 6: Hammer + RSI<30 + Price Bottom 20%
def s6(ind, i):
    return (is_hammer(ind, i) and
            ind['rsi'].iloc[i] < 30 and
            ind['range_pos'].iloc[i] < 0.2 and
            not np.isnan(ind['rsi'].iloc[i]))
STRATEGIES.append(("Hammer+RSI<30+Bottom", s6, 0.05, 0.03, 5))

# 7: Morning Star + RSI<35
def s7(ind, i):
    return (is_morning_star(ind, i) and
            ind['rsi'].iloc[i] < 35 and
            not np.isnan(ind['rsi'].iloc[i]))
STRATEGIES.append(("MorningStar+RSI<35", s7, 0.05, 0.03, 5))

# 8: Piercing Line + RSI<30
def s8(ind, i):
    return (is_piercing_line(ind, i) and
            ind['rsi'].iloc[i] < 30 and
            not np.isnan(ind['rsi'].iloc[i]))
STRATEGIES.append(("Piercing+RSI<30", s8, 0.05, 0.03, 5))

# 9: Tweezer Bottom + RSI<30
def s9(ind, i):
    return (is_tweezer_bottom(ind, i) and
            ind['rsi'].iloc[i] < 30 and
            not np.isnan(ind['rsi'].iloc[i]))
STRATEGIES.append(("Tweezer+RSI<30", s9, 0.05, 0.03, 5))

# 10: Three White Soldiers (momentum)
def s10(ind, i):
    return (is_three_white_soldiers(ind, i) and
            ind['close'].iloc[i] > ind['sma20'].iloc[i])
STRATEGIES.append(("3WhiteSoldiers>SMA20", s10, 0.05, 0.03, 5))

# 11: Inverted Hammer + RSI<25 + Prev Day Red
def s11(ind, i):
    return (is_inverted_hammer(ind, i) and
            ind['rsi'].iloc[i] < 25 and
            ind['is_red'].iloc[i] and
            not np.isnan(ind['rsi'].iloc[i]))
STRATEGIES.append(("InvHammer+RSI<25+Red", s11, 0.05, 0.03, 5))

# 12: Best combo — Engulfing + RSI<35 + PrevRed + <SMA20 + Vol
def s12(ind, i):
    return (is_bullish_engulfing(ind, i) and
            ind['rsi'].iloc[i] < 35 and
            ind['red_streak'].iloc[i-1] >= 1 and
            ind['close'].iloc[i] < ind['sma20'].iloc[i] and
            ind['vol_ratio'].iloc[i] > 1.2 and
            not np.isnan(ind['rsi'].iloc[i]))
STRATEGIES.append(("Engulf+RSI<35+Red+<SMA20+Vol", s12, 0.05, 0.03, 5))

# ── Backtest Engine ─────────────────────────────────────
def backtest_strategy(all_data, entry_fn, tp_pct, sl_pct, max_hold):
    all_signals = []
    
    for coin, data in all_data.items():
        if coin not in valid_coins:
            continue
        df = to_df(data)
        if len(df) < 60:
            continue
        ind = compute_indicators(df)
        n = len(df)
        
        for i in range(30, n - 1):
            try:
                ok = entry_fn(ind, i)
            except:
                ok = False
            if not ok:
                continue
            
            # Entry on next candle's close (close-only, no intra-candle)
            entry_idx = i + 1
            if entry_idx >= n:
                continue
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
        coin = sig['coin']
        entry_idx = sig['idx']
        if coin in active and active[coin] > entry_idx:
            continue
        
        data = all_data[coin]
        close_arr = np.array(data['close'])
        high_arr = np.array(data['high'])
        low_arr = np.array(data['low'])
        n = len(close_arr)
        
        tp_price = sig['entry_price'] * (1 + tp_pct)
        sl_price = sig['entry_price'] * (1 - sl_pct)
        
        exit_price = None
        exit_type = None
        exit_idx = None
        
        for j in range(entry_idx + 1, min(entry_idx + max_hold, n)):
            if low_arr[j] <= sl_price:
                exit_price = sl_price; exit_type = 'SL'; exit_idx = j; break
            elif high_arr[j] >= tp_price:
                exit_price = tp_price; exit_type = 'TP'; exit_idx = j; break
        
        if exit_price is None:
            end = min(entry_idx + max_hold, n - 1)
            exit_price = close_arr[end]
            exit_type = 'TIME' if end == entry_idx + max_hold else 'EOD'
            exit_idx = end
        
        pnl_pct = (exit_price / sig['entry_price'] - 1) * 100 - COMMISSION * 100
        size = capital * 0.10
        pnl_usd = size * pnl_pct / 100
        capital += pnl_usd
        
        trades.append({
            'pnl_pct': pnl_pct, 'pnl_usd': pnl_usd,
            'type': exit_type, 'capital': capital,
        })
        active[coin] = exit_idx
        active = {k: v for k, v in active.items() if v > entry_idx}
    
    return trades, capital

def analyze(name, trades, final_cap):
    if not trades:
        return {'name': name, 'trades': 0, 'wr': 0, 'return': 0, 'max_dd': 0, 'pf': 0, 
                'avg_trade': 0, 'avg_win': 0, 'avg_loss': 0}
    df = pd.DataFrame(trades)
    wins = df[df['pnl_pct'] > 0]
    losses = df[df['pnl_pct'] <= 0]
    wr = len(wins) / len(df) * 100
    eq = np.array([INITIAL_CAPITAL] + [t['capital'] for t in trades])
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100
    ret = (final_cap / INITIAL_CAPITAL - 1) * 100
    pf = abs(wins['pnl_usd'].sum() / losses['pnl_usd'].sum()) if len(losses) > 0 else float('inf')
    return {
        'name': name, 'trades': len(df), 'wr': round(wr, 1),
        'return': round(ret, 1), 'max_dd': round(dd.min(), 2),
        'pf': round(pf, 2), 'avg_trade': round(df['pnl_pct'].mean(), 2),
        'avg_win': round(wins['pnl_pct'].mean(), 2) if len(wins) > 0 else 0,
        'avg_loss': round(losses['pnl_pct'].mean(), 2) if len(losses) > 0 else 0,
    }

# ── Run All ─────────────────────────────────────────────
print(f"\n{'='*85}")
print(f"🕯️ CANDLESTICK PATTERN STRATEGIES — Backtest Results")
print(f"{'='*85}")
print(f"{'Strategy':<35s} {'Trades':>6s} {'WR':>6s} {'Return':>8s} {'MaxDD':>7s} {'PF':>6s} {'Avg':>7s}")
print(f"{'-'*85}")

results = []
for name, entry_fn, tp, sl, mh in STRATEGIES:
    trades, final_cap = backtest_strategy(all_data, entry_fn, tp, sl, mh)
    r = analyze(name, trades, final_cap)
    results.append(r)
    ret_str = f"+{r['return']}%" if r['return'] > 0 else f"{r['return']}%"
    print(f"{name:<35s} {r['trades']:>6d} {r['wr']:>5.1f}% {ret_str:>8s} {r['max_dd']:>6.2f}% {r['pf']:>5.2f} {r['avg_trade']:>+6.2f}%")

# ── Rankings ────────────────────────────────────────────
results.sort(key=lambda r: -r['return'])
print(f"\n{'='*85}")
print(f"🥇 TOP 5 BY RETURN")
print(f"{'='*85}")
for i, r in enumerate(results[:5]):
    extra = f" | AvgWin +{r['avg_win']}% | AvgLoss {r['avg_loss']}%"
    print(f"  #{i+1}: {r['name']} — {r['trades']} trades, WR {r['wr']}%, Return {r['return']:+.1f}%, DD {r['max_dd']}%, PF {r['pf']}{extra}")

results_by_wr = sorted(results, key=lambda r: -r['wr'])
print(f"\n{'='*85}")
print(f"🎯 TOP 5 BY WIN RATE")
print(f"{'='*85}")
for i, r in enumerate(results_by_wr[:5]):
    extra = f" | AvgWin +{r['avg_win']}% | AvgLoss {r['avg_loss']}%"
    print(f"  #{i+1}: {r['name']} — {r['trades']} trades, WR {r['wr']}%, Return {r['return']:+.1f}%, DD {r['max_dd']}%, PF {r['pf']}{extra}")

print(f"\n✅ All candlestick patterns tested!")
