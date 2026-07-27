#!/usr/bin/env python3
"""
NON-WHALE STRATEGIES — Test multiple indicator combinations
No whale detection at all. Pure technical indicators.
"""
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/data/trading28/backtests/pattern_data'
COMMISSION = 0.002
INITIAL_CAPITAL = 1000

# ── Load cached daily data ──────────────────────────────
cache_file = os.path.join(DATA_DIR, 'daily_all.json')
if not os.path.exists(cache_file):
    print("❌ No daily cache. Run pattern_discovery.py first.")
    exit(1)

with open(cache_file) as f:
    all_data = json.load(f)

# Load coin list
with open('/data/trading28/config/shariah_coins.json') as f:
    config = json.load(f)
coins_raw = config['halal'] + config['halal2']
seen = set()
coins_raw = [c for c in coins_raw if not (c in seen or seen.add(c))]
blacklist = {'QTUM','ZRO','IOTX','DYM','DGB','SAPIEN','XLM','EDU','BTC','INIT','PARTI','ROBO','PYTH','ANKR'}
valid_coins = set(c for c in coins_raw if c not in blacklist)

print(f"🔬 NON-WHALE STRATEGIES — Pure Technical Indicators")
print(f"   Coins available: {len(all_data)}")

def to_df(data):
    """Convert cached JSON to DataFrame."""
    df = pd.DataFrame({
        'open': data['open'],
        'high': data['high'],
        'low': data['low'],
        'close': data['close'],
        'volume': data['volume'],
    }, index=pd.to_datetime(data['dates']))
    return df

def compute_indicators(df):
    """Compute all technical indicators."""
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    ind = {}
    
    # RSI(14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    ind['rsi'] = 100 - (100 / (1 + rs))
    
    # Daily % change
    ind['pct'] = close.pct_change() * 100
    
    # Volume ratio (today vs 20-day avg)
    ind['vol_ratio'] = volume / volume.rolling(20).mean()
    
    # Bollinger Bands (20,2)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    ind['bb_upper'] = sma20 + 2 * std20
    ind['bb_lower'] = sma20 - 2 * std20
    ind['bb_width'] = (ind['bb_upper'] - ind['bb_lower']) / sma20
    ind['bb_pos'] = (close - ind['bb_lower']) / (ind['bb_upper'] - ind['bb_lower'])  # 0=bottom, 1=top
    
    # ATR(14)
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': abs(high - close.shift()),
        'lc': abs(low - close.shift()),
    }).max(axis=1)
    ind['atr'] = tr.rolling(14).mean()
    ind['atr_pct'] = ind['atr'] / close * 100
    
    # MA crossovers
    ind['sma10'] = close.rolling(10).mean()
    ind['sma20'] = close.rolling(20).mean()
    ind['sma50'] = close.rolling(50).mean()
    ind['ma_trend'] = (ind['sma10'] - ind['sma20']) / ind['sma20'] * 100  # positive = uptrend
    
    # Consecutive red days
    red = (ind['pct'] < 0).astype(int)
    streak_vals = []
    streak = 0
    for i in range(len(red)):
        if red.iloc[i] == 1:
            streak += 1
        else:
            streak = 0
        streak_vals.append(streak)
    ind['red_streak'] = pd.Series(streak_vals, index=red.index)
    
    # Price position in 20-day range
    high20 = high.rolling(20).max()
    low20 = low.rolling(20).min()
    ind['range_pos'] = (close - low20) / (high20 - low20)  # 0=20d low, 1=20d high
    
    # Volume trend (last 5d vs previous 15d)
    ind['vol_5d'] = volume.rolling(5).mean()
    ind['vol_15d'] = volume.rolling(15).mean()
    ind['vol_trend'] = ind['vol_5d'] / ind['vol_15d']
    
    # Doji detection (small body vs range)
    body = abs(close - df['open'])
    candle_range = high - low
    ind['is_doji'] = (body / candle_range < 0.1) & (candle_range > 0)
    
    return ind

# ── Strategy definitions ────────────────────────────────
# Each strategy: (name, entry_condition_fn, tp_pct, sl_pct, max_hold_days)
# entry_condition_fn receives (indicators_dict, index_i) returns bool

STRATEGIES = []

# Strategy 1: RSI Oversold + Previous Day Red
def s1_entry(ind, i):
    return (ind['rsi'].iloc[i] < 30 and 
            ind['pct'].iloc[i] < 0 and
            not np.isnan(ind['rsi'].iloc[i]))
STRATEGIES.append(("RSI<30 + PrevRed", s1_entry, 0.05, 0.025, 5))

# Strategy 2: RSI Oversold + Price at 20-day Bottom + Volume Rising
def s2_entry(ind, i):
    return (ind['rsi'].iloc[i] < 30 and
            ind['range_pos'].iloc[i] < 0.2 and
            ind['vol_trend'].iloc[i] > 1.0 and
            not np.isnan(ind['rsi'].iloc[i]))
STRATEGIES.append(("RSI<30 + Bottom20d + VolRising", s2_entry, 0.05, 0.025, 5))

# Strategy 3: RSI Oversold + Bollinger Bottom Touch
def s3_entry(ind, i):
    return (ind['rsi'].iloc[i] < 30 and
            ind['bb_pos'].iloc[i] < 0.1 and  # below lower band
            not np.isnan(ind['rsi'].iloc[i]))
STRATEGIES.append(("RSI<30 + BBLower", s3_entry, 0.05, 0.025, 5))

# Strategy 4: 3+ Red Days + RSI<35 + Price Bottom
def s4_entry(ind, i):
    return (ind['red_streak'].iloc[i] >= 3 and
            ind['rsi'].iloc[i] < 35 and
            ind['range_pos'].iloc[i] < 0.3 and
            not np.isnan(ind['rsi'].iloc[i]))
STRATEGIES.append(("3RedDays + RSI<35 + Bottom", s4_entry, 0.05, 0.025, 5))

# Strategy 5: RSI<25 + PrevDayRed + VolSpike
def s5_entry(ind, i):
    return (ind['rsi'].iloc[i] < 25 and
            ind['pct'].iloc[i] < 0 and
            ind['vol_ratio'].iloc[i] > 1.3 and
            not np.isnan(ind['rsi'].iloc[i]))
STRATEGIES.append(("RSI<25 + PrevRed + Vol>1.3x", s5_entry, 0.05, 0.025, 5))

# Strategy 6: RSI<30 + BB squeeze (narrow bands) + VolBreakout
def s6_entry(ind, i):
    return (ind['rsi'].iloc[i] < 30 and
            ind['bb_width'].iloc[i] < ind['bb_width'].rolling(20).mean().iloc[i] * 0.8 and  # squeeze
            ind['vol_ratio'].iloc[i] > 1.5 and
            not np.isnan(ind['rsi'].iloc[i]))
STRATEGIES.append(("RSI<30 + BBSqueeze + Vol", s6_entry, 0.05, 0.025, 5))

# Strategy 7: Pure RSI<20 + Next Day Green confirmation
def s7_entry(ind, i):
    return (ind['rsi'].iloc[i] < 20 and
            not np.isnan(ind['rsi'].iloc[i]))
STRATEGIES.append(("RSI<20 only", s7_entry, 0.05, 0.025, 5))

# Strategy 8: RSI<30 + MA trend reversal (price above SMA10 but was below)
def s8_entry(ind, i, df):
    if i < 2:
        return False
    close = df['close']
    sma10 = ind['sma10']
    return (ind['rsi'].iloc[i] < 30 and
            close.iloc[i] > sma10.iloc[i] and
            close.iloc[i-1] <= sma10.iloc[i-1] and  # just crossed above
            not np.isnan(ind['rsi'].iloc[i]))
STRATEGIES.append(("RSI<30 + MA10CrossUp", s8_entry, 0.05, 0.025, 5))

# Strategy 9: 2 Red Days + Doji + RSI<35
def s9_entry(ind, i):
    if i < 1:
        return False
    return (ind['red_streak'].iloc[i] >= 2 and
            ind['is_doji'].iloc[i] and
            ind['rsi'].iloc[i] < 35 and
            not np.isnan(ind['rsi'].iloc[i]))
STRATEGIES.append(("2Red + Doji + RSI<35", s9_entry, 0.05, 0.025, 5))

# Strategy 10: ATR expansion + RSI<30 (volatility signal)
def s10_entry(ind, i):
    return (ind['rsi'].iloc[i] < 30 and
            ind['atr_pct'].iloc[i] > ind['atr_pct'].rolling(20).mean().iloc[i] * 1.5 and
            ind['pct'].iloc[i] < 0 and
            not np.isnan(ind['rsi'].iloc[i]))
STRATEGIES.append(("RSI<30 + ATRexpand + Red", s10_entry, 0.05, 0.025, 5))

# ── Run all strategies ──────────────────────────────────
def backtest_strategy(all_data, entry_fn, tp_pct, sl_pct, max_hold):
    """Run backtest for one strategy across all coins."""
    all_signals = []
    
    for coin, data in all_data.items():
        if coin not in valid_coins:
            continue
        
        df = to_df(data)
        if len(df) < 40:
            continue
        
        ind = compute_indicators(df)
        n = len(df)
        
        for i in range(30, n - 1):  # -1 for confirmation candle
            try:
                ok = entry_fn(ind, i, df)
            except TypeError:
                ok = entry_fn(ind, i)
            if ok:
                # Next candle confirmation: must be green
                if df['close'].iloc[i+1] <= df['open'].iloc[i+1]:
                    continue
                
                entry_idx = i + 1
                entry_price = df['close'].iloc[entry_idx]
                entry_date = df.index[entry_idx]
                
                all_signals.append({
                    'coin': coin,
                    'idx': entry_idx,
                    'entry_price': entry_price,
                    'date': entry_date,
                })
    
    all_signals.sort(key=lambda s: s['date'])
    
    # Simulate trades
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
                exit_price = sl_price
                exit_type = 'SL'
                exit_idx = j
                break
            elif high_arr[j] >= tp_price:
                exit_price = tp_price
                exit_type = 'TP'
                exit_idx = j
                break
        
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
            'pnl_pct': pnl_pct,
            'pnl_usd': pnl_usd,
            'type': exit_type,
            'capital': capital,
        })
        
        active[coin] = exit_idx
        active = {k: v for k, v in active.items() if v > entry_idx}
    
    return trades, capital

def analyze_strategy(name, trades, final_cap):
    """Return summary dict."""
    if not trades:
        return {'name': name, 'trades': 0, 'wr': 0, 'return': 0, 'max_dd': 0, 'pf': 0, 'avg_trade': 0}
    
    df = pd.DataFrame(trades)
    wins = df[df['pnl_pct'] > 0]
    losses = df[df['pnl_pct'] <= 0]
    wr = len(wins) / len(df) * 100
    
    eq = [INITIAL_CAPITAL] + [t['capital'] for t in trades]
    eq = np.array(eq)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100
    max_dd = dd.min()
    
    total_ret = (final_cap / INITIAL_CAPITAL - 1) * 100
    
    pf = abs(wins['pnl_usd'].sum() / losses['pnl_usd'].sum()) if len(losses) > 0 and losses['pnl_usd'].sum() != 0 else float('inf')
    
    return {
        'name': name,
        'trades': len(df),
        'wr': round(wr, 1),
        'return': round(total_ret, 1),
        'max_dd': round(max_dd, 2),
        'pf': round(pf, 2),
        'avg_trade': round(df['pnl_pct'].mean(), 2),
        'wins': len(wins),
        'losses': len(losses),
        'avg_win': round(wins['pnl_pct'].mean(), 2) if len(wins) > 0 else 0,
        'avg_loss': round(losses['pnl_pct'].mean(), 2) if len(losses) > 0 else 0,
    }

# ── Run all ─────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"🏆 NON-WHALE STRATEGY COMPARISON")
print(f"{'='*80}")
print(f"{'Strategy':<35s} {'Trades':>6s} {'WR':>6s} {'Return':>8s} {'MaxDD':>7s} {'PF':>6s} {'Avg':>7s}")
print(f"{'-'*80}")

results = []
for name, entry_fn, tp, sl, mh in STRATEGIES:
    trades, final_cap = backtest_strategy(all_data, entry_fn, tp, sl, mh)
    r = analyze_strategy(name, trades, final_cap)
    results.append(r)
    
    ret_str = f"+{r['return']}%" if r['return'] > 0 else f"{r['return']}%"
    print(f"{name:<35s} {r['trades']:>6d} {r['wr']:>5.1f}% {ret_str:>8s} {r['max_dd']:>6.2f}% {r['pf']:>5.2f} {r['avg_trade']:>+6.2f}%")

# Sort by best return
results.sort(key=lambda r: -r['return'])

print(f"\n{'='*80}")
print(f"🥇 TOP 3 STRATEGIES (by return)")
print(f"{'='*80}")
for i, r in enumerate(results[:3]):
    print(f"\n  #{i+1}: {r['name']}")
    print(f"      Trades: {r['trades']} | WR: {r['wr']}% | Return: {r['return']:+.1f}%")
    print(f"      MaxDD: {r['max_dd']}% | PF: {r['pf']} | AvgWin: +{r['avg_win']}% | AvgLoss: {r['avg_loss']}%")

# Best by WR
results_by_wr = sorted(results, key=lambda r: -r['wr'])
print(f"\n{'='*80}")
print(f"🎯 TOP 3 STRATEGIES (by Win Rate)")
print(f"{'='*80}")
for i, r in enumerate(results_by_wr[:3]):
    print(f"\n  #{i+1}: {r['name']}")
    print(f"      Trades: {r['trades']} | WR: {r['wr']}% | Return: {r['return']:+.1f}%")
    print(f"      MaxDD: {r['max_dd']}% | PF: {r['pf']} | AvgWin: +{r['avg_win']}% | AvgLoss: {r['avg_loss']}%")

print(f"\n✅ Done! All strategies tested without whale indicators.")
