#!/usr/bin/env python3
"""
PROVEN YOUTUBE STRATEGIES — Backtest on 198 halal coins, 120 days
1. EMA 12/50 Crossover
2. EMA Breakout + ADX (NORN WEAVE style) 
3. Supertrend + EMA 200
4. EMA 9/21 + RSI Filter
5. Bollinger Squeeze Breakout
"""
import json, numpy as np, pandas as pd, os, time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/data/trading28/backtests/pattern_data'
COMMISSION = 0.002
INITIAL_CAPITAL = 1000

# ── Load 120d data ──
cache_file = os.path.join(DATA_DIR, 'daily_120d.json')
with open(cache_file) as f:
    all_data = json.load(f)

with open('/data/trading28/config/shariah_coins.json') as f:
    config = json.load(f)
coins_raw = config['halal'] + config['halal2']
seen = set()
coins_raw = [c for c in coins_raw if not (c in seen or seen.add(c))]
blacklist = {'QTUM','ZRO','IOTX','DYM','DGB','SAPIEN','XLM','EDU','BTC','INIT','PARTI','ROBO','PYTH','ANKR'}
valid_coins = set(c for c in coins_raw if c not in blacklist)

print(f"🎯 YOUTUBE PROVEN STRATEGIES — {len(all_data)} coins, 120 days\n")

def to_df(data):
    return pd.DataFrame({
        'open': data['open'], 'high': data['high'],
        'low': data['low'], 'close': data['close'], 'volume': data['volume'],
    }, index=pd.to_datetime([datetime.fromtimestamp(t/1000) for t in data['ts']]))

# ═══════════════════════════════════════════════════════
# STRATEGY 1: EMA 12/50 Crossover
# ═══════════════════════════════════════════════════════
# Buy: EMA12 crosses above EMA50 AND price > EMA12
# Sell: EMA12 crosses below EMA50
def strat_ema1250(all_data, tp_pct, sl_pct, max_hold=10):
    signals = []
    for coin, data in all_data.items():
        if coin not in valid_coins: continue
        close = np.array(data['close'])
        n = len(close)
        if n < 70: continue
        
        ema12 = pd.Series(close).ewm(span=12).mean().values
        ema50 = pd.Series(close).ewm(span=50).mean().values
        
        for i in range(55, n - 1):
            # Bullish cross
            if ema12[i] > ema50[i] and ema12[i-1] <= ema50[i-1]:
                if close[i] > ema12[i]:  # price above EMA12
                    signals.append({'coin': coin, 'idx': i, 'entry': close[i],
                                   'date': data['ts'][i]})
    
    return simulate_trades(signals, all_data, tp_pct, sl_pct, max_hold)

# ═══════════════════════════════════════════════════════
# STRATEGY 2: EMA Breakout + ADX (NORN WEAVE inspired)
# ═══════════════════════════════════════════════════════
# Entry: EMA20 rising + ADX>25 + Price breaks 5-bar high
def strat_ema_adx(all_data, tp_pct, sl_pct, max_hold=7):
    signals = []
    for coin, data in all_data.items():
        if coin not in valid_coins: continue
        close = np.array(data['close']); high = np.array(data['high'])
        low = np.array(data['low'])
        n = len(close)
        if n < 70: continue
        
        ema20 = pd.Series(close).ewm(span=20).mean().values
        
        # ADX(14)
        tr = np.maximum(high[1:] - low[1:], 
                        np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
        tr = np.insert(tr, 0, 0)
        atr = pd.Series(tr).rolling(14).mean().values
        
        up_move = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        up_move = np.insert(up_move, 0, 0)
        down_move = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]),
                             np.maximum(low[:-1] - low[1:], 0), 0)
        down_move = np.insert(down_move, 0, 0)
        
        plus_di = pd.Series(up_move).rolling(14).mean().values / atr * 100
        minus_di = pd.Series(down_move).rolling(14).mean().values / atr * 100
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = pd.Series(dx).rolling(14).mean().values
        
        # 5-bar high
        high5 = pd.Series(high).rolling(5).max().values
        
        for i in range(55, n - 1):
            if np.isnan(adx[i]): continue
            if adx[i] < 25: continue  # need momentum
            if ema20[i] <= ema20[i-3]: continue  # EMA must be rising
            if close[i] <= high5[i-1]: continue  # must break 5-bar high
            
            signals.append({'coin': coin, 'idx': i, 'entry': close[i],
                           'date': data['ts'][i]})
    
    return simulate_trades(signals, all_data, tp_pct, sl_pct, max_hold)

# ═══════════════════════════════════════════════════════
# STRATEGY 3: Supertrend + EMA200
# ═══════════════════════════════════════════════════════
# Supertrend(10,3) flips bullish + price > EMA200
def strat_supertrend(all_data, tp_pct, sl_pct, max_hold=7):
    signals = []
    for coin, data in all_data.items():
        if coin not in valid_coins: continue
        close = np.array(data['close']); high = np.array(data['high'])
        low = np.array(data['low'])
        n = len(close)
        if n < 220: continue  # need EMA200
        
        ema200 = pd.Series(close).ewm(span=200).mean().values
        
        # SuperTrend(10, 3)
        atr_len = 10; multiplier = 3
        tr = np.maximum(high[1:] - low[1:],
                       np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
        tr = np.insert(tr, 0, high[0] - low[0])
        atr_st = pd.Series(tr).rolling(atr_len).mean().values
        
        # Basic supertrend
        hl2 = (high + low) / 2
        upper = hl2 + multiplier * atr_st
        lower = hl2 - multiplier * atr_st
        
        st_trend = np.ones(n)  # 1=up, -1=down
        for i in range(1, n):
            if close[i] > upper[i-1]:
                st_trend[i] = 1
            elif close[i] < lower[i-1]:
                st_trend[i] = -1
            else:
                st_trend[i] = st_trend[i-1]
        
        for i in range(220, n - 1):
            if np.isnan(ema200[i]): continue
            # Supertrend flips bullish today
            if st_trend[i] == 1 and st_trend[i-1] == -1:
                if close[i] > ema200[i]:  # must be above EMA200
                    signals.append({'coin': coin, 'idx': i, 'entry': close[i],
                                   'date': data['ts'][i]})
    
    return simulate_trades(signals, all_data, tp_pct, sl_pct, max_hold)

# ═══════════════════════════════════════════════════════
# STRATEGY 4: EMA 9/21 + RSI Filter
# ═══════════════════════════════════════════════════════
# Buy: EMA9 > EMA21 + RSI>50 (bullish momentum)
def strat_ema921_rsi(all_data, tp_pct, sl_pct, max_hold=7):
    signals = []
    for coin, data in all_data.items():
        if coin not in valid_coins: continue
        close = np.array(data['close'])
        n = len(close)
        if n < 60: continue
        
        ema9 = pd.Series(close).ewm(span=9).mean().values
        ema21 = pd.Series(close).ewm(span=21).mean().values
        
        # RSI
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean().values
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().values
        rsi = 100 - (100 / (1 + gain / loss))
        
        for i in range(30, n - 1):
            if np.isnan(rsi[i]): continue
            # EMA9 crosses above EMA21 OR already above + RSI bullish
            cross_up = (ema9[i] > ema21[i] and ema9[i-1] <= ema21[i-1])
            if cross_up and rsi[i] > 45 and rsi[i] < 70:
                signals.append({'coin': coin, 'idx': i, 'entry': close[i],
                               'date': data['ts'][i]})
    
    return simulate_trades(signals, all_data, tp_pct, sl_pct, max_hold)

# ═══════════════════════════════════════════════════════
# STRATEGY 5: Bollinger Squeeze + Breakout
# ═══════════════════════════════════════════════════════
# BB(20,2) width at 6-month low + price breaks upper band + volume>1.5x
def strat_bb_squeeze(all_data, tp_pct, sl_pct, max_hold=7):
    signals = []
    for coin, data in all_data.items():
        if coin not in valid_coins: continue
        close = np.array(data['close']); volume = np.array(data['volume'])
        n = len(close)
        if n < 130: continue  # need 6-month lookback
        
        sma20 = pd.Series(close).rolling(20).mean().values
        std20 = pd.Series(close).rolling(20).std().values
        bb_upper = sma20 + 2 * std20
        bb_width = 4 * std20 / sma20  # (upper - lower) / sma20
        
        # 6-month min width (120 bars)
        bb_width_min = pd.Series(bb_width).rolling(120).min().values
        vol_avg = pd.Series(volume).rolling(20).mean().values
        
        for i in range(130, n - 1):
            if np.isnan(bb_width[i]): continue
            # Squeeze: width near 6-month low
            is_squeeze = bb_width[i] <= bb_width_min[i] * 1.1
            # Breakout: price above upper band
            if is_squeeze and close[i] > bb_upper[i]:
                if volume[i] > vol_avg[i] * 1.3:  # volume confirmation
                    signals.append({'coin': coin, 'idx': i, 'entry': close[i],
                                   'date': data['ts'][i]})
    
    return simulate_trades(signals, all_data, tp_pct, sl_pct, max_hold)

# ═══════════════════════════════════════════════════════
# Trade Simulator (close-only)
# ═══════════════════════════════════════════════════════
def simulate_trades(signals, all_data, tp_pct, sl_pct, max_hold):
    signals.sort(key=lambda s: s['date'])
    trades = []
    capital = INITIAL_CAPITAL
    active = {}
    
    for sig in signals:
        coin = sig['coin']; ei = sig['idx']
        if coin in active and active[coin] > ei: continue
        
        data = all_data[coin]
        c = np.array(data['close']); h = np.array(data['high'])
        l = np.array(data['low'])
        n = len(c)
        if ei >= n - 1: continue
        
        tp_p = sig['entry'] * (1 + tp_pct)
        sl_p = sig['entry'] * (1 - sl_pct)
        
        ep = None; et = None; ex = None
        for j in range(ei + 1, min(ei + max_hold, n)):
            if l[j] <= sl_p: ep = sl_p; et = 'SL'; ex = j; break
            elif h[j] >= tp_p: ep = tp_p; et = 'TP'; ex = j; break
        
        if ep is None:
            end = min(ei + max_hold, n - 1)
            ep = c[end]; et = 'TIME'; ex = end
        
        pnl = (ep / sig['entry'] - 1) * 100 - COMMISSION * 100
        sz = capital * 0.10
        capital += sz * pnl / 100
        trades.append({'pnl': pnl, 'type': et, 'cap': capital})
        active[coin] = ex
        active = {k: v for k, v in active.items() if v > ei}
    
    return trades, capital

def summarize(name, trades, final_cap):
    if not trades:
        return f"{name:<25s} 0 trades"
    df = pd.DataFrame(trades)
    wins = df[df['pnl'] > 0]; losses = df[df['pnl'] <= 0]
    wr = len(wins) / len(df) * 100
    eq = np.array([1000] + [t['cap'] for t in trades])
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100
    ret = (final_cap / 1000 - 1) * 100
    pf = abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 else 999
    tp_h = len(df[df['type'] == 'TP'])
    return (f"{name:<25s} {len(df):>4d} trades | WR {wr:>5.1f}% | "
            f"Return {ret:>+6.1f}% | DD {dd.min():>6.2f}% | PF {pf:.2f} | TP:{tp_h:>3d} | Avg {df['pnl'].mean():>+5.2f}%")

# ── Run all strategies with multiple TP/SL combos ──
STRATEGIES = [
    ("EMA 12/50 Crossover", strat_ema1250),
    ("EMA Breakout + ADX", strat_ema_adx),
    ("Supertrend + EMA200", strat_supertrend),
    ("EMA 9/21 + RSI", strat_ema921_rsi),
    ("BB Squeeze Breakout", strat_bb_squeeze),
]

TP_SL_COMBOS = [
    (0.05, 0.03, "TP5/SL3"),
    (0.10, 0.05, "TP10/SL5"),
    (0.15, 0.07, "TP15/SL7"),
]

print(f"{'='*95}")
print(f"{'Strategy':<25s} {'Trades':>5s} {'WR':>6s} {'Return':>7s} {'MaxDD':>7s} {'PF':>5s} {'TP':>4s} {'Avg':>6s}")
print(f"{'='*95}")

for tp, sl, label in TP_SL_COMBOS:
    print(f"\n── {label} ──")
    for name, fn in STRATEGIES:
        trades, final = fn(all_data, tp, sl)
        print(f"  {summarize(name, trades, final)}")

print(f"\n✅ All YouTube strategies tested!")
