#!/usr/bin/env python3
"""
FRESH STRATEGIES + WHALE BOTTOM — 120-day shootout
1. Support Bounce (20d low + reversal candle + RSI)
2. Keltner Channel Bounce
3. Parabolic SAR Reversal
4. Heikin-Ashi Reversal
5. Whale Bottom (exact rules from live strategy)
"""
import json, numpy as np, pandas as pd, os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/data/trading28/backtests/pattern_data'
COMMISSION = 0.002; INITIAL_CAPITAL = 1000

with open(f'{DATA_DIR}/daily_120d.json') as f:
    all_data = json.load(f)

with open('/data/trading28/config/shariah_coins.json') as f:
    config = json.load(f)
coins_raw = config['halal'] + config['halal2']
seen = set()
coins_raw = [c for c in coins_raw if not (c in seen or seen.add(c))]
blacklist = {'QTUM','ZRO','IOTX','DYM','DGB','SAPIEN','XLM','EDU','BTC','INIT','PARTI','ROBO','PYTH','ANKR'}
valid_coins = set(c for c in coins_raw if c not in blacklist)

print(f"🎯 FRESH STRATEGIES + WHALE — {len(all_data)} coins, 120 days\n")

def simulate(signals, all_data, tp, sl, max_hold):
    signals.sort(key=lambda s: s['date'])
    trades, cap = [], INITIAL_CAPITAL
    active = {}
    for sig in signals:
        coin, ei = sig['coin'], sig['idx']
        if coin in active and active[coin] > ei: continue
        d = all_data[coin]; c, h, l = np.array(d['close']), np.array(d['high']), np.array(d['low'])
        n = len(c)
        if ei >= n-1: continue
        tp_p = sig['entry']*(1+tp); sl_p = sig['entry']*(1-sl)
        ep = et = ex = None
        for j in range(ei+1, min(ei+max_hold, n)):
            if l[j] <= sl_p: ep=sl_p; et='SL'; ex=j; break
            elif h[j] >= tp_p: ep=tp_p; et='TP'; ex=j; break
        if ep is None:
            end = min(ei+max_hold, n-1); ep=c[end]; et='TIME'; ex=end
        pnl = (ep/sig['entry']-1)*100 - COMMISSION*100
        sz = cap*0.10; cap += sz*pnl/100
        trades.append({'pnl':pnl, 'type':et, 'cap':cap})
        active[coin]=ex
        active={k:v for k,v in active.items() if v>ei}
    return trades, cap

def summarize(name, trades, final_cap):
    if not trades: return f"{name:<30s} 0 trades"
    df = pd.DataFrame(trades)
    wins, losses = df[df['pnl']>0], df[df['pnl']<=0]
    wr = len(wins)/len(df)*100
    eq = np.array([1000]+[t['cap'] for t in trades])
    dd = (eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq)*100
    ret = (final_cap/1000-1)*100
    pf = abs(wins['pnl'].sum()/losses['pnl'].sum()) if len(losses)>0 else 999
    tp_h = len(df[df['type']=='TP'])
    return (f"{name:<30s} {len(df):>4d} | WR {wr:>5.1f}% | "
            f"Ret {ret:>+6.1f}% | DD {dd.min():>6.2f}% | PF {pf:.2f} | TP:{tp_h:>3d} | Avg {df['pnl'].mean():>+5.2f}%")

# ═══════════════════════════════════════════════════════
# 1: SUPPORT BOUNCE — 20d low + reversal + RSI
# ═══════════════════════════════════════════════════════
def strat_support_bounce(all_data, tp, sl, max_hold=7):
    signals = []
    for coin, data in all_data.items():
        if coin not in valid_coins: continue
        c = np.array(data['close']); h = np.array(data['high'])
        l = np.array(data['low']); o = np.array(data['open']); n = len(c)
        if n < 60: continue
        
        low20 = pd.Series(l).rolling(20).min().values
        
        delta = pd.Series(c).diff()
        gain = delta.where(delta>0,0).rolling(14).mean().values
        loss = (-delta.where(delta<0,0)).rolling(14).mean().values
        rsi = np.where(loss>0, 100-(100/(1+gain/loss)), 50)
        
        for i in range(30, n-1):
            if np.isnan(rsi[i]): continue
            # Touch 20d low zone (within 2%)
            near_support = abs(c[i] - low20[i]) / low20[i] < 0.02
            # Reversal candle: today green and yesterday red
            reversal = c[i] > o[i] and c[i-1] < o[i-1]
            # RSI oversold
            oversold = rsi[i] < 35
            
            if near_support and reversal and oversold:
                signals.append({'coin':coin, 'idx':i, 'entry':c[i], 'date':data['ts'][i]})
    return simulate(signals, all_data, tp, sl, max_hold)

# ═══════════════════════════════════════════════════════
# 2: KELTNER CHANNEL BOUNCE
# ═══════════════════════════════════════════════════════
def strat_keltner(all_data, tp, sl, max_hold=7):
    signals = []
    for coin, data in all_data.items():
        if coin not in valid_coins: continue
        c = np.array(data['close']); h = np.array(data['high'])
        l = np.array(data['low']); o = np.array(data['open']); n = len(c)
        if n < 60: continue
        
        ema20 = pd.Series(c).ewm(span=20).mean().values
        tr = np.maximum(h[1:]-l[1:], np.maximum(abs(h[1:]-c[:-1]), abs(l[1:]-c[:-1])))
        tr = np.insert(tr, 0, h[0]-l[0])
        atr10 = pd.Series(tr).rolling(10).mean().values
        
        kc_upper = ema20 + 2*atr10
        kc_lower = ema20 - 2*atr10
        
        delta = pd.Series(c).diff()
        gain = delta.where(delta>0,0).rolling(14).mean().values
        loss = (-delta.where(delta<0,0)).rolling(14).mean().values
        rsi = np.where(loss>0, 100-(100/(1+gain/loss)), 50)
        
        for i in range(30, n-1):
            if np.isnan(rsi[i]): continue
            # Below lower Keltner band
            below_lower = c[i] < kc_lower[i]
            # Reversal: green candle
            green = c[i] > o[i]
            # RSI < 30
            if below_lower and green and rsi[i] < 30:
                signals.append({'coin':coin, 'idx':i, 'entry':c[i], 'date':data['ts'][i]})
    return simulate(signals, all_data, tp, sl, max_hold)

# ═══════════════════════════════════════════════════════
# 3: PARABOLIC SAR REVERSAL
# ═══════════════════════════════════════════════════════
def strat_psar(all_data, tp, sl, max_hold=7):
    signals = []
    for coin, data in all_data.items():
        if coin not in valid_coins: continue
        c = np.array(data['close']); h = np.array(data['high'])
        l = np.array(data['low']); n = len(c)
        if n < 60: continue
        
        # Basic PSAR
        af = 0.02; af_max = 0.20
        psar = np.zeros(n)
        trend = np.zeros(n)  # 1=up, -1=down
        ep = np.zeros(n)  # extreme point
        
        # Init
        trend[0] = 1; psar[0] = l[0]; ep[0] = h[0]
        
        for i in range(1, n):
            if trend[i-1] == 1:  # uptrend
                psar[i] = psar[i-1] + af * (ep[i-1] - psar[i-1])
                psar[i] = min(psar[i], l[i-1], l[i-2] if i>=2 else l[i-1])
                if l[i] < psar[i]:
                    trend[i] = -1; psar[i] = ep[i-1]; ep[i] = l[i]; af = 0.02
                else:
                    trend[i] = 1
                    if h[i] > ep[i-1]:
                        ep[i] = h[i]; af = min(af+0.02, af_max)
                    else:
                        ep[i] = ep[i-1]
            else:  # downtrend
                psar[i] = psar[i-1] - af * (psar[i-1] - ep[i-1])
                psar[i] = max(psar[i], h[i-1], h[i-2] if i>=2 else h[i-1])
                if h[i] > psar[i]:
                    trend[i] = 1; psar[i] = ep[i-1]; ep[i] = h[i]; af = 0.02
                else:
                    trend[i] = -1
                    if l[i] < ep[i-1]:
                        ep[i] = l[i]; af = min(af+0.02, af_max)
                    else:
                        ep[i] = ep[i-1]
        
        delta = pd.Series(c).diff()
        gain = delta.where(delta>0,0).rolling(14).mean().values
        loss = (-delta.where(delta<0,0)).rolling(14).mean().values
        rsi = np.where(loss>0, 100-(100/(1+gain/loss)), 50)
        
        for i in range(30, n-1):
            if np.isnan(rsi[i]): continue
            # PSAR flips from below to above (downtrend → uptrend)
            flip = trend[i] == 1 and trend[i-1] == -1
            if flip and rsi[i] < 45:
                signals.append({'coin':coin, 'idx':i, 'entry':c[i], 'date':data['ts'][i]})
    return simulate(signals, all_data, tp, sl, max_hold)

# ═══════════════════════════════════════════════════════
# 4: HEIKIN-ASHI REVERSAL
# ═══════════════════════════════════════════════════════
def strat_heikin_ashi(all_data, tp, sl, max_hold=7):
    signals = []
    for coin, data in all_data.items():
        if coin not in valid_coins: continue
        c = np.array(data['close']); h = np.array(data['high'])
        l = np.array(data['low']); o = np.array(data['open']); n = len(c)
        if n < 60: continue
        
        # Heikin-Ashi
        ha_close = (o + h + l + c) / 4
        ha_open = np.zeros(n)
        ha_high = np.zeros(n)
        ha_low = np.zeros(n)
        ha_open[0] = o[0]
        ha_high[0] = h[0]; ha_low[0] = l[0]
        
        for i in range(1, n):
            ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
            ha_high[i] = max(h[i], ha_open[i], ha_close[i])
            ha_low[i] = min(l[i], ha_open[i], ha_close[i])
        
        ha_green = ha_close > ha_open
        ha_red = ha_close < ha_open
        
        delta = pd.Series(c).diff()
        gain = delta.where(delta>0,0).rolling(14).mean().values
        loss = (-delta.where(delta<0,0)).rolling(14).mean().values
        rsi = np.where(loss>0, 100-(100/(1+gain/loss)), 50)
        
        for i in range(30, n-1):
            if np.isnan(rsi[i]): continue
            # 2+ red HA candles then green
            prev_red = ha_red[i-1] and (i<2 or ha_red[i-2])
            curr_green = ha_green[i]
            
            if prev_red and curr_green and rsi[i] < 35:
                signals.append({'coin':coin, 'idx':i, 'entry':c[i], 'date':data['ts'][i]})
    return simulate(signals, all_data, tp, sl, max_hold)

# ═══════════════════════════════════════════════════════
# 5: WHALE BOTTOM (exact rules from live strategy)
# ═══════════════════════════════════════════════════════
def strat_whale_bottom(all_data, tp_pct, sl_pct, max_hold_hours=144):  # 6 days = 144h equivalent
    """Exact whale_bottom rules: whale ≥ 0.50 + RSI < 25 + green confirmation"""
    signals = []
    
    for coin, data in all_data.items():
        if coin not in valid_coins: continue
        c = np.array(data['close']); v = np.array(data['volume'])
        o_arr = np.array(data['open']); n = len(c)
        if n < 60: continue
        
        # Whale indicator: (vol - avg_vol) / avg_vol for last 2 candles
        vol_avg_50 = pd.Series(v).rolling(50).mean().values
        
        # RSI(14)
        delta = pd.Series(c).diff()
        gain = delta.where(delta>0,0).rolling(14).mean().values
        loss = (-delta.where(delta<0,0)).rolling(14).mean().values
        rsi = np.where(loss>0, 100-(100/(1+gain/loss)), 50)
        
        for i in range(55, n-2):
            if np.isnan(rsi[i]) or np.isnan(vol_avg_50[i]):
                continue
            
            # Whale val: how much volume spike relative to average
            whale_val = (v[i] - vol_avg_50[i]) / vol_avg_50[i] if vol_avg_50[i] > 0 else 0
            
            # Entry conditions (matching live strategy)
            if whale_val < 0.50: continue  # whale ≥ 0.50
            if rsi[i] >= 25: continue  # RSI < 25
            
            # Green confirmation candle (next candle close > open)
            if c[i+1] <= o_arr[i+1]: continue
            
            # Entry at close of confirmation candle
            entry_idx = i + 1
            signals.append({'coin': coin, 'idx': entry_idx, 'entry': c[entry_idx],
                           'date': data['ts'][entry_idx]})
    
    # Convert max_hold to days for daily data
    max_hold_days = max(1, max_hold_hours // 24)
    return simulate(signals, all_data, tp_pct, sl_pct, max_hold_days)

# ═══════════════════════════════════════════════════════
# 6: VOLUME CLIMAX REVERSAL (capitulation)
# ═══════════════════════════════════════════════════════
def strat_vol_climax(all_data, tp, sl, max_hold=7):
    """Highest volume in 50 days + red candle + RSI<30 → capitulation buy"""
    signals = []
    for coin, data in all_data.items():
        if coin not in valid_coins: continue
        c = np.array(data['close']); v = np.array(data['volume'])
        o_arr = np.array(data['open']); n = len(c)
        if n < 70: continue
        
        vol_max50 = pd.Series(v).rolling(50).max().values
        
        delta = pd.Series(c).diff()
        gain = delta.where(delta>0,0).rolling(14).mean().values
        loss = (-delta.where(delta<0,0)).rolling(14).mean().values
        rsi = np.where(loss>0, 100-(100/(1+gain/loss)), 50)
        
        for i in range(55, n-1):
            if np.isnan(rsi[i]): continue
            # Highest volume in 50 days
            if v[i] < vol_max50[i] * 0.95: continue  # near record volume
            # Red candle
            if c[i] >= o_arr[i]: continue
            # RSI oversold
            if rsi[i] >= 30: continue
            
            # Buy next day
            signals.append({'coin':coin, 'idx':i+1, 'entry':c[i+1] if i+1<n else c[i],
                           'date':data['ts'][i]})
    return simulate(signals, all_data, tp, sl, max_hold)

# ═══════════════════════════════════════════════════════
# 7: DUAL TIMEFRAME MOMENTUM (daily + weekly equivalent)
# ═══════════════════════════════════════════════════════
def strat_dual_momentum(all_data, tp, sl, max_hold=10):
    """Daily RSI<30 + 5-day return < -15% (weekly selloff)"""
    signals = []
    for coin, data in all_data.items():
        if coin not in valid_coins: continue
        c = np.array(data['close']); n = len(c)
        if n < 60: continue
        
        ret_5d = pd.Series(c).pct_change(5).values * 100  # 5-day return
        
        delta = pd.Series(c).diff()
        gain = delta.where(delta>0,0).rolling(14).mean().values
        loss = (-delta.where(delta<0,0)).rolling(14).mean().values
        rsi = np.where(loss>0, 100-(100/(1+gain/loss)), 50)
        
        for i in range(30, n-1):
            if np.isnan(rsi[i]) or np.isnan(ret_5d[i]): continue
            # Daily oversold + weekly selloff
            if rsi[i] < 30 and ret_5d[i] < -15:
                signals.append({'coin':coin, 'idx':i+1, 'entry':c[i+1] if i+1<n else c[i],
                               'date':data['ts'][i]})
    return simulate(signals, all_data, tp, sl, max_hold)

# ── Run All ─────────────────────────────────────────────
STRATEGIES = [
    ("🐋 Whale Bottom (live rules)", strat_whale_bottom),
    ("Support Bounce (20d)", strat_support_bounce),
    ("Keltner Channel Bounce", strat_keltner),
    ("Parabolic SAR Reversal", strat_psar),
    ("Heikin-Ashi Reversal", strat_heikin_ashi),
    ("Volume Climax Capitulation", strat_vol_climax),
    ("Dual Momentum (RSI+5d selloff)", strat_dual_momentum),
]

TP_SL = [(0.035, 0.015, "TP3.5/SL1.5"), (0.05, 0.025, "TP5/SL2.5"), (0.10, 0.05, "TP10/SL5")]

for tp, sl, label in TP_SL:
    print(f"\n{'='*95}")
    print(f"📐 {label}")
    print(f"{'='*95}")
    print(f"{'Strategy':<30s} {'Trades':>5s} {'WR':>6s} {'Return':>7s} {'MaxDD':>7s} {'PF':>5s} {'TP':>4s} {'Avg':>6s}")
    print(f"{'-'*80}")
    
    for name, fn in STRATEGIES:
        trades, final = fn(all_data, tp, sl)
        print(f"  {summarize(name, trades, final)}")

print(f"\n✅ Done!")
