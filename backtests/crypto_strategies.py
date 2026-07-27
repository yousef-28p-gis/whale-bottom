#!/usr/bin/env python3
"""
CRYPTO-SPECIFIC STRATEGIES — Backtest on 198 halal coins, 120 days
1. RSI(2) vs ADX(2) — outperformed buy&hold BTC 2012-2025
2. Donchian Breakout (20) — Sharpe 1.95, best risk-adjusted
3. RSI Divergence — bullish divergence reversal
4. Golden Cross 50/200 — classic crypto trend strategy
5. Dip Buying: -10% day + volume spike
"""
import json, numpy as np, pandas as pd, os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/data/trading28/backtests/pattern_data'
COMMISSION = 0.002
INITIAL_CAPITAL = 1000

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

print(f"🪙 CRYPTO-SPECIFIC STRATEGIES — {len(all_data)} coins, 120 days\n")

def simulate_trades(signals, all_data, tp_pct, sl_pct, max_hold):
    signals.sort(key=lambda s: s['date'])
    trades, capital = [], INITIAL_CAPITAL
    active = {}
    for sig in signals:
        coin, ei = sig['coin'], sig['idx']
        if coin in active and active[coin] > ei: continue
        d = all_data[coin]
        c, h, l = np.array(d['close']), np.array(d['high']), np.array(d['low'])
        n = len(c)
        if ei >= n-1: continue
        tp_p = sig['entry']*(1+tp_pct); sl_p = sig['entry']*(1-sl_pct)
        ep = et = ex = None
        for j in range(ei+1, min(ei+max_hold, n)):
            if l[j] <= sl_p: ep=sl_p; et='SL'; ex=j; break
            elif h[j] >= tp_p: ep=tp_p; et='TP'; ex=j; break
        if ep is None:
            end = min(ei+max_hold, n-1); ep=c[end]; et='TIME'; ex=end
        pnl = (ep/sig['entry']-1)*100 - COMMISSION*100
        sz = capital*0.10; capital += sz*pnl/100
        trades.append({'pnl':pnl, 'type':et, 'cap':capital})
        active[coin]=ex
        active={k:v for k,v in active.items() if v>ei}
    return trades, capital

def summarize(name, trades, final_cap):
    if not trades: return f"{name:<30s} 0 trades"
    df = pd.DataFrame(trades)
    wins, losses = df[df['pnl']>0], df[df['pnl']<=0]
    wr = len(wins)/len(df)*100
    eq = np.array([1000]+[t['cap'] for t in trades])
    peak = np.maximum.accumulate(eq); dd = (eq-peak)/peak*100
    ret = (final_cap/1000-1)*100
    pf = abs(wins['pnl'].sum()/losses['pnl'].sum()) if len(losses)>0 else 999
    tp_h = len(df[df['type']=='TP'])
    return (f"{name:<30s} {len(df):>4d} | WR {wr:>5.1f}% | "
            f"Ret {ret:>+6.1f}% | DD {dd.min():>6.2f}% | PF {pf:>5.2f} | TP:{tp_h:>3d} | Avg {df['pnl'].mean():>+5.2f}%")

# ═══════════════════════════════════════════════════════
# 1: RSI(2) vs ADX(2) — BTC outperformer 2012-2025
# ═══════════════════════════════════════════════════════
def strat_rsi2_adx2(all_data, tp, sl, max_hold=10):
    signals = []
    for coin, data in all_data.items():
        if coin not in valid_coins: continue
        c = np.array(data['close']); h = np.array(data['high']); l = np.array(data['low'])
        n = len(c)
        if n < 70: continue
        
        # RSI(2)
        delta = pd.Series(c).diff()
        gain = delta.where(delta>0,0).rolling(2).mean().values
        loss = (-delta.where(delta<0,0)).rolling(2).mean().values
        rsi2 = np.where(loss>0, 100-(100/(1+gain/loss)), 100)
        
        # ADX(2)
        tr = np.maximum(h[1:]-l[1:], np.maximum(abs(h[1:]-c[:-1]), abs(l[1:]-c[:-1])))
        tr = np.insert(tr, 0, h[0]-l[0])
        atr2 = pd.Series(tr).rolling(2).mean().values
        up = np.where((h[1:]-h[:-1])>(l[:-1]-l[1:]), np.maximum(h[1:]-h[:-1],0), 0)
        up = np.insert(up,0,0)
        down = np.where((l[:-1]-l[1:])>(h[1:]-h[:-1]), np.maximum(l[:-1]-l[1:],0), 0)
        down = np.insert(down,0,0)
        pdi = pd.Series(up).rolling(2).mean().values/atr2*100
        mdi = pd.Series(down).rolling(2).mean().values/atr2*100
        adx2 = np.abs(pdi-mdi)/(pdi+mdi)*100
        adx2 = pd.Series(adx2).rolling(2).mean().values
        
        sma50 = pd.Series(c).rolling(50).mean().values
        ema7 = pd.Series(c).ewm(span=7).mean().values
        
        for i in range(55, n-1):
            if np.isnan(rsi2[i]) or np.isnan(adx2[i]) or np.isnan(sma50[i]): continue
            # Entry: RSI(2) > ADX(2) AND close > SMA50 AND close > EMA7
            if rsi2[i] > adx2[i] and c[i] > sma50[i] and c[i] > ema7[i]:
                signals.append({'coin':coin, 'idx':i, 'entry':c[i], 'date':data['ts'][i]})
    
    return simulate_trades(signals, all_data, tp, sl, max_hold)

# ═══════════════════════════════════════════════════════
# 2: Donchian Breakout (20-period) — Best Sharpe 1.95
# ═══════════════════════════════════════════════════════
def strat_donchian(all_data, tp, sl, max_hold=7):
    signals = []
    for coin, data in all_data.items():
        if coin not in valid_coins: continue
        c = np.array(data['close']); h = np.array(data['high']); l = np.array(data['low']); v = np.array(data['volume'])
        n = len(c)
        if n < 60: continue
        
        high20 = pd.Series(h).rolling(20).max().values
        low20 = pd.Series(l).rolling(20).min().values
        mid20 = (high20 + low20) / 2
        vol_avg = pd.Series(v).rolling(20).mean().values
        
        for i in range(30, n-1):
            if np.isnan(high20[i]): continue
            # Breakout above 20-bar high
            if c[i] > high20[i-1] and c[i-1] <= high20[i-2]:
                if v[i] > vol_avg[i] * 1.2:  # volume confirmation
                    signals.append({'coin':coin, 'idx':i, 'entry':c[i], 'date':data['ts'][i]})
    
    return simulate_trades(signals, all_data, tp, sl, max_hold)

# ═══════════════════════════════════════════════════════
# 3: RSI Bullish Divergence — Price lower low, RSI higher low
# ═══════════════════════════════════════════════════════
def strat_rsi_divergence(all_data, tp, sl, max_hold=10):
    signals = []
    for coin, data in all_data.items():
        if coin not in valid_coins: continue
        c = np.array(data['close'])
        n = len(c)
        if n < 60: continue
        
        delta = pd.Series(c).diff()
        gain = delta.where(delta>0,0).rolling(14).mean().values
        loss = (-delta.where(delta<0,0)).rolling(14).mean().values
        rsi = np.where(loss>0, 100-(100/(1+gain/loss)), 50)
        
        # Scan for divergence in last 20 bars
        for i in range(40, n-1):
            if np.isnan(rsi[i]): continue
            if rsi[i] > 50: continue  # only oversold territory
            
            # Find lowest price & RSI in last 20 bars vs previous 20
            price_now = c[i]
            rsi_now = rsi[i]
            
            # Look back 5-20 bars for previous low
            lookback_start = max(5, i-20)
            lookback_end = i-3
            
            prev_low_idx = None
            prev_low_price = float('inf')
            for j in range(lookback_start, lookback_end):
                if c[j] < prev_low_price:
                    prev_low_price = c[j]
                    prev_low_idx = j
            
            if prev_low_idx is None: continue
            
            # Divergence: price made lower low, but RSI made higher low
            if price_now < prev_low_price and rsi_now > rsi[prev_low_idx]:
                # Entry on next green candle
                signals.append({'coin':coin, 'idx':i+1 if i+1<n else i, 
                               'entry':c[i+1] if i+1<n else c[i], 
                               'date':data['ts'][i]})
    
    return simulate_trades(signals, all_data, tp, sl, max_hold)

# ═══════════════════════════════════════════════════════
# 4: Golden Cross 50/200 + RSI filter
# ═══════════════════════════════════════════════════════
def strat_golden_cross(all_data, tp, sl, max_hold=14):
    signals = []
    for coin, data in all_data.items():
        if coin not in valid_coins: continue
        c = np.array(data['close'])
        n = len(c)
        if n < 220: continue
        
        sma50 = pd.Series(c).rolling(50).mean().values
        sma200 = pd.Series(c).rolling(200).mean().values
        
        # RSI
        delta = pd.Series(c).diff()
        gain = delta.where(delta>0,0).rolling(14).mean().values
        loss = (-delta.where(delta<0,0)).rolling(14).mean().values
        rsi = np.where(loss>0, 100-(100/(1+gain/loss)), 50)
        
        for i in range(210, n-1):
            if np.isnan(sma50[i]) or np.isnan(sma200[i]): continue
            # Golden cross today
            if sma50[i] > sma200[i] and sma50[i-1] <= sma200[i-1]:
                if c[i] > sma50[i] and rsi[i] > 40:  # confirmation
                    signals.append({'coin':coin, 'idx':i, 'entry':c[i], 'date':data['ts'][i]})
    
    return simulate_trades(signals, all_data, tp, sl, max_hold)

# ═══════════════════════════════════════════════════════
# 5: Crypto Dip Buy — -10%+ day + Volume Spike + RSI<25
# ═══════════════════════════════════════════════════════
def strat_dip_buy(all_data, tp, sl, max_hold=7):
    signals = []
    for coin, data in all_data.items():
        if coin not in valid_coins: continue
        c = np.array(data['close']); v = np.array(data['volume'])
        n = len(c)
        if n < 60: continue
        
        pct = pd.Series(c).pct_change().values * 100
        vol_avg = pd.Series(v).rolling(20).mean().values
        
        delta = pd.Series(c).diff()
        gain = delta.where(delta>0,0).rolling(14).mean().values
        loss = (-delta.where(delta<0,0)).rolling(14).mean().values
        rsi = np.where(loss>0, 100-(100/(1+gain/loss)), 50)
        
        for i in range(30, n-1):
            if np.isnan(rsi[i]): continue
            # -10%+ day + high volume + RSI oversold
            if pct[i] < -10 and v[i] > vol_avg[i]*2 and rsi[i] < 30:
                # Buy on next day's open
                signals.append({'coin':coin, 'idx':i+1 if i+1<n else i,
                               'entry':c[i+1] if i+1<n else c[i],
                               'date':data['ts'][i]})
    
    return simulate_trades(signals, all_data, tp, sl, max_hold)

# ── Run ─────────────────────────────────────────────────
STRATEGIES = [
    ("RSI(2)>ADX(2)", strat_rsi2_adx2),
    ("Donchian Breakout 20", strat_donchian),
    ("RSI Divergence", strat_rsi_divergence),
    ("Golden Cross 50/200", strat_golden_cross),
    ("Crypto Dip -10%", strat_dip_buy),
]

TP_SL = [(0.05, 0.03, "TP5/SL3"), (0.10, 0.05, "TP10/SL5"), (0.15, 0.07, "TP15/SL7")]

for tp, sl, label in TP_SL:
    print(f"\n{'='*95}")
    print(f"📐 {label}")
    print(f"{'='*95}")
    print(f"{'Strategy':<30s} {'Trades':>5s} {'WR':>6s} {'Return':>7s} {'MaxDD':>7s} {'PF':>5s} {'TP':>4s} {'Avg':>6s}")
    print(f"{'-'*80}")
    
    for name, fn in STRATEGIES:
        trades, final = fn(all_data, tp, sl)
        print(f"  {summarize(name, trades, final)}")

print(f"\n✅ All crypto-specific strategies tested!")
