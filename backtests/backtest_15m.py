#!/usr/bin/env python3
"""
15-MINUTE TIMEFRAME — 30-day backtest, all strategies + whale_bottom proper
"""
import ccxt, json, numpy as np, pandas as pd, os, time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/data/trading28/backtests/pattern_data'
os.makedirs(DATA_DIR, exist_ok=True)
COMMISSION = 0.002
INITIAL_CAPITAL = 1000
BACKTEST_DAYS = 30
TIMEFRAME = '15m'

with open('/data/trading28/config/shariah_coins.json') as f:
    config = json.load(f)
coins_raw = config['halal'] + config['halal2']
seen = set()
coins_raw = [c for c in coins_raw if not (c in seen or seen.add(c))]
blacklist = {'QTUM','ZRO','IOTX','DYM','DGB','SAPIEN','XLM','EDU','BTC','INIT','PARTI','ROBO','PYTH','ANKR'}
coins = [c for c in coins_raw if c not in blacklist]

print(f"⏱️ 15-MINUTE BACKTEST — {BACKTEST_DAYS} days, {len(coins)} coins")
print(f"   This is the REAL whale_bottom timeframe\n")

_EXCHANGE = None
def get_exchange():
    global _EXCHANGE
    if _EXCHANGE is None:
        _EXCHANGE = ccxt.binance({'timeout': 15000, 'enableRateLimit': True})
    return _EXCHANGE

# ── Fetch 15m data ─────────────────────────────────────
def fetch_15m():
    cache_file = os.path.join(DATA_DIR, '15m_30d.json')
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            data = json.load(f)
        print(f"📦 Cache: {len(data)} coins")
        return data
    
    exchange = get_exchange()
    since = exchange.parse8601((datetime.now() - timedelta(days=BACKTEST_DAYS+3)).isoformat())
    all_data = {}
    
    for i, coin in enumerate(coins):
        try:
            ohlcv = exchange.fetch_ohlcv(f"{coin}/USDT", TIMEFRAME, since=since, limit=3000)
            if len(ohlcv) >= 500:
                all_data[coin] = {
                    'ts': [int(o[0]) for o in ohlcv],
                    'open': [float(o[1]) for o in ohlcv],
                    'high': [float(o[2]) for o in ohlcv],
                    'low': [float(o[3]) for o in ohlcv],
                    'close': [float(o[4]) for o in ohlcv],
                    'volume': [float(o[5]) for o in ohlcv],
                }
            if (i+1) % 15 == 0:
                print(f"  📊 {i+1}/{len(coins)}")
        except:
            pass
        time.sleep(0.03)
    
    with open(cache_file, 'w') as f:
        json.dump(all_data, f)
    print(f"✅ {len(all_data)} coins")
    return all_data

# ── Simulate trades (15m version) ──────────────────────
def simulate(signals, all_data, tp, sl, max_hold_candles):
    """max_hold_candles in 15m candles (96 = 24h, 384 = 4 days)"""
    signals.sort(key=lambda s: s['date'])
    trades, cap = [], INITIAL_CAPITAL
    active = {}
    
    for sig in signals:
        coin, ei = sig['coin'], sig['idx']
        if coin in active and active[coin] > ei: continue
        
        d = all_data[coin]
        c, h, l = np.array(d['close']), np.array(d['high']), np.array(d['low'])
        n = len(c)
        if ei >= n-1: continue
        
        tp_p = sig['entry']*(1+tp)
        sl_p = sig['entry']*(1-sl)
        ep = et = ex = None
        
        for j in range(ei+1, min(ei+max_hold_candles, n)):
            if l[j] <= sl_p: ep=sl_p; et='SL'; ex=j; break
            elif h[j] >= tp_p: ep=tp_p; et='TP'; ex=j; break
        
        if ep is None:
            end = min(ei+max_hold_candles, n-1)
            ep=c[end]; et='TIME'; ex=end
        
        pnl = (ep/sig['entry']-1)*100 - COMMISSION*100
        sz = cap*0.10; cap += sz*pnl/100
        trades.append({'pnl':pnl, 'type':et, 'cap':cap})
        active[coin]=ex
        active={k:v for k,v in active.items() if v>ei}
    
    return trades, cap

def summarize(name, trades, final_cap):
    if not trades: return f"{name:<35s} 0 trades"
    df = pd.DataFrame(trades)
    wins, losses = df[df['pnl']>0], df[df['pnl']<=0]
    wr = len(wins)/len(df)*100
    eq = np.array([1000]+[t['cap'] for t in trades])
    dd = (eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq)*100
    ret = (final_cap/1000-1)*100
    pf = abs(wins['pnl'].sum()/losses['pnl'].sum()) if len(losses)>0 else 999
    return (f"{name:<35s} {len(df):>4d} | WR {wr:>5.1f}% | "
            f"Ret {ret:>+6.1f}% | DD {dd.min():>6.2f}% | PF {pf:.2f} | "
            f"TP:{len(df[df['type']=='TP']):>3d} | Avg {df['pnl'].mean():>+5.2f}%")

# ═══════════════════════════════════════════════════════
# 1: WHALE BOTTOM — EXACT LIVE RULES
# ═══════════════════════════════════════════════════════
def strat_whale_bottom_exact(all_data, tp, sl, max_hold):
    """Exact live rules: whale≥0.50, RSI<25, green confirm, TP=3.5%, SL=1.5%"""
    signals = []
    for coin, data in all_data.items():
        c = np.array(data['close']); v = np.array(data['volume'])
        o_arr = np.array(data['open']); n = len(c)
        if n < 200: continue
        
        vol_avg = pd.Series(v).rolling(50).mean().values
        
        delta = pd.Series(c).diff()
        gain = delta.where(delta>0,0).rolling(14).mean().values
        loss = (-delta.where(delta<0,0)).rolling(14).mean().values
        rsi = np.where(loss>0, 100-(100/(1+gain/loss)), 50)
        
        for i in range(100, n-3):  # -3 for conf + entry candle
            if np.isnan(rsi[i]) or vol_avg[i] <= 0: continue
            
            whale = (v[i] - vol_avg[i]) / vol_avg[i]
            if whale < 0.50: continue
            if rsi[i] >= 25: continue
            
            # Green confirmation: candle i+1 close > open
            if c[i+1] <= o_arr[i+1]: continue
            
            # Entry on candle i+2 close (after confirmation)
            entry_idx = i + 2
            if entry_idx >= n: continue
            
            signals.append({'coin':coin, 'idx':entry_idx, 'entry':c[entry_idx],
                           'date':data['ts'][entry_idx]})
    
    return simulate(signals, all_data, tp, sl, max_hold)

# ═══════════════════════════════════════════════════════
# 2: RSI<30 + Prev Red + Green Confirm (15m version)
# ═══════════════════════════════════════════════════════
def strat_rsi30_prevred_15m(all_data, tp, sl, max_hold):
    """Daily winner adapted to 15m"""
    signals = []
    for coin, data in all_data.items():
        c = np.array(data['close']); o_arr = np.array(data['open']); n = len(c)
        if n < 200: continue
        
        delta = pd.Series(c).diff()
        gain = delta.where(delta>0,0).rolling(14).mean().values
        loss = (-delta.where(delta<0,0)).rolling(14).mean().values
        rsi = np.where(loss>0, 100-(100/(1+gain/loss)), 50)
        
        for i in range(100, n-3):
            if np.isnan(rsi[i]): continue
            if rsi[i] >= 30: continue  # RSI<30
            if c[i] >= o_arr[i]: continue  # red candle
            
            # Green confirmation next candle
            if c[i+1] <= o_arr[i+1]: continue
            
            entry_idx = i + 2
            if entry_idx >= n: continue
            signals.append({'coin':coin, 'idx':entry_idx, 'entry':c[entry_idx],
                           'date':data['ts'][entry_idx]})
    
    return simulate(signals, all_data, tp, sl, max_hold)

# ═══════════════════════════════════════════════════════
# 3: DUAL MOMENTUM 15m (RSI<25 + 96-candle drop > 10%)
# ═══════════════════════════════════════════════════════
def strat_dual_momentum_15m(all_data, tp, sl, max_hold):
    """15m version of the daily winner"""
    signals = []
    for coin, data in all_data.items():
        c = np.array(data['close']); o_arr = np.array(data['open']); n = len(c)
        if n < 200: continue
        
        delta = pd.Series(c).diff()
        gain = delta.where(delta>0,0).rolling(14).mean().values
        loss = (-delta.where(delta<0,0)).rolling(14).mean().values
        rsi = np.where(loss>0, 100-(100/(1+gain/loss)), 50)
        
        # 96-candle return (~1 day on 15m)
        ret_96 = pd.Series(c).pct_change(96).values * 100
        
        for i in range(150, n-3):
            if np.isnan(rsi[i]) or np.isnan(ret_96[i]): continue
            if rsi[i] >= 25: continue
            if ret_96[i] > -10: continue  # must've dropped 10%+ in 24h
            
            # Green confirmation
            if c[i+1] <= o_arr[i+1]: continue
            
            entry_idx = i + 2
            if entry_idx >= n: continue
            signals.append({'coin':coin, 'idx':entry_idx, 'entry':c[entry_idx],
                           'date':data['ts'][entry_idx]})
    
    return simulate(signals, all_data, tp, sl, max_hold)

# ═══════════════════════════════════════════════════════
# 4: SUPPORT BOUNCE 15m (96-candle low + reversal + RSI)
# ═══════════════════════════════════════════════════════
def strat_support_bounce_15m(all_data, tp, sl, max_hold):
    signals = []
    for coin, data in all_data.items():
        c = np.array(data['close']); l = np.array(data['low'])
        o_arr = np.array(data['open']); n = len(c)
        if n < 200: continue
        
        low96 = pd.Series(l).rolling(96).min().values
        
        delta = pd.Series(c).diff()
        gain = delta.where(delta>0,0).rolling(14).mean().values
        loss = (-delta.where(delta<0,0)).rolling(14).mean().values
        rsi = np.where(loss>0, 100-(100/(1+gain/loss)), 50)
        
        for i in range(150, n-3):
            if np.isnan(rsi[i]) or np.isnan(low96[i]): continue
            if low96[i] <= 0: continue
            
            near_low = abs(c[i] - low96[i]) / low96[i] < 0.01  # within 1%
            reversal = c[i] > o_arr[i] and c[i-1] < o_arr[i-1]  # red→green
            oversold = rsi[i] < 35
            
            if near_low and reversal and oversold:
                if c[i+1] <= o_arr[i+1]: continue  # green confirm
                entry_idx = i + 2
                if entry_idx >= n:
                    entry_idx = i+1
                signals.append({'coin':coin, 'idx':entry_idx, 'entry':c[entry_idx],
                               'date':data['ts'][entry_idx]})
    
    return simulate(signals, all_data, tp, sl, max_hold)

# ═══════════════════════════════════════════════════════
# 5: VOLUME SPIKE REVERSAL 15m
# ═══════════════════════════════════════════════════════
def strat_vol_spike_15m(all_data, tp, sl, max_hold):
    """Volume 3x avg + red candle + RSI<35"""
    signals = []
    for coin, data in all_data.items():
        c = np.array(data['close']); v = np.array(data['volume'])
        o_arr = np.array(data['open']); n = len(c)
        if n < 200: continue
        
        vol_avg = pd.Series(v).rolling(96).mean().values
        
        delta = pd.Series(c).diff()
        gain = delta.where(delta>0,0).rolling(14).mean().values
        loss = (-delta.where(delta<0,0)).rolling(14).mean().values
        rsi = np.where(loss>0, 100-(100/(1+gain/loss)), 50)
        
        for i in range(150, n-3):
            if np.isnan(rsi[i]) or vol_avg[i] <= 0: continue
            if v[i] < vol_avg[i] * 3: continue  # 3x volume spike
            if c[i] >= o_arr[i]: continue  # red candle
            if rsi[i] >= 35: continue
            
            if c[i+1] <= o_arr[i+1]: continue
            entry_idx = i + 2
            if entry_idx >= n: continue
            
            signals.append({'coin':coin, 'idx':entry_idx, 'entry':c[entry_idx],
                           'date':data['ts'][entry_idx]})
    
    return simulate(signals, all_data, tp, sl, max_hold)

# ═══════════════════════════════════════════════════════
# 6: BB OVERSOLD + GREEN CONFIRM 15m
# ═══════════════════════════════════════════════════════
def strat_bb_oversold_15m(all_data, tp, sl, max_hold):
    signals = []
    for coin, data in all_data.items():
        c = np.array(data['close']); o_arr = np.array(data['open']); n = len(c)
        if n < 200: continue
        
        sma20 = pd.Series(c).rolling(20).mean().values
        std20 = pd.Series(c).rolling(20).std().values
        bb_lower = sma20 - 2*std20
        
        delta = pd.Series(c).diff()
        gain = delta.where(delta>0,0).rolling(14).mean().values
        loss = (-delta.where(delta<0,0)).rolling(14).mean().values
        rsi = np.where(loss>0, 100-(100/(1+gain/loss)), 50)
        
        for i in range(100, n-3):
            if np.isnan(rsi[i]) or np.isnan(bb_lower[i]): continue
            if c[i] > bb_lower[i]: continue  # must be below lower BB
            if rsi[i] >= 30: continue
            
            if c[i+1] <= o_arr[i+1]: continue
            entry_idx = i + 2
            if entry_idx >= n: continue
            
            signals.append({'coin':coin, 'idx':entry_idx, 'entry':c[entry_idx],
                           'date':data['ts'][entry_idx]})
    
    return simulate(signals, all_data, tp, sl, max_hold)

# ═══════════════════════════════════════════════════════
# 7: CONSECUTIVE REDS + RSI + GREEN 15m
# ═══════════════════════════════════════════════════════
def strat_reds_bounce_15m(all_data, tp, sl, max_hold):
    """5+ consecutive red candles + RSI<25 + green confirm"""
    signals = []
    for coin, data in all_data.items():
        c = np.array(data['close']); o_arr = np.array(data['open']); n = len(c)
        if n < 200: continue
        
        delta = pd.Series(c).diff()
        gain = delta.where(delta>0,0).rolling(14).mean().values
        loss = (-delta.where(delta<0,0)).rolling(14).mean().values
        rsi = np.where(loss>0, 100-(100/(1+gain/loss)), 50)
        
        # Count consecutive reds
        red_streak = np.zeros(n, dtype=int)
        for i in range(1, n):
            if c[i] < o_arr[i]:
                red_streak[i] = red_streak[i-1] + 1
            else:
                red_streak[i] = 0
        
        for i in range(100, n-3):
            if np.isnan(rsi[i]): continue
            if red_streak[i] < 5: continue
            if rsi[i] >= 25: continue
            
            if c[i+1] <= o_arr[i+1]: continue
            entry_idx = i + 2
            if entry_idx >= n: continue
            
            signals.append({'coin':coin, 'idx':entry_idx, 'entry':c[entry_idx],
                           'date':data['ts'][entry_idx]})
    
    return simulate(signals, all_data, tp, sl, max_hold)

# ── Run All ─────────────────────────────────────────────
print("\n── Fetching 15m data for 198 coins ──")
all_data = fetch_15m()
print(f"   {len(all_data)} coins ready\n")

STRATEGIES = [
    ("🐋 Whale Bottom (TP3.5/SL1.5/MH24h)", strat_whale_bottom_exact, 0.035, 0.015, 96),
    ("🐋 Whale Bottom (TP5/SL2.5/MH24h)", strat_whale_bottom_exact, 0.05, 0.025, 96),
    ("🐋 Whale Bottom (TP7/SL3/MH48h)", strat_whale_bottom_exact, 0.07, 0.03, 192),
    ("RSI<30 + PrevRed", strat_rsi30_prevred_15m, 0.03, 0.015, 96),
    ("RSI<30 + PrevRed (TP5)", strat_rsi30_prevred_15m, 0.05, 0.025, 96),
    ("Dual Momentum 15m", strat_dual_momentum_15m, 0.05, 0.025, 96),
    ("Support Bounce 15m", strat_support_bounce_15m, 0.05, 0.025, 96),
    ("Volume Spike 3x", strat_vol_spike_15m, 0.05, 0.025, 96),
    ("BB Oversold Bounce", strat_bb_oversold_15m, 0.05, 0.025, 96),
    ("5+ Reds Bounce", strat_reds_bounce_15m, 0.05, 0.025, 96),
]

print(f"{'='*95}")
print(f"⏱️ 15-MINUTE BACKTEST — {BACKTEST_DAYS}-day Results")
print(f"{'='*95}")
print(f"{'Strategy':<35s} {'Trades':>5s} {'WR':>6s} {'Return':>7s} {'MaxDD':>7s} {'PF':>5s} {'TP':>4s} {'Avg':>6s}")
print(f"{'-'*80}")

for name, fn, tp, sl, mh in STRATEGIES:
    trades, final = fn(all_data, tp, sl, mh)
    print(f"  {summarize(name, trades, final)}")

print(f"\n✅ 15-minute backtest complete!")
