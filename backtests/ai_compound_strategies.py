#!/usr/bin/env python3
"""
AI-INSPIRED COMPOUND STRATEGIES — Multi-indicator confirmation
Based on ChatGPT/AI-generated strategies from YouTube + Pine Script community
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

print(f"🤖 AI COMPOUND STRATEGIES — {len(all_data)} coins, 120 days\n")

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

def get_indicators(c, h, l, v):
    """Compute all indicators once per coin."""
    ind = {}
    n = len(c)
    
    # RSI(14)
    delta = pd.Series(c).diff()
    gain = delta.where(delta>0,0)
    loss = (-delta.where(delta<0,0))
    ind['rsi'] = (100 - (100/(1+gain.rolling(14).mean()/loss.rolling(14).mean()))).values
    
    # MACD
    ema12 = pd.Series(c).ewm(span=12).mean().values
    ema26 = pd.Series(c).ewm(span=26).mean().values
    ind['macd'] = ema12 - ema26
    ind['macd_signal'] = pd.Series(ind['macd']).ewm(span=9).mean().values
    ind['macd_hist'] = ind['macd'] - ind['macd_signal']
    
    # Bollinger
    sma20 = pd.Series(c).rolling(20).mean().values
    std20 = pd.Series(c).rolling(20).std().values
    ind['bb_upper'] = sma20 + 2*std20
    ind['bb_lower'] = sma20 - 2*std20
    ind['bb_width'] = 4*std20/sma20
    ind['bb_pos'] = (c - ind['bb_lower'])/(ind['bb_upper'] - ind['bb_lower'])
    
    # ADX(14)
    tr = np.maximum(h[1:]-l[1:], np.maximum(abs(h[1:]-c[:-1]), abs(l[1:]-c[:-1])))
    tr = np.insert(tr,0,h[0]-l[0])
    atr14 = pd.Series(tr).rolling(14).mean().values
    up = np.where((h[1:]-h[:-1])>(l[:-1]-l[1:]), np.maximum(h[1:]-h[:-1],0),0)
    up = np.insert(up,0,0)
    down = np.where((l[:-1]-l[1:])>(h[1:]-h[:-1]), np.maximum(l[:-1]-l[1:],0),0)
    down = np.insert(down,0,0)
    pdi = pd.Series(up).rolling(14).mean().values/atr14*100
    mdi = pd.Series(down).rolling(14).mean().values/atr14*100
    dx = np.abs(pdi-mdi)/(pdi+mdi)*100
    ind['adx'] = pd.Series(dx).rolling(14).mean().values
    
    # ATR%
    ind['atr_pct'] = atr14/c*100
    
    # EMA
    ind['ema20'] = pd.Series(c).ewm(span=20).mean().values
    ind['ema50'] = pd.Series(c).ewm(span=50).mean().values
    
    # Volume
    ind['vol_ratio'] = v/pd.Series(v).rolling(20).mean().values
    ind['pct'] = pd.Series(c).pct_change().values*100
    
    # Stochastic
    l14 = pd.Series(l).rolling(14).min().values
    h14 = pd.Series(h).rolling(14).max().values
    ind['stoch_k'] = (c-l14)/(h14-l14)*100
    ind['stoch_d'] = pd.Series(ind['stoch_k']).rolling(3).mean().values
    
    # Red streak
    red = (ind['pct'] < 0).astype(int)
    streak = [0]
    for i in range(1,len(red)): streak.append(streak[-1]+1 if red[i] else 0)
    ind['red_streak'] = np.array(streak)
    
    return ind

# ═══════════════════════════════════════════════════════
# 1: TRIPLE CONFIRMATION (ChatGPT classic)
# RSI<30 + MACD cross up + BB bottom touch + Volume>1.5x
# ═══════════════════════════════════════════════════════
def strat_triple_confirm(all_data, tp, sl, max_hold=7):
    signals = []
    for coin, data in all_data.items():
        if coin not in valid_coins: continue
        c = np.array(data['close']); h = np.array(data['high'])
        l = np.array(data['low']); v = np.array(data['volume']); n = len(c)
        if n < 60: continue
        ind = get_indicators(c, h, l, v)
        
        for i in range(50, n-1):
            if np.isnan(ind['rsi'][i]): continue
            # All 3 confirmations
            rsi_ok = ind['rsi'][i] < 30
            macd_ok = ind['macd'][i] > ind['macd_signal'][i] and ind['macd'][i-1] <= ind['macd_signal'][i-1]
            bb_ok = ind['bb_pos'][i] < 0.1  # near lower band
            vol_ok = ind['vol_ratio'][i] > 1.5
            
            score = sum([rsi_ok, macd_ok, bb_ok, vol_ok])
            if score >= 3:  # at least 3 of 4
                signals.append({'coin':coin, 'idx':i, 'entry':c[i], 'date':data['ts'][i]})
    return simulate(signals, all_data, tp, sl, max_hold)

# ═══════════════════════════════════════════════════════
# 2: MOMENTUM STACK (EMA + ADX + Volume)
# EMA20>EMA50 + ADX>25 + Vol>1.3x + MACD>0
# ═══════════════════════════════════════════════════════
def strat_momentum_stack(all_data, tp, sl, max_hold=7):
    signals = []
    for coin, data in all_data.items():
        if coin not in valid_coins: continue
        c = np.array(data['close']); h = np.array(data['high'])
        l = np.array(data['low']); v = np.array(data['volume']); n = len(c)
        if n < 70: continue
        ind = get_indicators(c, h, l, v)
        
        for i in range(60, n-1):
            if np.isnan(ind['adx'][i]): continue
            ema_ok = ind['ema20'][i] > ind['ema50'][i]
            adx_ok = ind['adx'][i] > 25
            vol_ok = ind['vol_ratio'][i] > 1.3
            macd_ok = ind['macd'][i] > 0
            # Price near EMA20 (pullback)
            pullback = abs(c[i] - ind['ema20'][i])/ind['ema20'][i] < 0.03
            
            score = sum([ema_ok, adx_ok, vol_ok, macd_ok, pullback])
            if score >= 4:
                signals.append({'coin':coin, 'idx':i, 'entry':c[i], 'date':data['ts'][i]})
    return simulate(signals, all_data, tp, sl, max_hold)

# ═══════════════════════════════════════════════════════
# 3: OVERSOLD REVERSAL SCORING (RSI+Stoch+BB+ATR+Volume)
# Score 0-5, enter when score >= 4
# ═══════════════════════════════════════════════════════
def strat_scoring_system(all_data, tp, sl, max_hold=7):
    signals = []
    for coin, data in all_data.items():
        if coin not in valid_coins: continue
        c = np.array(data['close']); h = np.array(data['high'])
        l = np.array(data['low']); v = np.array(data['volume']); n = len(c)
        if n < 60: continue
        ind = get_indicators(c, h, l, v)
        
        for i in range(50, n-1):
            if np.isnan(ind['rsi'][i]): continue
            
            score = 0
            if ind['rsi'][i] < 35: score += 1
            if ind['rsi'][i] < 25: score += 1  # extra for deep oversold
            if ind['stoch_k'][i] < 20: score += 1
            if ind['bb_pos'][i] < 0.2: score += 1
            if ind['pct'][i] < 0: score += 1  # red day
            if ind['red_streak'][i] >= 2: score += 1  # consecutive reds
            if ind['vol_ratio'][i] > 1.2: score += 1
            
            if score >= 4 and ind['pct'][i] < 0:  # must be red day
                signals.append({'coin':coin, 'idx':i+1, 'entry':c[i+1] if i+1<n else c[i],
                               'date':data['ts'][i]})
    return simulate(signals, all_data, tp, sl, max_hold)

# ═══════════════════════════════════════════════════════
# 4: VOLATILITY REGIME ADAPTIVE (ATR-based)
# High ATR regime: breakout strategy
# Low ATR regime: mean reversion (RSI oversold)
# ═══════════════════════════════════════════════════════
def strat_regime_adaptive(all_data, tp, sl, max_hold=7):
    signals = []
    for coin, data in all_data.items():
        if coin not in valid_coins: continue
        c = np.array(data['close']); h = np.array(data['high'])
        l = np.array(data['low']); v = np.array(data['volume']); n = len(c)
        if n < 80: continue
        ind = get_indicators(c, h, l, v)
        
        # ATR% 20-bar average
        atr_avg20 = pd.Series(ind['atr_pct']).rolling(20).mean().values
        high20 = pd.Series(h).rolling(20).max().values
        
        for i in range(60, n-1):
            if np.isnan(ind['adx'][i]) or np.isnan(atr_avg20[i]): continue
            
            high_vol = ind['atr_pct'][i] > atr_avg20[i] * 1.5
            
            if high_vol:
                # High volatility: breakout above 20-bar high + ADX > 20
                if c[i] > high20[i-1] and ind['adx'][i] > 20 and ind['vol_ratio'][i] > 1.3:
                    signals.append({'coin':coin, 'idx':i, 'entry':c[i], 'date':data['ts'][i]})
            else:
                # Low volatility: mean reversion RSI<25 + BB bottom
                if ind['rsi'][i] < 25 and ind['bb_pos'][i] < 0.15 and ind['pct'][i] < 0:
                    signals.append({'coin':coin, 'idx':i+1, 'entry':c[i+1] if i+1<n else c[i],
                                   'date':data['ts'][i]})
    return simulate(signals, all_data, tp, sl, max_hold)

# ═══════════════════════════════════════════════════════
# 5: MACD ZERO-LINE REVERSAL (ChatGPT viral strategy)
# MACD histogram turns positive from below zero + RSI<40 + Volume
# ═══════════════════════════════════════════════════════
def strat_macd_zero_reversal(all_data, tp, sl, max_hold=7):
    signals = []
    for coin, data in all_data.items():
        if coin not in valid_coins: continue
        c = np.array(data['close']); h = np.array(data['high'])
        l = np.array(data['low']); v = np.array(data['volume']); n = len(c)
        if n < 60: continue
        ind = get_indicators(c, h, l, v)
        
        for i in range(50, n-1):
            if np.isnan(ind['rsi'][i]): continue
            
            # MACD hist turns positive (was negative, now positive)
            hist_turn = ind['macd_hist'][i] > 0 and ind['macd_hist'][i-1] <= 0
            # MACD below zero (deep)
            macd_deep = ind['macd'][i] < -0.5 * np.std(ind['macd'][max(0,i-50):i+1]) if i>=50 else True
            
            if hist_turn and macd_deep and ind['rsi'][i] < 40:
                signals.append({'coin':coin, 'idx':i, 'entry':c[i], 'date':data['ts'][i]})
    return simulate(signals, all_data, tp, sl, max_hold)

# ═══════════════════════════════════════════════════════
# 6: BB SQUEEZE PRO (narrowing bands + volume + RSI)
# BB width at 6-month low + RSI<35 + Volume expansion
# ═══════════════════════════════════════════════════════
def strat_bb_squeeze_pro(all_data, tp, sl, max_hold=10):
    signals = []
    for coin, data in all_data.items():
        if coin not in valid_coins: continue
        c = np.array(data['close']); h = np.array(data['high'])
        l = np.array(data['low']); v = np.array(data['volume']); n = len(c)
        if n < 130: continue
        ind = get_indicators(c, h, l, v)
        
        bb_w_min120 = pd.Series(ind['bb_width']).rolling(120).min().values
        
        for i in range(130, n-1):
            if np.isnan(ind['rsi'][i]) or np.isnan(bb_w_min120[i]): continue
            
            squeeze = ind['bb_width'][i] <= bb_w_min120[i] * 1.05
            vol_expanding = ind['vol_ratio'][i] > 1.5
            
            if squeeze and vol_expanding and ind['rsi'][i] < 35:
                signals.append({'coin':coin, 'idx':i, 'entry':c[i], 'date':data['ts'][i]})
    return simulate(signals, all_data, tp, sl, max_hold)

# ═══════════════════════════════════════════════════════
# 7: EMA RIBBON COMPRESSION (multiple EMAs converging)
# 10/20/30/50 EMAs within 3% range + RSI<40 + breakout up
# ═══════════════════════════════════════════════════════
def strat_ema_ribbon(all_data, tp, sl, max_hold=10):
    signals = []
    for coin, data in all_data.items():
        if coin not in valid_coins: continue
        c = np.array(data['close']); h = np.array(data['high'])
        l = np.array(data['low']); v = np.array(data['volume']); n = len(c)
        if n < 70: continue
        ind = get_indicators(c, h, l, v)
        
        # Extra EMAs
        ema10 = pd.Series(c).ewm(span=10).mean().values
        ema30 = pd.Series(c).ewm(span=30).mean().values
        
        for i in range(55, n-1):
            if np.isnan(ind['rsi'][i]): continue
            
            emas = [ema10[i], ind['ema20'][i], ema30[i], ind['ema50'][i]]
            if any(np.isnan(e) for e in emas): continue
            
            ema_range = (max(emas) - min(emas)) / np.mean(emas)
            compression = ema_range < 0.03  # all EMAs within 3%
            
            # Breakout: price above all EMAs
            breakout = c[i] > max(emas) and c[i-1] <= max(emas)
            
            if compression and breakout and ind['rsi'][i] < 45:
                signals.append({'coin':coin, 'idx':i, 'entry':c[i], 'date':data['ts'][i]})
    return simulate(signals, all_data, tp, sl, max_hold)

# ═══════════════════════════════════════════════════════
# 8: DEEP OVERSOLD SNIPER (multiple timeframe oversold)
# Daily RSI<20 + Stoch<10 + MACD hist rising + red streak>=3
# ═══════════════════════════════════════════════════════
def strat_deep_oversold(all_data, tp, sl, max_hold=10):
    signals = []
    for coin, data in all_data.items():
        if coin not in valid_coins: continue
        c = np.array(data['close']); h = np.array(data['high'])
        l = np.array(data['low']); v = np.array(data['volume']); n = len(c)
        if n < 60: continue
        ind = get_indicators(c, h, l, v)
        
        for i in range(50, n-1):
            if np.isnan(ind['rsi'][i]): continue
            
            # Multiple extreme conditions
            if (ind['rsi'][i] < 20 and 
                ind['stoch_k'][i] < 10 and
                ind['red_streak'][i] >= 3 and
                ind['macd_hist'][i] > ind['macd_hist'][i-1]):  # hist improving
                
                signals.append({'coin':coin, 'idx':i+1, 'entry':c[i+1] if i+1<n else c[i],
                               'date':data['ts'][i]})
    return simulate(signals, all_data, tp, sl, max_hold)

# ── Run All ─────────────────────────────────────────────
STRATEGIES = [
    ("Triple Confirm (RSI+MACD+BB+Vol)", strat_triple_confirm),
    ("Momentum Stack (EMA+ADX+Vol+MACD)", strat_momentum_stack),
    ("Scoring System (7 signals)", strat_scoring_system),
    ("Regime Adaptive (ATR-based)", strat_regime_adaptive),
    ("MACD Zero-Line Reversal", strat_macd_zero_reversal),
    ("BB Squeeze Pro (120d)", strat_bb_squeeze_pro),
    ("EMA Ribbon Compression", strat_ema_ribbon),
    ("Deep Oversold Sniper", strat_deep_oversold),
]

TP_SL = [(0.05, 0.03, "TP5/SL3"), (0.10, 0.05, "TP10/SL5"), (0.15, 0.07, "TP15/SL7")]

for tp, sl, label in TP_SL:
    print(f"\n{'='*95}")
    print(f"📐 {label}")
    print(f"{'='*95}")
    print(f"{'Strategy':<35s} {'Trades':>5s} {'WR':>6s} {'Return':>7s} {'MaxDD':>7s} {'PF':>5s} {'TP':>4s} {'Avg':>6s}")
    print(f"{'-'*80}")
    
    for name, fn in STRATEGIES:
        trades, final = fn(all_data, tp, sl)
        print(f"  {summarize(name, trades, final)}")

print(f"\n✅ All AI compound strategies tested!")
