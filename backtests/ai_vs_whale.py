#!/usr/bin/env python3
"""
AI-STRATEGY SHOOTOUT vs WHALE BOTTOM
Using SAME 5-year 15m cache data (no invented formulas)
Exact rules from published sources
"""
import json, numpy as np, pandas as pd, os
from datetime import timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

CACHE_DIR = '/data/trading28/data/5year_halal'
COMM = 0.20
CAPITAL = 1000.0
MAX_POS = 2
POS_PCT = 50

with open('/data/trading28/config/shariah_coins.json') as f:
    config = json.load(f)
valid_coins = set(config['halal'] + config['halal2'])
BLACKLIST = {'QTUM','ZRO','IOTX','DYM','DGB','SAPIEN','XLM','EDU','BTC','INIT','PARTI','0G','ROBO','PYTH','ANKR'}

coin_files = sorted([f for f in os.listdir(CACHE_DIR) if f.endswith('.json')])

def load_coin(fname):
    sym = fname.replace('_15m.json','')
    if sym in BLACKLIST or sym not in valid_coins:
        return None, None
    with open(f'{CACHE_DIR}/{fname}') as f:
        data = json.load(f)
    df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.sort_values('ts').reset_index(drop=True)
    return sym, df

def portfolio_sim(trades_sorted, max_pos=2, pos_pct=50, max_h=6):
    """Identical to backtest_halal_clean.py portfolio simulation"""
    capital = CAPITAL; peak = CAPITAL; max_dd = 0.0
    active = []; skipped = 0; taken = 0; exec_trades = []
    
    for t in trades_sorted:
        dt = t['dt']
        still_active = []
        for exit_dt, cost, pnl_amt in active:
            if dt >= exit_dt:
                capital += cost + pnl_amt
            else:
                still_active.append((exit_dt, cost, pnl_amt))
        active = still_active
        
        if len(active) >= max_pos:
            skipped += 1; continue
        pos_size = capital * pos_pct / 100
        if capital < pos_size:
            skipped += 1; continue
        pnl_amt = pos_size * t['pnl'] / 100
        capital -= pos_size
        active.append((dt + timedelta(hours=max_h), pos_size, pnl_amt))
        taken += 1; exec_trades.append(t)
        
        equity = capital + sum(pc + pd for _, pc, pd in active)
        if equity > peak: peak = equity
        dd = (equity - peak) / peak * 100
        if dd < max_dd: max_dd = dd
    
    for _, cost, pnl_amt in active:
        capital += cost + pnl_amt
    
    return exec_trades, taken, skipped, capital, max_dd

def summary(name, exec_trades, taken, skipped, final_cap, max_dd, years=5):
    nets = [t['pnl'] for t in exec_trades]
    wins = sum(1 for n in nets if n > 0)
    wr = wins/taken*100 if taken else 0
    ret = (final_cap/CAPITAL-1)*100
    ann = (final_cap/CAPITAL)**(1/years)-1 if final_cap > 0 else -1
    print(f"\n{'='*60}")
    print(f"🤖 {name}")
    print(f"{'='*60}")
    print(f"📋 إشارات: {taken+skipped} | ✅ منفذة: {taken} | ⏭️ متخطية: {skipped}")
    print(f"📈 WR: {wr:.1f}% | 💼 ${CAPITAL:.0f} → ${final_cap:.0f} ({ret:+.1f}%)")
    print(f"📉 سحب: {max_dd:.2f}% | 📈 سنوي: {ann*100:+.1f}%")
    return {'name': name, 'trades': taken, 'wr': wr, 'return': ret, 'dd': max_dd, 'annual': ann*100}

# ═══════════════════════════════════════════════════════
# 1: RSI+MACD+3MA TREND (from Alpaca ChatGPT guide)
# ═══════════════════════════════════════════════════════
def strat_rsi_macd_trend():
    """RSI<30 bounce + MACD cross + 3 MA trend aligned"""
    all_trades = []
    
    for fname in coin_files:
        sym, df = load_coin(fname)
        if sym is None or len(df) < 500: continue
        
        close = df['close'].values; open_ = df['open'].values
        n = len(df)
        
        # RSI(14)
        delta = pd.Series(close).diff()
        gain = delta.where(delta>0,0).rolling(14).mean().values
        loss = (-delta.where(delta<0,0)).rolling(14).mean().values
        rsi = np.where(loss>0, 100-(100/(1+gain/loss)), 50)
        
        # MACD(12,26,9)
        ema12 = pd.Series(close).ewm(span=12).mean().values
        ema26 = pd.Series(close).ewm(span=26).mean().values
        macd = ema12 - ema26
        macd_signal = pd.Series(macd).ewm(span=9).mean().values
        
        # 3 MAs for trend
        sma20 = pd.Series(close).rolling(20).mean().values
        sma50 = pd.Series(close).rolling(50).mean().values
        sma200 = pd.Series(close).rolling(200).mean().values
        
        for i in range(250, n-10):
            if np.isnan(rsi[i]): continue
            
            # Entry: RSI bounce from <30 + MACD cross up within 5 bars + uptrend
            rsi_bounce = rsi[i] > 30 and rsi[i-1] < 30
            macd_cross = macd[i] > macd_signal[i] and macd[i-1] <= macd_signal[i-1]
            uptrend = sma20[i] > sma50[i] > sma200[i]
            
            if not (rsi_bounce and uptrend): continue
            if not macd_cross:
                # Check within 5 bars
                found = False
                for j in range(max(250,i-4), i+1):
                    if macd[j] > macd_signal[j] and macd[j-1] <= macd_signal[j-1]:
                        found = True; break
                if not found: continue
            
            ep = float(close[i])
            tp_p = ep * 1.035; sl_p = ep * 0.985  # TP=3.5%, SL=1.5%
            
            for k in range(i+1, min(i+24, n)):  # 6h on 15m
                cur = float(close[k])
                if cur >= tp_p: pnl = 3.5 - COMM; exit_ = 'TP'; break
                if cur <= sl_p: pnl = -1.5 - COMM; exit_ = 'SL'; break
            else:
                cur = float(close[min(i+24, n-1)])
                pnl = round((cur-ep)/ep*100-COMM,4); exit_ = 'TIME'
            
            all_trades.append({'dt': df['ts'].iloc[i], 'pnl': pnl, 'exit': exit_})
    
    all_trades.sort(key=lambda t: t['dt'])
    et, taken, skipped, cap, dd = portfolio_sim(all_trades)
    return summary("RSI+MACD+3MA Trend", et, taken, skipped, cap, dd)

# ═══════════════════════════════════════════════════════
# 2: FREQTRADE HYPEROPT STYLE — RSI+BB+Volume+Stoch
# ═══════════════════════════════════════════════════════
def strat_freqtrade_style():
    """Multi-signal scoring: RSI<30 + BB lower + Stoch<20 + Vol>1.5x"""
    all_trades = []
    
    for fname in coin_files:
        sym, df = load_coin(fname)
        if sym is None or len(df) < 500: continue
        
        close = df['close'].values; open_ = df['open'].values
        high = df['high'].values; low = df['low'].values
        vol = df['volume'].values; n = len(df)
        
        # RSI
        delta = pd.Series(close).diff()
        gain = delta.where(delta>0,0).rolling(14).mean().values
        loss = (-delta.where(delta<0,0)).rolling(14).mean().values
        rsi = np.where(loss>0, 100-(100/(1+gain/loss)), 50)
        
        # BB
        sma20 = pd.Series(close).rolling(20).mean().values
        std20 = pd.Series(close).rolling(20).std().values
        bb_lower = sma20 - 2*std20
        
        # Stoch
        l14 = pd.Series(low).rolling(14).min().values
        h14 = pd.Series(high).rolling(14).max().values
        stoch = (close - l14) / (h14 - l14) * 100
        
        # Volume
        vol_avg = pd.Series(vol).rolling(20).mean().values
        
        for i in range(200, n-10):
            if np.isnan(rsi[i]): continue
            
            score = 0
            if rsi[i] < 30: score += 1
            if close[i] < bb_lower[i]: score += 1
            if stoch[i] < 20: score += 1
            if vol[i] > vol_avg[i] * 1.5: score += 1
            
            if score < 3: continue  # need 3/4 signals
            
            ep = float(close[i])
            tp_p = ep * 1.035; sl_p = ep * 0.985
            
            for k in range(i+1, min(i+24, n)):
                cur = float(close[k])
                if cur >= tp_p: pnl = 3.5 - COMM; exit_ = 'TP'; break
                if cur <= sl_p: pnl = -1.5 - COMM; exit_ = 'SL'; break
            else:
                cur = float(close[min(i+24, n-1)])
                pnl = round((cur-ep)/ep*100-COMM,4); exit_ = 'TIME'
            
            all_trades.append({'dt': df['ts'].iloc[i], 'pnl': pnl, 'exit': exit_})
    
    all_trades.sort(key=lambda t: t['dt'])
    et, taken, skipped, cap, dd = portfolio_sim(all_trades)
    return summary("Freqtrade Style (RSI+BB+Stoch+Vol)", et, taken, skipped, cap, dd)

# ═══════════════════════════════════════════════════════
# 3: MARTIN GALE REVERSAL (popular AI strategy claim)
# RSI<15 + LLV 50-bar + Volume climax
# ═══════════════════════════════════════════════════════
def strat_deep_reversal():
    """Extreme oversold only: RSI<15 + 50-bar low + volume 3x"""
    all_trades = []
    
    for fname in coin_files:
        sym, df = load_coin(fname)
        if sym is None or len(df) < 500: continue
        
        close = df['close'].values; n = len(df)
        low = df['low'].values; vol = df['volume'].values
        
        delta = pd.Series(close).diff()
        gain = delta.where(delta>0,0).rolling(14).mean().values
        loss = (-delta.where(delta<0,0)).rolling(14).mean().values
        rsi = np.where(loss>0, 100-(100/(1+gain/loss)), 50)
        
        llv50 = pd.Series(low).rolling(50).min().values
        vol_avg = pd.Series(vol).rolling(50).mean().values
        
        for i in range(200, n-10):
            if np.isnan(rsi[i]): continue
            if rsi[i] >= 15: continue
            if low[i] > llv50[i] * 1.01: continue  # not at 50-bar low
            if vol[i] < vol_avg[i] * 3: continue  # no volume climax
            
            ep = float(close[i])
            tp_p = ep * 1.05; sl_p = ep * 0.97  # wider because extreme
            
            for k in range(i+1, min(i+48, n)):  # 12h
                cur = float(close[k])
                if cur >= tp_p: pnl = 5.0 - COMM; exit_ = 'TP'; break
                if cur <= sl_p: pnl = -3.0 - COMM; exit_ = 'SL'; break
            else:
                cur = float(close[min(i+48, n-1)])
                pnl = round((cur-ep)/ep*100-COMM,4); exit_ = 'TIME'
            
            all_trades.append({'dt': df['ts'].iloc[i], 'pnl': pnl, 'exit': exit_})
    
    all_trades.sort(key=lambda t: t['dt'])
    et, taken, skipped, cap, dd = portfolio_sim(all_trades)
    return summary("Deep Reversal (RSI<15+LLV+Vol3x)", et, taken, skipped, cap, dd)

# ═══════════════════════════════════════════════════════
# 4: AI MOMENTUM — EMA Ribbon + ADX + Volume
# ═══════════════════════════════════════════════════════
def strat_ai_momentum():
    """EMA 9/21/50 aligned + ADX>25 + Volume 1.5x"""
    all_trades = []
    
    for fname in coin_files:
        sym, df = load_coin(fname)
        if sym is None or len(df) < 500: continue
        
        close = df['close'].values; high = df['high'].values
        low = df['low'].values; vol = df['volume'].values; n = len(df)
        
        ema9 = pd.Series(close).ewm(span=9).mean().values
        ema21 = pd.Series(close).ewm(span=21).mean().values
        ema50 = pd.Series(close).ewm(span=50).mean().values
        
        # ADX(14)
        tr = np.maximum(high[1:]-low[1:], np.maximum(abs(high[1:]-close[:-1]), abs(low[1:]-close[:-1])))
        tr = np.insert(tr,0,high[0]-low[0])
        atr14 = pd.Series(tr).rolling(14).mean().values
        up = np.where((high[1:]-high[:-1])>(low[:-1]-low[1:]), np.maximum(high[1:]-high[:-1],0),0)
        up = np.insert(up,0,0)
        down = np.where((low[:-1]-low[1:])>(high[1:]-high[:-1]), np.maximum(low[:-1]-low[1:],0),0)
        down = np.insert(down,0,0)
        pdi = pd.Series(up).rolling(14).mean().values/atr14*100
        mdi = pd.Series(down).rolling(14).mean().values/atr14*100
        dx = np.abs(pdi-mdi)/(pdi+mdi)*100
        adx = pd.Series(dx).rolling(14).mean().values
        
        vol_avg = pd.Series(vol).rolling(20).mean().values
        
        for i in range(200, n-10):
            if np.isnan(adx[i]): continue
            ribbon = ema9[i] > ema21[i] > ema50[i]  # all aligned
            if not ribbon: continue
            if adx[i] < 25: continue
            if vol[i] < vol_avg[i] * 1.5: continue
            
            ep = float(close[i])
            tp_p = ep * 1.035; sl_p = ep * 0.985
            
            for k in range(i+1, min(i+24, n)):
                cur = float(close[k])
                if cur >= tp_p: pnl = 3.5 - COMM; exit_ = 'TP'; break
                if cur <= sl_p: pnl = -1.5 - COMM; exit_ = 'SL'; break
            else:
                cur = float(close[min(i+24, n-1)])
                pnl = round((cur-ep)/ep*100-COMM,4); exit_ = 'TIME'
            
            all_trades.append({'dt': df['ts'].iloc[i], 'pnl': pnl, 'exit': exit_})
    
    all_trades.sort(key=lambda t: t['dt'])
    et, taken, skipped, cap, dd = portfolio_sim(all_trades)
    return summary("AI Momentum (EMA Ribbon+ADX+Vol)", et, taken, skipped, cap, dd)

# ═══════════════════════════════════════════════════════
# 5: CHATGPT CLASSIC — RSI Divergence + MACD + Support
# ═══════════════════════════════════════════════════════
def strat_chatgpt_classic():
    """RSI bullish divergence + MACD turning + near support"""
    all_trades = []
    
    for fname in coin_files:
        sym, df = load_coin(fname)
        if sym is None or len(df) < 500: continue
        
        close = df['close'].values; n = len(df)
        low = df['low'].values
        
        delta = pd.Series(close).diff()
        gain = delta.where(delta>0,0).rolling(14).mean().values
        loss = (-delta.where(delta<0,0)).rolling(14).mean().values
        rsi = np.where(loss>0, 100-(100/(1+gain/loss)), 50)
        
        ema12 = pd.Series(close).ewm(span=12).mean().values
        ema26 = pd.Series(close).ewm(span=26).mean().values
        macd = ema12 - ema26
        
        low50 = pd.Series(low).rolling(50).min().values
        
        for i in range(250, n-10):
            if np.isnan(rsi[i]): continue
            if rsi[i] > 40: continue  # not oversold zone
            
            # RSI divergence: price lower low, RSI higher low in last 10 bars
            price_ll = False; rsi_hl = False
            for j in range(i-10, i-2):
                if j < 200: continue
                if close[i] < close[j] and rsi[i] > rsi[j]:
                    price_ll = True; rsi_hl = True; break
            
            if not (price_ll and rsi_hl): continue
            
            # MACD turning up
            if macd[i] <= macd[i-1]: continue
            if macd[i-1] <= macd[i-2]: continue  # 2 bars turning
            
            # Near support
            if low[i] > low50[i] * 1.02: continue
            
            ep = float(close[i])
            tp_p = ep * 1.035; sl_p = ep * 0.985
            
            for k in range(i+1, min(i+24, n)):
                cur = float(close[k])
                if cur >= tp_p: pnl = 3.5 - COMM; exit_ = 'TP'; break
                if cur <= sl_p: pnl = -1.5 - COMM; exit_ = 'SL'; break
            else:
                cur = float(close[min(i+24, n-1)])
                pnl = round((cur-ep)/ep*100-COMM,4); exit_ = 'TIME'
            
            all_trades.append({'dt': df['ts'].iloc[i], 'pnl': pnl, 'exit': exit_})
    
    all_trades.sort(key=lambda t: t['dt'])
    et, taken, skipped, cap, dd = portfolio_sim(all_trades)
    return summary("ChatGPT Classic (RSI Div+MACD+Support)", et, taken, skipped, cap, dd)

# ── Run all ─────────────────────────────────────────────
print("🤖 AI STRATEGIES vs WHALE BOTTOM — 5-Year 15m Backtest")
print("=" * 60)
print(f"Using SAME cache: {CACHE_DIR}")
print(f"198 halal coins, 5 years, 15-minute candles")

results = []
results.append(strat_rsi_macd_trend())
results.append(strat_freqtrade_style())
results.append(strat_deep_reversal())
results.append(strat_ai_momentum())
results.append(strat_chatgpt_classic())

# Whale bottom baseline
print(f"\n{'='*60}")
print(f"🐋 Whale Bottom (baseline — from backtest_halal_clean.py)")
print(f"{'='*60}")
print(f"📈 WR: 77.1% | 💼 $1000 → $38026 (+3702.6%)")
print(f"📉 سحب: -4.29% | 📈 سنوي: +107.0%")
results.append({'name': '🐋 Whale Bottom', 'trades': 1062, 'wr': 77.1, 'return': 3702.6, 'dd': -4.29, 'annual': 107.0})

# ── Comparison table ──
print(f"\n{'='*80}")
print(f"🏆 FINAL COMPARISON — 5-Year 15m")
print(f"{'='*80}")
print(f"{'Strategy':<35s} {'Trades':>6s} {'WR':>6s} {'Return':>8s} {'MaxDD':>7s} {'Annual':>7s}")
print(f"{'-'*80}")
for r in sorted(results, key=lambda x: -x['return']):
    print(f"{r['name']:<35s} {r['trades']:>6d} {r['wr']:>5.1f}% {r['return']:>+7.1f}% {r['dd']:>6.2f}% {r['annual']:>+6.1f}%")

print(f"\n✅ All AI strategies tested!")
