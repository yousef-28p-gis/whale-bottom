#!/usr/bin/env python3
"""
نسخ كامل — كل عملات بايننس — 3 أشهر
استراتيجية: حجم 2x + اختراق قمة 5
TP=0.7% SL=1.8% MAX=12h
"""
import ccxt, numpy as np, pandas as pd, json, os, time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

COMM = 0.2  # 0.2%
CAP = 1000
TP1_PCT = 0.7
SL_PCT = 1.8
MAX_BARS = 48
DATA_DIR = '/data/trading28/data/ea_replicate'
os.makedirs(DATA_DIR, exist_ok=True)

# Load coins
exchange = ccxt.binance({'timeout': 15000})
markets = exchange.load_markets()
coins = sorted([s.replace('/USDT','') for s in markets if s.endswith('/USDT') 
                and markets[s]['active'] and 'USDT' in s])
# Remove stablecoins and low-volume
skip = {'USDC','BUSD','DAI','TUSD','USDE','FDUSD','USDD','FRAX','LUSD','PYUSD','USDJ','RLUSD','XAUT','EUR','PAXG','BTC','ETH'}
coins = [c for c in coins if c not in skip]
print(f"🔍 {len(coins)} عملة | 3 أشهر | 15m")

# Fetch one coin's data
def fetch_coin(coin):
    cache_file = os.path.join(DATA_DIR, f'{coin}.json')
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)
    try:
        since = exchange.parse8601('2026-05-01T00:00:00Z')
        ohlcv = exchange.fetch_ohlcv(f'{coin}/USDT', '15m', since=since, limit=10000)
        if len(ohlcv) < 500: return None
        data = {'c': [float(o[4]) for o in ohlcv], 'h': [float(o[2]) for o in ohlcv],
                'l': [float(o[3]) for o in ohlcv], 'v': [float(o[5]) for o in ohlcv]}
        with open(cache_file, 'w') as f: json.dump(data, f)
        return data
    except:
        return None

# Run backtest on one coin
def backtest_coin(coin, data):
    c = np.array(data['c']); h = np.array(data['h'])
    l = np.array(data['l']); v = np.array(data['v'])
    n = len(c)
    
    avg_vol = pd.Series(v).rolling(20).mean().values
    
    trades = []
    for i in range(20, n-1):
        # Entry: volume > 2x avg + breaking 5-bar high + green candle
        if not (v[i] > avg_vol[i]*2.0 and c[i] > max(h[max(0,i-5):i]) and c[i] > c[i-1]):
            continue
        
        ep = c[i]
        tp1 = ep * (1 + TP1_PCT/100)
        sl = ep * (1 - SL_PCT/100)
        
        ex = et = None
        for j in range(i+1, min(i+MAX_BARS, n)):
            if l[j] <= sl: ex = sl; et = 'SL'; break
            elif h[j] >= tp1: ex = tp1; et = 'TP'; break
        if not ex: ex = c[min(i+MAX_BARS, n-1)]; et = 'TIME'
        
        pnl = (ex/ep - 1)*100 - COMM
        trades.append({'pnl': pnl, 'type': et})
    
    return trades

# Main
all_trades = []
done = 0
for coin in coins:
    data = fetch_coin(coin)
    if not data: continue
    
    trades = backtest_coin(coin, data)
    all_trades.extend(trades)
    done += 1
    if done % 20 == 0:
        print(f"  📊 {done}/{len(coins)} عملة | {len(all_trades)} صفقة...", flush=True)

print(f"\n✅ تم: {done} عملة | {len(all_trades)} صفقة")

if not all_trades:
    print("لا توجد صفقات!")
    exit()

# Stats
wins = [t for t in all_trades if t['pnl'] > 0]
losses = [t for t in all_trades if t['pnl'] <= 0]
tp_n = sum(1 for t in all_trades if t['type']=='TP')
sl_n = sum(1 for t in all_trades if t['type']=='SL')
tm_n = sum(1 for t in all_trades if t['type']=='TIME')

wr = len(wins)/len(all_trades)*100
avg_w = np.mean([t['pnl'] for t in wins]) if wins else 0
avg_l = np.mean([t['pnl'] for t in losses]) if losses else 0
rr = abs(avg_w/avg_l) if avg_l != 0 else 0

print(f"\n{'='*60}")
print(f"📊 نتائج النسخ — 3 أشهر — كل العملات")
print(f"{'='*60}")
print(f"📋 إجمالي الصفقات: {len(all_trades)}")
print(f"🟢 ربح: {len(wins)} | 🔴 خسارة: {len(losses)}")
print(f"📈 WR: {wr:.1f}%")
print(f"🎯 TP: {tp_n} | 🛑 SL: {sl_n} | ⏰ TIME: {tm_n}")
print(f"🟢 متوسط الربح: +{avg_w:.2f}%")
print(f"🔴 متوسط الخسارة: {avg_l:.2f}%")
print(f"📊 R:R: {rr:.2f}")

# Portfolio sim
curve = [CAP]
for t in sorted(all_trades, key=lambda x: x.get('_order', 0)) if '_order' in all_trades[0] else all_trades:
    sz = curve[-1] * 0.10
    curve.append(curve[-1] + sz * t['pnl']/100)

final = curve[-1]
net = (final/CAP - 1)*100
peak = np.maximum.accumulate(curve)
dd = np.min((curve-peak)/peak*100)

print(f"\n💼 محفظة: ${final:.0f} ({net:+.1f}%) | سحب: {dd:.1f}%")

# Compare with EA
print(f"\n{'─'*60}")
print(f"📊 مقارنة مع EA Free Signals:")
print(f"   EA: 208 صفقة | WR 78% | TP +0.7% | SL -1.8%")
print(f"   نسختنا: {len(all_trades)} صفقة | WR {wr:.0f}% | TP +{avg_w:.2f}% | SL {avg_l:.2f}%")
